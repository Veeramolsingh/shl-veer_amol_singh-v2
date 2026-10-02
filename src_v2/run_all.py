r"""Run the whole v2 pipeline (steps 1-5) in THIS terminal, one step after the other.

    .\.venv\Scripts\python.exe src_v2\run_all.py

* Live output stays on your screen (progress bars included) and is also written to outputs/v2/run_log.txt.
* A step whose output file already exists is skipped, so the script can be re-run any time (step 5 always runs).
* GPU steps wait while another v2 GPU job (e.g. step 1 started by hand) is still running.
* If a step fails it does not exit: it prints the error context, writes outputs/v2/run_status.txt and WAITS until
  a .py file in src/ or src_v2/ is edited, then retries that step (max 5 attempts). A CUDA out-of-memory error
  is retried immediately with --batch-size 4 (DeBERTa step).
"""
import codecs
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common_v2 import DEBERTA_FILE, ERRTYPES_FILE, QWEN_FILE, V2_OUT, WHISPER_FILE

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src_v2"
LOG, STATUS = V2_OUT / "run_log.txt", V2_OUT / "run_status.txt"
FINAL = V2_OUT / "submission_v2.csv"  # shown when everything is done (the v3 runner overrides this)
MAX_TRIES = 5
FIX_WAIT_S = 4 * 3600

# (name, script, args, output file that marks the step as done - None = always run, uses the GPU?)
STEPS = [
    ("1/5 Whisper encoder embeddings", "01_whisper_encoder.py", [], WHISPER_FILE, True),
    ("2/5 Qwen transcript embeddings", "02_llm_embeddings.py", [], QWEN_FILE, True),
    ("3/5 Grammar-error types", "03_error_types.py", [], ERRTYPES_FILE, False),
    ("4/5 DeBERTa (no label-0, 3 seeds)", "04_finetune_deberta.py", [], DEBERTA_FILE, True),
    ("5/5 Ensemble + submission", "05_train_ensemble.py", ["--grouped-check"], None, False),
]


def say(msg, status=None):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(f"\n{line}", flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{line}\n")
    if status:
        STATUS.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {status}: {msg}\n", encoding="utf-8")


def gpu_jobs_running():
    """Command lines of v2 GPU scripts (steps 1, 2, 4) started outside this runner."""
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001 - if we cannot tell, assume the GPU is free
        return []
    return [ln for ln in out.splitlines() if re.search(r"src_v[23][\\/]0[124]_", ln)]


def code_stamp():
    files = list(V2.glob("*.py")) + list((ROOT / "src").glob("*.py"))
    return max(f.stat().st_mtime for f in files)


def run(script, args):
    """Run one script, mirror its output to the screen and to the log; -> (exit code, last ~6000 characters)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    proc = subprocess.Popen([sys.executable, str(V2 / script), *args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    with open(LOG, "a", encoding="utf-8") as log:
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            sys.stdout.write(text)
            sys.stdout.flush()
            log.write(text.replace("\r", "\n"))
            log.flush()
            tail = (tail + text)[-6000:]
    return proc.wait(), tail


def wait_for_fix(since):
    deadline = time.time() + FIX_WAIT_S
    while time.time() < deadline:
        time.sleep(5)
        if code_stamp() > since:
            time.sleep(3)  # let a multi-file edit finish
            return True
    return False


def main():
    sys.stdout.reconfigure(errors="replace")
    t_start = time.time()
    say(f"v2 pipeline started (python {sys.version.split()[0]}); log: {LOG}", "RUNNING")
    durations = {}
    for name, script, args, out_file, uses_gpu in STEPS:
        if out_file is not None and out_file.exists():
            say(f"[{name}] output already exists - skipping", "RUNNING")
            continue
        if uses_gpu:
            waited = False
            while gpu_jobs_running():
                if not waited:
                    say(f"[{name}] another v2 GPU job is still running - waiting for it to finish ...", "WAITING")
                    waited = True
                time.sleep(15)
            if out_file is not None and out_file.exists():  # the job we waited for produced it
                say(f"[{name}] output was produced meanwhile - skipping", "RUNNING")
                continue
        args = list(args)
        for attempt in range(1, MAX_TRIES + 1):
            stamp = code_stamp()
            say(f"[{name}] attempt {attempt}: python src_v2/{script} {' '.join(args)}", "RUNNING")
            t0 = time.time()
            code, tail = run(script, args)
            ok = code == 0 and (out_file is None or out_file.exists())
            if ok:
                durations[name] = time.time() - t0
                say(f"[{name}] done in {durations[name] / 60:.1f} min", "RUNNING")
                break
            if "out of memory" in tail.lower() and "--batch-size" not in args and script.startswith("04"):
                say(f"[{name}] CUDA out of memory - retrying with --batch-size 4", "RUNNING")
                args += ["--batch-size", "4"]
                continue
            say(f"[{name}] FAILED (exit code {code}). Waiting for a fix (edit any .py file in src_v2/ to retry) ...",
                "FAILED")
            print("---- last output ----\n" + tail[-1500:] + "\n---------------------", flush=True)
            if attempt == MAX_TRIES or not wait_for_fix(stamp):
                say(f"[{name}] giving up", "GAVE_UP")
                sys.exit(1)
            say(f"[{name}] code changed - retrying", "RUNNING")

    total = (time.time() - t_start) / 60
    say(f"ALL DONE in {total:.1f} min. Submit: {FINAL}", "DONE")
    print("\nStep timings (min): " + ", ".join(f"{k.split()[0]} {v / 60:.1f}" for k, v in durations.items()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("stopped by user (Ctrl+C)", "STOPPED")
        sys.exit(130)
