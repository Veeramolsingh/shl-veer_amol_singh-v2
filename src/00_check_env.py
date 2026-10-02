"""Step 0 - Check that PyTorch sees the GPU and that faster-whisper can run on it."""
import platform
import sys

import numpy as np

from common import DATA_DIR, expose_cuda_dlls


def main():
    print(f"Python      : {sys.version.split()[0]} ({platform.system()})")

    import torch

    print(f"PyTorch     : {torch.__version__}")
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False -> reinstall torch from the cu128 index (see README).")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU         : {props.name}, {props.total_memory / 2**30:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")

    expose_cuda_dlls()
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", device="cuda", compute_type="int8_float16")
        segments, _ = model.transcribe(np.zeros(2 * 16_000, dtype=np.float32))
        list(segments)
        print("[OK]   faster-whisper runs on the GPU")
    except Exception as e:  # noqa: BLE001 - report any failure to the user
        print(f"[FAIL] faster-whisper on GPU: {e!r}")
        print("       Fallback: python src/02_transcribe.py --device cpu --model small  (slower, less accurate)")

    print(f"Dataset     : {'found' if (DATA_DIR / 'train.csv').exists() else 'not extracted yet (run 01_prepare_data.py)'}")


if __name__ == "__main__":
    main()
