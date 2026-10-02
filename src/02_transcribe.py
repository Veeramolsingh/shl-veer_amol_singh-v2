"""Step 2 - Speech-to-text with Whisper (faster-whisper / CTranslate2).

For every train and test clip we store:
  * the transcript (decoded as English) with word-level timestamps and confidences,
  * Whisper's language-ID probability for English - many label-0 clips are not (English) speech,
  * segment decoding statistics (avg log-prob, no-speech prob, compression ratio).

A disfluent `initial_prompt` nudges Whisper to keep fillers and false starts instead of
producing a "cleaned-up" transcript - Whisper's language model otherwise silently fixes
some of the very grammar errors we want to measure.

Results are appended line by line to features/transcripts.jsonl, so the script can be
stopped (Ctrl+C) and restarted; finished files are skipped.
"""
import argparse
import json

from tqdm import tqdm

from common import SR, TRANSCRIPTS, audio_path, expose_cuda_dlls, load_audio, load_meta, load_transcripts

PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. So, uh, I- I was going there."


def language_probs(model, audio):
    """Whisper language ID on the first 30 s -> (P(English), top language, P(top language))."""
    try:
        _, _, probs = model.detect_language(audio)
    except (AttributeError, TypeError):  # older faster-whisper: detection happens inside transcribe()
        _, info = model.transcribe(audio[: 30 * SR], language=None)
        probs = info.all_language_probs or []
    probs = dict(probs)
    top = max(probs, key=probs.get) if probs else "en"
    return float(probs.get("en", 0.0)), top, float(probs.get(top, 0.0))


def transcribe_one(model, audio, beam_size):
    en_prob, top_lang, top_prob = language_probs(model, audio)
    segments, info = model.transcribe(
        audio,
        language="en",                     # force English: accented English is often mis-detected
        beam_size=beam_size,
        word_timestamps=True,              # needed for pause / speech-rate features
        condition_on_previous_text=False,  # prevents repetition loops across segments
        initial_prompt=PROMPT,
        vad_filter=True,                   # skip silence -> no hallucinated text on silent clips
    )
    segs, words = [], []
    for s in segments:
        segs.append({
            "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 4), "no_speech_prob": round(s.no_speech_prob, 4),
            "compression_ratio": round(s.compression_ratio, 4),
        })
        words += [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2), "p": round(w.probability, 4)}
                  for w in (s.words or [])]
    return {
        "text": " ".join(s["text"] for s in segs).strip(),
        "duration": round(len(audio) / SR, 2),
        "speech_duration": round(float(getattr(info, "duration_after_vad", 0.0) or 0.0), 2),
        "en_prob": round(en_prob, 4), "top_lang": top_lang, "top_lang_prob": round(top_prob, 4),
        "segments": segs, "words": words,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo", help="faster-whisper model name")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only transcribe N files (smoke test)")
    args = ap.parse_args()

    if args.device == "cuda":
        expose_cuda_dlls()
    from faster_whisper import WhisperModel

    compute_type = "int8_float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    meta = load_meta()
    done = set(load_transcripts(missing_ok=True))
    todo = meta[~meta["key"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already transcribed, {len(todo)} to go (model={args.model}, device={args.device})")

    with open(TRANSCRIPTS, "a", encoding="utf-8") as fout:
        for row in tqdm(todo.itertuples(), total=len(todo), desc="transcribing"):
            rec = transcribe_one(model, load_audio(audio_path(row.split, row.filename)), args.beam)
            fout.write(json.dumps({"key": row.key, **rec}) + "\n")
            fout.flush()

    if args.limit:
        print("\nSample transcripts:")
        for rec in list(load_transcripts().values())[: args.limit]:
            print(f"- {rec['key']}  en_prob={rec['en_prob']}\n  {rec['text'][:300]}")


if __name__ == "__main__":
    main()
