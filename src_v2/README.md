# src_v2

Scripts for the second submission. They build on the features made by the scripts in src/ (transcripts, WavLM, CoLA, grammar correction, text features), so run src/01 to src/04 first. New features go to features/v2/ and results to outputs/v2/.

| Step | Command | Output |
|---|---|---|
| 1 | python src_v2/01_whisper_encoder.py | Whisper encoder embeddings (about 8 min on a 4 GB GPU) |
| 2 | python src_v2/02_llm_embeddings.py | Qwen2.5-1.5B transcript embeddings (about 3 min) |
| 3 | python src_v2/03_error_types.py | grammar error types (seconds, CPU) |
| 4 | python src_v2/04_finetune_deberta.py | DeBERTa without the label-0 clips (about 27 min) |
| 5 | python src_v2/05_train_ensemble.py --grouped-check | outputs/v2/submission_v2.csv and cv_report_v2.json |
| 6 (optional) | python src_v2/06_lb_probes.py make | probe files used to check the leaderboard bias |

Steps 1 to 4 are independent of each other and step 5 needs all four. python src_v2/run_all.py runs steps 1 to 5 in order and skips steps whose output already exists.

## What changed from the first submission

- Batch flag: 45 s clips are a separate recording batch (56% of the test set, 17% of train) and score lower on the same prompts. Every model gets is_45s and is_other as inputs.
- The 37 label-0 clips are not used to train the text models or the stacker, since none of the test clips look like them. This takes DeBERTa from 0.636 to 0.584 RMSE.
- New models: ridge on Qwen2.5-1.5B embeddings and ridge on Whisper encoder embeddings.
- Grammar error types (articles, prepositions, verb forms, repairs) added to the hand-made features.
- One non-negative stacker fitted on clips with score above 0, and predictions clipped to [2, 5].

## Results

Test-weighted RMSE, with each clip weighted to match the test set's batch mix.

| | first submission | this submission |
|---|---|---|
| 5-fold CV | 0.5355 | 0.5110 |
| 6 splits with unseen prompts (no DeBERTa) | 0.5625 | 0.5343 |

The new version was better on all 6 unseen-prompt splits.
