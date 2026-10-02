# SHL Grammar Scoring Engine (second submission)

Veer Amol Singh (veer.amol636@gmail.com)

The report, code and results are in veer_amol_singh_SHL_v2.ipynb (already executed). It includes the training RMSE, the cross-validated RMSE and the plots.

Public leaderboard: 0.3400 with outputs/v2/submission_v2.csv (first submission was 0.3540).

| | RMSE |
|---|---|
| Training data | 0.2251 |
| 5-fold cross-validation | 0.4817 |
| Baseline (predict the mean) | 1.0137 |

## What changed from the first submission

- The test set is mostly 45 s clips (56%) while train is mostly 60 s clips. These are separate recording batches, so every model gets a batch flag as input.
- The 37 label-0 clips are a separate batch that does not appear in the test set, so the text models and the stacker are trained without them. Predictions are clipped to [2, 5].
- New models: ridge on Whisper encoder embeddings (audio) and ridge on Qwen2.5-1.5B transcript embeddings.
- Grammar error types (articles, prepositions, verb forms and so on) added to the hand-made features.

## Files

- veer_amol_singh_SHL_v2.ipynb: notebook with the report
- src/: scripts for the base features (transcripts, WavLM, grammar models)
- src_v2/: scripts for the second submission (new features, ensemble, leaderboard probes)
- outputs/v2/submission_v2.csv: the submitted predictions
- outputs/v2/cv_report_v2.json: CV numbers from the script run
- outputs/v2/*.png: plots from the notebook

The dataset is private, so it is not included.

## Running

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Put the dataset in data/Dataset_Final (train.csv, test.csv, train/, test/). Then either open the notebook, set RUN_HEAVY = True and run all cells, or run the scripts:

```
python src/01_prepare_data.py
python src/02_transcribe.py
python src/03_audio_features.py
python src/04_text_features.py
python src_v2/run_all.py
```

Everything was run on a laptop with an RTX 3050 (4 GB). The full run takes about 3 hours, mostly the GPU feature extraction.
