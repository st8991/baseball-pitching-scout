# baseball-pitching-scout (PitchScout)

Can computer vision steal a pitcher's signs? This is a per-pitcher scouting tool, with a first case study on **Mason Miller's reported glove tell**. Type in a pitcher, and it builds a report with:

1. **Tendencies:** pitch mix overall, by count, by batter side, and what follows each pitch.
2. **Stats model (supervised):** LightGBM on the game situation known before the pitch.
3. **Vision layer (supervised):** MediaPipe Pose keypoints of the pitcher from the Baseball Savant clip, **cut off before release**, combined with the stats model.
4. **Vision layer (unsupervised):** PCA plus a Gaussian mixture that clusters deliveries without labels, then gets checked against the real pitch types.
5. **What the vision model is looking at:** body-part × time-before-release heatmaps (TreeSHAP contributions), tells by pitch type, skeleton overlays on example clips, a 2D map of deliveries, and the average skeleton per cluster.

## Result: Mason Miller glove tell

Run on real data on Nov 5, 2026: 474 usable Savant clips from 2024-25 (FF 228, SL 213, CH 33), glove features taken only from frames before release.

| Comparison | Glove pose AUC | p (500 perms) | Body control AUC | Glove ARI |
|---|---|---|---|---|
| FF vs SL+CH | 0.599 | 0.002 | 0.531 | 0.010 |
| FF vs SL | 0.598 | 0.004 | 0.564 | 0.006 |
| FF vs CH | 0.413 | 0.852 | 0.364 | -0.008 |

**Weak evidence of a fastball-vs-slider glove tell.** Supervised, the glove beats chance and the body-only control, but AUC 0.60 is not a usable tell. Unsupervised, the glove clusters don't line up with pitch type at all (ARI ≈ 0). There is no fastball-vs-changeup signal (only 33 changeup clips). **Robustness caveat:** the strongest glove differences come from MediaPipe's glove-hand finger points (spread, thumb opening, tilt), which are mostly guesses when the hand is inside a glove. With those dropped and only wrist and elbow features kept, FF vs SL falls to AUC 0.576 (p 0.024), barely above the body-only control (0.564), and FF vs SL+CH to 0.550 (p 0.056). So the evidence that the *glove itself* gives the pitch away is weaker still. Filtering finger points by visibility is the natural next step.

Details, the season split and caveats are at the end of `notebooks/05_mason_miller_glove_tell.ipynb`.

## Notebooks

| Notebook | What it covers |
|---|---|
| `notebooks/01_data_and_tendencies.ipynb` | Statcast pull, pitch mix by count / batter side / previous pitch, mix over time |
| `notebooks/02_stats_model.ipynb` | Supervised stats baseline on held-out games, feature importance, calibration |
| `notebooks/03_pose_extraction.ipynb` | Savant clips, MediaPipe pose, release detection, pre-release window, overlays |
| `notebooks/04_supervised_vs_unsupervised.ipynb` | Head-to-head on the same clips and folds, cluster alignment, attention heatmaps |
| `notebooks/05_mason_miller_glove_tell.ipynb` | The glove tell hypothesis test: supervised AUC + permutation p-value, unsupervised ARI/NMI, body-only control, set vs windup, by season |

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Every notebook reads `Config(pitcher="Mason Miller", ...)` in its first cell. Set `PITCHSCOUT_DEMO=1` to run the notebooks offline on the committed synthetic sample (`tests/fixtures/sample`). Demo mode only checks that the code runs; its numbers describe no real pitcher.

## Glove tell test

`pitchscout/glove.py` uses two kinds of glove evidence, both taken only from frames before release:

- **Pose:** the glove-hand wrist and MediaPipe hand points (index, pinky, thumb). From these it measures glove height, angle, spread, thumb opening, distance to the chest, and elbow flare, separately in the **set** (before leg lift) and the **windup**.
- **Pixels:** crops around the glove, summarized with HOG plus color histograms and compressed with PCA. It also writes the mean crop per class and a difference image.

Each feature set gets a game-grouped CV AUC with a label-permutation p-value (supervised) and a 2-cluster GMM scored with ARI/NMI/purity (unsupervised). Body features with the glove removed serve as the control.

## Run the full report

```bash
pip install -r requirements.txt
python scout.py "Zack Wheeler" --start 2024-03-28 --end 2025-09-28            # stats only, quick
python scout.py "Zack Wheeler" --start 2024-03-28 --end 2025-09-28 --clips 600  # plus vision layers
python scout.py "Mason Miller" --clips 600 --tell                                # plus the glove tell test
```

On a headless Linux box, MediaPipe pose needs the EGL/GLES libraries
(`sudo apt-get install libegl1 libgles2 libgl1`); without them every clip is skipped
with `libEGL.so.1: cannot open shared object file`.

You need network access to `baseballsavant.mlb.com` (Statcast and clips). Statcast pulls, pose tracks, features and model results are stored in the local database (`data/pitchscout.duckdb`, see below) and video clips in `cache/clips/`, so re-runs are quick. Output goes to `reports/`:

| File | What it is |
|---|---|
| `<pitcher>_report.html` | Self-contained interactive report |
| `model_comparison.csv` | Accuracy and log loss for every model, ready for a paper table |
| `unsupervised_alignment.json` | k, BIC curve, purity, ARI, NMI, silhouette |
| `cluster_composition.csv` | Pitch-type share inside each cluster |
| `vision_part_importance.csv` | Share of the vision model's attention by body part |
| `glove_tell_test.csv`, `glove_tell_summary.json` | Glove tell test results and verdict (with `--tell`) |
| `glove_mean_crops.png` | Mean glove crop per class and their difference |
| `overlay_<play_id>.mp4/.png` | Skeleton overlays for the example clips |

Options: `--window` (seconds before release the vision window starts, default 1.5) and `--cutoff` (seconds before release it stops, default 0.10). Raising `--cutoff` is a good ablation: tells that survive a bigger cutoff are ones a hitter could actually pick up.

## Local database

Everything the pipeline produces goes into one DuckDB file, `data/pitchscout.duckdb` (set `PITCHSCOUT_DB` to put it elsewhere). There's no server to run.

| Table | One row per |
|---|---|
| `pitchers` | pitcher |
| `statcast_pulls` | stored (pitcher, date range) pull, so a covered range is never re-downloaded |
| `pitches` | Statcast pitch, keyed by `(game_pk, at_bat_number, pitch_number)`, with its `play_id` once known. Every Statcast column is kept. |
| `clips` | Savant clip tried, with status (`ok`, `no_clip`, `no_pose`, `error`) |
| `pose_tracks` / `pose_frames` | clip + window: release frame and window, then every MediaPipe landmark per frame |
| `features` | clip + feature set (`pose`, `glove_pose`) + feature, long format |
| `model_runs` | report or notebook run, with the git commit, config and a JSON summary (verdict, cluster alignment) |
| `model_results` | metric per model per run (accuracy, log loss, AUC, permutation p, ARI, ...) |

```bash
python -m pitchscout.db                      # row counts and recent runs
duckdb data/pitchscout.duckdb                # plain SQL, if you have the duckdb CLI
```

```python
from pitchscout import db
with db.connect() as con:
    db.results(con)                          # latest run, one row per model
    con.sql("SELECT pitch_type, avg(release_speed) FROM pitches GROUP BY 1").df()
```

Old `cache/statcast_*.parquet` and `cache/pose/*.npz` files are imported the first time they're needed. DuckDB allows one writer at a time, so the code opens the file only for as long as it needs it.

**Tests run offline** against `tests/fixtures/sample`, a small slice of the database (pitches, pose tracks) loaded in memory. It is synthetic for now: Statcast-shaped rows for a made-up hard-throwing righty, not Mason Miller's real pitches. Once you've pulled real data, `python -m pitchscout.db make-fixture --from-db data/pitchscout.duckdb --pitcher <id>` swaps in a real sample. See `tests/fixtures/sample/README.md`.

## How it avoids fooling itself

- **No future information.** Situation features only use what is known before the pitch. Previous-pitch features stay inside the at-bat.
- **No peeking at release.** Pose frames after `release - cutoff` are dropped. A test checks that changing post-cutoff frames doesn't change the features.
- **No same-game leakage.** The stats model is scored on the latest 20% of games. The vision models use cross-validation grouped by game, and get out-of-fold stats probabilities as their prior.
- **A fair supervised vs. unsupervised comparison.** Both use the same clips and the same folds. The clustering never sees labels; labels only map clusters to pitch shares on the training folds.

## Layout

```
pitchscout/data.py          Statcast pull (pybaseball) + cleaning
pitchscout/features.py      situation features, time split
pitchscout/baseline.py      tendencies + stats model (early-stopped LightGBM)
pitchscout/video.py         play_id lookup + Savant clip download
pitchscout/pose.py          shot-cut detection, pitcher tracking, release detection, normalization
pitchscout/fusion.py        supervised stats+pose model, TreeSHAP attention grids
pitchscout/unsupervised.py  PCA + GMM clusters, alignment metrics, cluster-as-predictor
pitchscout/glove.py          glove tell features and hypothesis test
pitchscout/workspace.py      notebook helpers (Config, get_pitches, get_clips, save_run, demo mode)
pitchscout/db.py             local DuckDB store: schema, read/write helpers, test sample
pitchscout/plotting.py       matplotlib helpers for the notebooks
pitchscout/viz.py           skeleton overlays weighted by attention
pitchscout/report.py        HTML report + templates/report.html
pitchscout/pipeline.py      end-to-end run + paper tables
notebooks/                   the analysis, step by step
tests/                       runs on synthetic fixtures (no network)
```

## Known limits

- **Network.** Clips come from `sporty-clips.mlb.com`; data from `baseballsavant.mlb.com`. Both must be reachable. Pose extraction runs at about 20 s per clip per CPU core, so 600 clips take a few hours on one core.
- **Grip is out of reach.** Broadcast resolution can't show the grip. The vision layer sees body mechanics: hand set, glove position, tempo, arm path, and stride.
- **Camera angle varies.** Broadcast center-field angles differ by ballpark, and normalization only partly corrects this. Add the venue as a feature, or analyze home and away games separately, if results look park-driven.
- **Clip usage.** Clips belong to MLB. Use them for personal research only and don't redistribute them.
