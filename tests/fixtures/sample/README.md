# Test sample

A small, committed slice of the database so tests and the notebooks' demo mode run
with no network. `db.fixture_db()` loads it into an in-memory DuckDB.

| File | Rows | What |
|---|---|---|
| `pitches.parquet` | 968 | Statcast-shaped pitches, 60 one-inning games |
| `pitchers.parquet` | 1 | the pitcher row (`synthetic = true`) |
| `clips.parquet` | 150 | clip status for the pitches that have pose |
| `pose_tracks.parquet` | 150 | fps, release frame and model window per clip |
| `pose_frames.parquet` | 297,000 | 33 landmarks x 60 frames x 150 clips |

**These rows are synthetic.** Baseball Savant wasn't reachable when this was built, so
`demo.miller_shaped_statcast()` generates rows shaped like a hard-throwing right-handed
closer (mostly 4-seamers around 100 mph, a slider, a few changeups) under a made-up
pitcher id (900001). They are not Mason Miller's pitches and no result from them should
be reported. Pose tracks come from `demo.fake_track`, which plants a glove-height tell on
changeups so the tests can check the models find it.

Rebuild it (deterministic):

```bash
python -m pitchscout.db make-fixture
```

Swap in real rows once you've pulled them locally (copies the latest 60 games and any
stored pose tracks; video files and paths are never copied):

```bash
python scout.py "Mason Miller" --clips 150
python -m pitchscout.db make-fixture --from-db data/pitchscout.duckdb --pitcher <MLBAM id>
```

If you do that, the planted-tell tests in `tests/test_pipeline.py` still use the
synthetic generators, but `tests/test_db.py` will run on the real sample; adjust the
pitch-type assertion in `test_fixture_loads` to his actual mix.
