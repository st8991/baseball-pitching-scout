"""End-to-end: pitcher name -> data -> models -> report + paper-ready tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline, data, fusion, glove, pose, unsupervised, video, viz


def process_clips(df: pd.DataFrame, throws: str, n_clips: int, window_s: float, cutoff_s: float,
                  log=print, glove_extra: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Download + pose-extract up to n_clips pitches. Returns pose feature rows (indexed
    like df) and the PoseTrack per row."""
    sample = video.sample_for_video(df, n_clips)
    sample = video.attach_play_ids(sample)
    tracks, rows = {}, {}
    pose_dir = Path("cache/pose"); pose_dir.mkdir(parents=True, exist_ok=True)
    for n, (i, r) in enumerate(sample.iterrows(), 1):
        pid = r.get("play_id")
        if not isinstance(pid, str):
            continue
        cache = pose_dir / f"{pid}_{window_s}_{cutoff_s}.npz"
        try:
            if cache.exists():
                tr = pose.load(cache)
            else:
                clip = video.download_clip(pid)
                tr = pose.extract(clip, throws, window_s, cutoff_s) if clip else None
                if tr is None:
                    continue
                pose.save(tr, cache)
        except Exception as e:  # one bad clip shouldn't stop the run
            log(f"  skip {pid}: {e}")
            continue
        f = fusion.pose_features(tr, throws)
        if f:
            tracks[i], rows[i] = (tr, pid), f
            if glove_extra is not None:
                g = glove_extra
                g.setdefault("pose", {})[i] = glove.glove_pose_features(tr, throws)
                clip = video.CLIPS / f"{pid}.mp4"
                if clip.exists():
                    c = glove.glove_crops(clip, tr, throws)
                    if c is not None:
                        g.setdefault("crops", {})[i] = c
                        g.setdefault("pix", {})[i] = glove.crop_features(c)
        if n % 25 == 0:
            log(f"  {n}/{len(sample)} clips, {len(rows)} usable")
    return pd.DataFrame.from_dict(rows, orient="index"), tracks


def run(pitcher: str, start: str, end: str, n_clips: int = 0, out_dir: Path = Path("reports"),
        window_s: float = 1.5, cutoff_s: float = 0.10, log=print, pitcher_id: int | None = None,
        tell: bool = False, positive: set[str] | None = None, negative: set[str] | None = None,
        n_perm: int = 200) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = pitcher_id or data.lookup_pitcher(pitcher)
    log(f"{pitcher}: MLBAM id {pid}")
    df = data.clean(data.load_pitches(pid, start, end))
    throws = df["p_throws"].mode().iloc[0]
    log(f"{len(df)} pitches, types: {df['pitch_type'].value_counts().to_dict()}")

    tend = baseline.tendencies(df)
    base = baseline.fit_baseline(df)
    log(f"stats model: {base.metrics}")

    fus = uns = tell_res = panel = None
    strips = {}
    if n_clips:
        gx = {} if tell else None
        pose_df, tracks = process_clips(df, throws, n_clips, window_s, cutoff_s, log, gx)
        log(f"{len(pose_df)} usable clips")
        if len(pose_df) >= 30:
            idx = pose_df.index.to_numpy()
            prior = base.oof_proba[idx]
            y = df.loc[idx, "pitch_type"].to_numpy()
            games = df.loc[idx, "game_pk"].to_numpy()
            fus = fusion.fit_fusion(pose_df, prior, y, games, base.classes)
            uns = unsupervised.fit_unsupervised(pose_df, y, games, base.classes, prior=prior)
            log(f"supervised: {fus.metrics}")
            log(f"unsupervised: {uns.metrics} {uns.alignment}")
            strips = render_examples(fus, tracks, throws, out_dir)
            export_tables(out_dir, pitcher, base, fus, uns)
            if tell:
                pos = positive or (set(df["pitch_type"]) & glove.FASTBALLS)
                gp = pd.DataFrame.from_dict(gx.get("pose", {}), orient="index").reindex(idx)
                gpx = pd.DataFrame.from_dict(gx.get("pix", {}), orient="index") if gx.get("pix") else None
                tell_res = glove.run_tell_test(gp, gpx, pose_df, y, games, pos, negative,
                                               gx.get("crops"), n_perm, data.PITCH_NAMES)
                panel = glove.crop_panel(tell_res, out_dir / "glove_mean_crops.png")
                export_tell(out_dir, pitcher, tell_res)
                log(tell_res.verdict)

    slug = pitcher.lower().replace(" ", "_")
    out = out_dir / f"{slug}_report.html"
    from .report import build
    build(out, pitcher, throws, start, end, df, tend, base, fus, uns, strips, window_s, cutoff_s,
          tell=tell_res, tell_panel=panel)
    log(f"report: {out}")
    return out


def render_examples(fus, tracks, throws, out_dir: Path) -> dict:
    """Skeleton overlays (PNG strip + mp4) for the clips the report showcases."""
    strips = {}
    for i in fusion.pick_examples(fus):
        row = fus.index[i]
        k = fus.classes.index(fusion.pitch_explanation(fus, i)["pred"])
        grid = {}
        for name, val in zip(fus.feature_names, fus.contrib[i, :, k]):
            part, b = fusion.parse_feature(name)
            if b is not None:
                grid.setdefault(part, np.zeros(fusion.N_BINS))[b] += val
        tr, pid = tracks[row]
        png = out_dir / f"overlay_{pid}.png"
        viz.render_overlay(video.CLIPS / f"{pid}.mp4", tr, viz.joint_weights(grid, throws), throws,
                           out_dir / f"overlay_{pid}.mp4", png)
        strips[row] = png
    return strips


def export_tables(out_dir: Path, pitcher: str, base, fus, uns) -> None:
    """CSV/JSON a paper can cite directly."""
    rows = [
        ("Stats", "Overall mix", "baseline", base.metrics["naive_mix"], "held-out games"),
        ("Stats", "Mix by count", "baseline", base.metrics["count_only"], "held-out games"),
        ("Stats", "LightGBM situation model", "supervised", base.metrics["stats_model"], "held-out games"),
        ("Clips", "Overall mix", "baseline", fus.metrics["naive_mix"], "game-grouped CV"),
        ("Clips", "Stats only (out-of-fold)", "supervised", fus.metrics["stats_only"], "game-grouped CV"),
        ("Clips", "Vision only (LightGBM on pose)", "supervised", fus.metrics["vision_only"], "game-grouped CV"),
        ("Clips", "Stats + vision (LightGBM)", "supervised", fus.metrics["stats_plus_vision"], "game-grouped CV"),
        ("Clips", "Clusters only (PCA + GMM)", "unsupervised", uns.metrics["clusters_only"], "game-grouped CV"),
        ("Clips", "Stats + clusters", "unsupervised", uns.metrics["stats_plus_clusters"], "game-grouped CV"),
    ]
    pd.DataFrame([{"pitcher": pitcher, "dataset": d, "model": m, "family": f,
                   "accuracy": r["accuracy"], "log_loss": r["log_loss"], "evaluation": e}
                  for d, m, f, r, e in rows]).to_csv(out_dir / "model_comparison.csv", index=False)
    json.dump({"pitcher": pitcher, "k": uns.k, "bic": uns.bic, **uns.alignment},
              open(out_dir / "unsupervised_alignment.json", "w"), indent=2)
    uns.composition.to_csv(out_dir / "cluster_composition.csv")
    fusion.part_importance(fus).rename("share").to_csv(out_dir / "vision_part_importance.csv")


def export_tell(out_dir: Path, pitcher: str, res) -> None:
    rows = [{"pitcher": pitcher, "positive": res.positive_label, "negative": res.negative_label,
             "feature_set": k, **v} for k, v in res.sets.items()]
    pd.DataFrame(rows).to_csv(out_dir / "glove_tell_test.csv", index=False)
    json.dump({"verdict": res.verdict, "n": res.n, "n_positive": res.n_pos,
               "phase_auc": res.phase_auc, "top_glove_differences": res.top_features},
              open(out_dir / "glove_tell_summary.json", "w"), indent=2)
