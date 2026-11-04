"""Assemble the scouting report (single self-contained HTML file)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PITCH_NAMES
from .fusion import N_BINS, attention_grid, part_importance, pick_examples, pitch_explanation
from .pose import BONES, JOINTS

TEMPLATE = Path(__file__).parent / "templates" / "report.html"
FEATURE_LABELS = {
    "count": "Count", "balls": "Balls", "strikes": "Strikes", "stand": "Batter side",
    "prev_pitch": "Previous pitch", "prev2_pitch": "Two pitches ago", "prev_result": "Previous result",
    "outs_when_up": "Outs", "inning": "Inning", "on_1b": "Runner on 1st", "on_2b": "Runner on 2nd",
    "on_3b": "Runner on 3rd", "score_diff": "Score margin", "times_thru": "Times through order",
    "pitch_of_ab": "Pitch # in at-bat", "pitch_of_game": "Pitch # in game", "prev_speed": "Previous velo",
}
NOTES = (
    "Stats layer: LightGBM on situation features known before the pitch (count, batter side, base/out "
    "state, score, inning, times through the order, previous pitches in the at-bat). Scored on the most "
    "recent 20% of games, never seen in training. Supervised vision layer: MediaPipe Pose keypoints for the "
    "pitcher, cut off before release, resampled to {bins} time steps, combined with out-of-fold stats "
    "probabilities and cross-validated by game. Importance is LightGBM's per-pitch feature contribution "
    "(TreeSHAP), summed by body part and time step. Unsupervised layer: standardized pose features, PCA, "
    "then a diagonal Gaussian mixture with k chosen by BIC, fit without labels. Labels are used only "
    "afterwards, to score alignment and to map clusters to pitch shares on training folds."
)


def _df_to_nested(df: pd.DataFrame) -> dict:
    return {str(i): {str(c): float(v) for c, v in row.items()} for i, row in df.iterrows()}


def _img(path: Path | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _situation(row: pd.Series) -> str:
    side = "LHB" if row.get("stand") == "L" else "RHB"
    return f"{int(row['balls'])}-{int(row['strikes'])} count vs {side}, inning {int(row['inning'])}"


def build(out: Path, pitcher: str, throws: str, start: str, end: str, df: pd.DataFrame,
          tend: dict, base, fusion=None, unsup=None, strips: dict | None = None,
          window_s: float = 1.5, cutoff_s: float = 0.1, synthetic: bool = False,
          tell=None, tell_panel: Path | None = None) -> Path:
    classes = base.classes
    D = {
        "pitcher": pitcher, "throws": throws, "start": start, "end": end,
        "n_pitches": int(len(df)), "classes": classes,
        "pitch_names": {c: PITCH_NAMES.get(c, c) for c in classes},
        "mix": {k: float(v) for k, v in tend["mix"].items()},
        "counts": list(tend["by_count"].index),
        "by_count": _df_to_nested(tend["by_count"]),
        "n_by_count": {k: int(v) for k, v in tend["n_by_count"].items()},
        "by_hand": _df_to_nested(tend["by_hand"]),
        "after_prev": _df_to_nested(tend["after_prev"]),
        "baseline": base.metrics,
        "stats_importance": {k: float(v) for k, v in base.importance.items()},
        "feature_labels": FEATURE_LABELS,
        "fusion": fusion.metrics if fusion is not None else None,
        "window_s": window_s, "cutoff_s": cutoff_s, "synthetic": synthetic,
        "notes": NOTES.format(bins=N_BINS),
        "joints": JOINTS, "bones": [[JOINTS.index(a), JOINTS.index(b)] for a, b in BONES],
    }
    if fusion is not None:
        step = (window_s - cutoff_s) / N_BINS
        D["time_labels"] = [f"-{window_s - step * (b + 0.5):.2f}s" for b in range(N_BINS)]
        g = attention_grid(fusion)
        D["attention"] = {"rows": list(g.index), "values": g.values.round(5).tolist()}
        D["part_importance"] = {k: float(v) for k, v in part_importance(fusion).items()}
        tells = {}
        for k, c in enumerate(classes):
            mask = fusion.y == c
            if mask.sum() == 0:
                continue
            tg = attention_grid(fusion, mask=mask, cls=k, signed=True).head(10)
            tells[c] = {"rows": list(tg.index), "values": tg.values.round(5).tolist()}
        D["tells"] = tells
        # Examples: two confident hits where vision changed the call, plus one miss.
        pick = pick_examples(fusion)
        exs = []
        for i in pick:
            e = pitch_explanation(fusion, i)
            row = df.loc[fusion.index[i]]
            exs.append({
                "pred": e["pred"], "true": e["true"], "ok": e["pred"] == e["true"],
                "situation": _situation(row),
                "prior": dict(zip(classes, map(float, e["prior"]))),
                "proba": dict(zip(classes, map(float, e["proba"]))),
                "parts": [[k, float(v)] for k, v in e["per_part"].head(6).items()],
                "strip": _img((strips or {}).get(fusion.index[i])),
            })
        D["examples"] = exs
    if unsup is not None:
        D["unsup"] = {
            "k": unsup.k, "bic": unsup.bic, "metrics": unsup.metrics, "alignment": unsup.alignment,
            "composition": _df_to_nested(unsup.composition),
            "sizes": {str(k): int(v) for k, v in unsup.cluster_sizes.items()},
            "points": [[round(float(a), 3), round(float(b), 3), int(c), str(t)]
                       for (a, b), c, t in zip(unsup.embedding, unsup.clusters, fusion.y if fusion is not None else [])],
            "mean_pose": {str(c): np.nan_to_num(v).round(3).tolist() for c, v in unsup.mean_pose.items()},
        }
    if tell is not None:
        D["tell"] = {"pos": tell.positive_label, "neg": tell.negative_label, "n": tell.n,
                     "n_pos": tell.n_pos, "sets": tell.sets, "phase_auc": tell.phase_auc,
                     "top": tell.top_features, "verdict": tell.verdict, "panel": _img(tell_panel),
                     "n_perm": tell.n_perm}
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(D).replace("</", "<\\/"))
    out.write_text(html)
    return out
