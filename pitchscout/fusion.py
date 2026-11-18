"""Fuse the stats prior with pose features and explain what the vision layer uses."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from .pose import JOINTS, PoseTrack, normalize, resample

N_BINS = 8
# Lead-side (glove-side) after mirroring is "l_", throwing side is "r_".
PART_LABEL = {
    "nose": "Head", "l_shoulder": "Glove shoulder", "r_shoulder": "Throwing shoulder",
    "l_elbow": "Glove elbow", "r_elbow": "Throwing elbow", "l_wrist": "Glove hand",
    "r_wrist": "Throwing hand", "l_hip": "Glove hip", "r_hip": "Throwing hip",
    "l_knee": "Front knee", "r_knee": "Back knee", "l_ankle": "Front foot", "r_ankle": "Back foot",
    "hands_sep": "Hand separation", "throw_elbow_angle": "Throwing elbow angle",
    "trunk_lean": "Trunk lean", "shoulder_tilt": "Shoulder tilt", "stride": "Stride length",
    "arm_height": "Throwing arm height",
}


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def pose_features(track: PoseTrack, throws: str, n_bins: int = N_BINS) -> dict[str, float]:
    xy = normalize(track.frames, throws)[track.start:track.end]
    if len(xy) < 2:
        return {}
    r = resample(xy, n_bins)  # (n_bins, joints, 2)
    ix = {j: i for i, j in enumerate(JOINTS)}
    feats = {}
    for b in range(n_bins):
        for j in JOINTS:
            feats[f"{j}|x|{b}"] = r[b, ix[j], 0]
            feats[f"{j}|y|{b}"] = r[b, ix[j], 1]
        P = lambda j: r[b, ix[j]]
        feats[f"hands_sep||{b}"] = np.linalg.norm(P("l_wrist") - P("r_wrist"))
        feats[f"throw_elbow_angle||{b}"] = _angle(P("r_shoulder"), P("r_elbow"), P("r_wrist"))
        feats[f"trunk_lean||{b}"] = (P("l_shoulder") + P("r_shoulder"))[0] / 2
        feats[f"shoulder_tilt||{b}"] = P("l_shoulder")[1] - P("r_shoulder")[1]
        feats[f"stride||{b}"] = np.linalg.norm(P("l_ankle") - P("r_ankle"))
        feats[f"arm_height||{b}"] = P("r_shoulder")[1] - P("r_wrist")[1]
    # Tempo: time from peak leg lift (front knee highest) to release.
    full = normalize(track.frames, throws)
    knee_y = full[: track.release + 1, ix["l_knee"], 1]
    lift = int(np.argmin(knee_y)) if len(knee_y) else track.release
    feats["tempo||"] = (track.release - lift) / track.fps
    return feats


def parse_feature(name: str) -> tuple[str, int | None]:
    """'r_wrist|y|3' -> ('Throwing hand', 3); 'tempo||' -> ('Tempo', None)."""
    part, _, b = name.split("|")
    if part.startswith("prior_"):
        return "Stats prior", None
    if part == "tempo":
        return "Tempo (leg lift to release)", None
    return PART_LABEL.get(part, part), (int(b) if b != "" else None)


@dataclass
class FusionResult:
    classes: list[str]
    metrics: dict
    contrib: np.ndarray          # (n, F, K) per-pitch, per-feature, per-class contribution (log-odds)
    feature_names: list[str]
    proba: np.ndarray            # out-of-fold fused probabilities
    prior: np.ndarray            # stats-only probabilities for the same pitches
    y: np.ndarray
    index: np.ndarray            # row index into the pitch dataframe
    per_pitch: dict = field(default_factory=dict)  # model -> {"log_loss": (n,), "correct": (n,)}


def paired_game_bootstrap(a: np.ndarray, b: np.ndarray, games: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float, float]:
    """mean(a - b) for per-pitch scores of two models on the same pitches, with a 2.5-97.5%
    interval from resampling whole games (errors are correlated within a game)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    uniq, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    boot = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def _model(seed=0):
    return lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3.0,
        random_state=seed, verbose=-1,
    )


def _offset_model(seed=0):
    # Fewer trees than _model: the trees only fit a residual on top of the stats prior.
    return lgb.LGBMClassifier(
        n_estimators=60, learning_rate=0.03, num_leaves=7, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3.0,
        random_state=seed, verbose=-1,
    )


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _inner_oof_vision(Xp: pd.DataFrame, y: np.ndarray, games: np.ndarray, labels: np.ndarray,
                      n_inner: int = 3, seed: int = 0) -> np.ndarray:
    """Out-of-fold vision-only probabilities inside a training fold (game-grouped), columns in
    the order of `labels`. Used to train the stacker without leaking the vision fit."""
    uniq = np.unique(games)
    rng = np.random.default_rng(seed + 100)
    inner = pd.Series(rng.integers(0, n_inner, len(uniq)), index=uniq).loc[games].to_numpy()
    out = np.full((len(y), len(labels)), 1.0 / len(labels))
    for j in range(n_inner):
        tr, te = inner != j, inner == j
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = _model(seed).fit(Xp[tr], y[tr])
        idx = [int(np.flatnonzero(labels == c)[0]) for c in m.classes_]
        out[np.ix_(te, idx)] = m.predict_proba(Xp[te])
    out = np.clip(out, 1e-6, 1)
    return out / out.sum(1, keepdims=True)


def fit_fusion(pose_df: pd.DataFrame, prior: np.ndarray, y: np.ndarray, games: np.ndarray,
               classes: list[str], n_folds: int = 5) -> FusionResult:
    """pose_df: one row per clip (pose features). prior: stats-model probs for the same
    pitches (out-of-fold). Cross-validated by game so no game is in train and test."""
    P = pd.DataFrame(np.log(np.clip(prior, 1e-4, 1)), index=pose_df.index,
                     columns=[f"prior_{c}||" for c in classes])
    X = pd.concat([P, pose_df], axis=1)
    pose_only = pose_df
    K, n = len(classes), len(X)
    uniq = np.unique(games)
    rng = np.random.default_rng(1)
    fold = pd.Series(rng.integers(0, n_folds, len(uniq)), index=uniq).loc[games].to_numpy()

    logp = np.log(np.clip(prior, 1e-6, 1))
    fused = np.zeros((n, K))
    vis = np.zeros((n, K))
    offset = np.zeros((n, K))
    stacked = np.zeros((n, K))
    contrib = np.zeros((n, X.shape[1], K))
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        cols = [classes.index(c) for c in np.unique(y[tr])]
        m = _model(k).fit(X[tr], y[tr])
        fused[np.ix_(te, cols)] = m.predict_proba(X[te])
        c = m.predict_proba(X[te], pred_contrib=True).reshape(te.sum(), len(cols), X.shape[1] + 1)
        contrib[np.ix_(te, np.arange(X.shape[1]), cols)] = c[:, :, :-1].transpose(0, 2, 1)
        mv = _model(k).fit(pose_only[tr], y[tr])
        vis[np.ix_(te, cols)] = mv.predict_proba(pose_only[te])
        # Offset: trees fit a residual on top of log(prior); predict() does not add it back.
        lp_tr, lp_te = logp[np.ix_(tr, cols)], logp[np.ix_(te, cols)]
        if len(cols) == 2:  # binary LightGBM uses one raw score: log-odds of the second class
            mo = _offset_model(k).fit(pose_only[tr], y[tr], init_score=lp_tr[:, 1] - lp_tr[:, 0])
            raw = mo.predict(pose_only[te], raw_score=True) + lp_te[:, 1] - lp_te[:, 0]
            offset[np.ix_(te, cols)] = _softmax(np.c_[np.zeros_like(raw), raw])
        else:
            mo = _offset_model(k).fit(pose_only[tr], y[tr], init_score=lp_tr)
            raw = mo.predict(pose_only[te], raw_score=True).reshape(te.sum(), len(cols))
            offset[np.ix_(te, cols)] = _softmax(raw + lp_te)
        # Stacked: logistic regression on [log prior, log vision], with the vision column
        # from nested out-of-fold predictions on the training games.
        labels = np.asarray(classes)[cols]
        v_tr = _inner_oof_vision(pose_only[tr], y[tr], games[tr], labels, seed=k)
        v_te = np.clip(mv.predict_proba(pose_only[te]), 1e-6, 1)
        Z_tr = np.hstack([logp[np.ix_(tr, cols)], np.log(v_tr)])
        Z_te = np.hstack([logp[np.ix_(te, cols)], np.log(v_te)])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z_tr, y[tr])
        idx = [cols[list(labels).index(c)] for c in lr.classes_]
        stacked[np.ix_(te, idx)] = lr.predict_proba(Z_te)
    fused = np.clip(fused, 1e-6, 1); fused /= fused.sum(1, keepdims=True)
    vis = np.clip(vis, 1e-6, 1); vis /= vis.sum(1, keepdims=True)
    for a in (offset, stacked):
        a[a.sum(1) == 0] = prior[a.sum(1) == 0]  # folds skipped above fall back to the prior
        np.clip(a, 1e-6, 1, out=a); a /= a.sum(1, keepdims=True)

    def met(p):
        return {"log_loss": float(log_loss(y, p, labels=classes)),
                "accuracy": float(accuracy_score(y, np.asarray(classes)[p.argmax(1)]))}

    yi = np.array([classes.index(c) for c in y])
    per_pitch = {name: {"log_loss": -np.log(np.clip(p[np.arange(n), yi], 1e-15, 1)),
                        "correct": (p.argmax(1) == yi).astype(float)}
                 for name, p in (("stats_only", prior), ("vision_only", vis), ("stats_plus_vision", fused),
                                 ("stats_offset_vision", offset), ("stats_stacked_vision", stacked))}

    def gain(name):
        # Paired, game-resampled test of the vision gain: positive delta_logloss means this model
        # has lower log loss than stats alone; positive delta_accuracy means more correct.
        s_, f_ = per_pitch["stats_only"], per_pitch[name]
        dl, dl_lo, dl_hi = paired_game_bootstrap(s_["log_loss"], f_["log_loss"], games)
        da, da_lo, da_hi = paired_game_bootstrap(f_["correct"], s_["correct"], games)
        return {"delta_logloss": dl, "delta_logloss_lo": dl_lo, "delta_logloss_hi": dl_hi,
                "delta_accuracy": da, "delta_accuracy_lo": da_lo, "delta_accuracy_hi": da_hi}

    mix = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(1e-6).to_numpy()
    metrics = {
        "naive_mix": met(np.tile(mix, (n, 1))),
        "stats_only": met(prior),
        "vision_only": met(vis),
        "stats_plus_vision": {**met(fused), **gain("stats_plus_vision")},
        "stats_offset_vision": {**met(offset), **gain("stats_offset_vision")},
        "stats_stacked_vision": {**met(stacked), **gain("stats_stacked_vision")},
        "n_clips": int(n), "n_games": int(len(uniq)),
    }
    return FusionResult(classes, metrics, contrib, list(X.columns), fused, prior, y,
                        pose_df.index.to_numpy(), per_pitch)


def attention_grid(res: FusionResult, mask: np.ndarray | None = None, cls: int | None = None,
                   signed: bool = False) -> pd.DataFrame:
    """Body part x time-bin importance. Default: mean |contribution| across classes.
    With cls and signed=True: mean signed push toward that pitch type."""
    c = res.contrib if mask is None else res.contrib[mask]
    if cls is None:
        v = np.abs(c).sum(2).mean(0)
    else:
        v = c[:, :, cls].mean(0) if signed else np.abs(c[:, :, cls]).mean(0)
    rows = {}
    for name, val in zip(res.feature_names, v):
        part, b = parse_feature(name)
        if b is None:
            continue
        rows.setdefault(part, np.zeros(N_BINS))[b] += val
    grid = pd.DataFrame(rows).T
    grid.columns = [f"t{b}" for b in range(N_BINS)]
    return grid.loc[grid.abs().sum(axis=1).sort_values(ascending=False).index]


def part_importance(res: FusionResult) -> pd.Series:
    """Share of total |contribution| by body part, plus the stats prior and tempo."""
    v = np.abs(res.contrib).sum(2).mean(0)
    s = pd.Series(v, index=[parse_feature(f)[0] for f in res.feature_names]).groupby(level=0).sum()
    return (s / s.sum()).sort_values(ascending=False)


def pitch_explanation(res: FusionResult, i: int) -> dict:
    """For clip i: what pushed the model toward its predicted pitch, per body part & bin."""
    k = int(res.proba[i].argmax())
    c = res.contrib[i, :, k]
    per_part, per_bin = {}, np.zeros(N_BINS)
    for name, val in zip(res.feature_names, c):
        part, b = parse_feature(name)
        per_part[part] = per_part.get(part, 0.0) + val
        if b is not None:
            per_bin[b] += abs(val)
    return {"pred": res.classes[k], "true": str(res.y[i]), "proba": res.proba[i],
            "prior": res.prior[i], "per_part": pd.Series(per_part).sort_values(key=abs, ascending=False),
            "per_bin": per_bin}


def pick_examples(res: FusionResult) -> list[int]:
    """Clips to showcase: up to two confident hits (preferring ones where vision fixed
    a wrong stats guess) and the most confident miss."""
    pred, prior_pred = res.proba.argmax(1), res.prior.argmax(1)
    truth = np.array([res.classes.index(v) for v in res.y])
    conf = res.proba.max(1)
    fixed = np.where((pred == truth) & (prior_pred != truth))[0]
    hits = np.where(pred == truth)[0]
    miss = np.where(pred != truth)[0]
    pick = [int(i) for i in fixed[np.argsort(-conf[fixed])][:2]]
    pick += [int(i) for i in hits[np.argsort(-conf[hits])] if i not in pick][: 2 - len(pick)]
    pick += [int(i) for i in miss[np.argsort(-conf[miss])][:1]]
    return pick
