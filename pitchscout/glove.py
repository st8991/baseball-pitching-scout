"""Glove tell test: can the glove alone separate fastballs from offspeed?

Two kinds of glove evidence, both taken only from frames before release:
  * pose: glove-hand wrist plus the MediaPipe hand points (index, pinky, thumb)
    on the glove side, giving glove position, angle and spread, in two phases:
    SET (clip start to peak leg lift) and WINDUP (leg lift to the cutoff before release).
  * pixels: a crop centered on the glove hand, summarized with HOG and color
    histograms, plus mean crops per class and a difference image.

The test reports, for glove features vs a body-without-glove control:
  * supervised: game-grouped CV AUC/accuracy with a game-level bootstrap CI, and two
    label-permutation p-values: within-game (primary; pitches are only exchangeable inside
    a game) and a free shuffle across all pitches (kept for comparison)
  * unsupervised: k=2 Gaussian mixture on the same features, ARI / NMI / purity, each with
    a within-game permutation p-value (clusters fixed, labels shuffled; no refit)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .pose import PoseTrack, normalize_all, read_frames, resample
from .unsupervised import alignment_perm_p, permute_within as _permute_within

FASTBALLS = {"FF", "SI", "FC"}
PHASE_BINS = 4
CROP = 64

# Glove side is the left side after mirroring (normalize_all mirrors lefties).
G = {"wrist": 15, "pinky": 17, "index": 19, "thumb": 21, "elbow": 13, "shoulder": 11,
     "t_shoulder": 12, "t_wrist": 16}


def phases(track: PoseTrack, throws: str) -> dict[str, tuple[int, int]]:
    """SET = clip start .. peak leg lift; WINDUP = leg lift .. model cutoff."""
    xy = normalize_all(track.frames, throws)
    knee = xy[: track.release + 1, 25, 1]  # front knee (mirrored) y; smaller = higher
    lift = int(np.argmin(knee)) if len(knee) else track.start
    lift = int(np.clip(lift, 1, max(1, track.end - 2)))
    return {"set": (0, lift), "windup": (lift, track.end)}


def glove_pose_features(track: PoseTrack, throws: str) -> dict[str, float]:
    xy = normalize_all(track.frames, throws)
    out = {}
    for ph, (a, b) in phases(track, throws).items():
        seg = xy[a:b]
        if len(seg) < 2:
            continue
        r = resample(seg, PHASE_BINS)
        for k in range(PHASE_BINS):
            P = lambda n: r[k, G[n]]
            w = P("wrist")
            v_idx, v_pky, v_thb = P("index") - w, P("pinky") - w, P("thumb") - w
            f = {
                "wrist_x": w[0], "wrist_y": w[1],
                "angle": np.degrees(np.arctan2(v_idx[1], v_idx[0])),
                "spread": np.linalg.norm(P("index") - P("pinky")),
                "thumb_open": np.linalg.norm(v_thb),
                "tilt": np.degrees(np.arctan2(v_pky[1], v_pky[0])),
                "to_chest": np.linalg.norm(w - (P("shoulder") + P("t_shoulder")) / 2),
                "to_throw_hand": np.linalg.norm(w - P("t_wrist")),
                "elbow_flare": P("elbow")[0] - P("shoulder")[0],
            }
            for n, v in f.items():
                out[f"glove_{n}|{ph}|{k}"] = float(v)
        # Static summary of the set: how still and where the glove rests.
        out[f"glove_rest_y|{ph}|"] = float(np.median(seg[:, G["wrist"], 1]))
        out[f"glove_motion|{ph}|"] = float(np.abs(np.diff(seg[:, G["wrist"]], axis=0)).sum(1).mean())
    return out


def glove_crops(clip: Path, track: PoseTrack, throws: str, n: int = 6) -> np.ndarray | None:
    """(n, CROP, CROP, 3) BGR crops around the glove hand, spread over the SET and WINDUP
    phases (3 each). Lefties are flipped so the glove is on the same side."""
    frames, _ = read_frames(clip)
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    ph = phases(track, throws)
    idx = np.r_[np.linspace(*ph["set"], n // 2, endpoint=False),
                np.linspace(*ph["windup"], n - n // 2, endpoint=False)].astype(int)
    gw = 16 if throws == "L" else 15  # real glove wrist index
    out = []
    for t in idx:
        if t >= len(frames) or t >= len(track.frames):
            return None
        p = track.frames[t]
        torso = np.linalg.norm((p[11, :2] + p[12, :2]) / 2 - (p[23, :2] + p[24, :2]) / 2) * h
        half = int(max(12, 0.45 * torso))
        if not np.isfinite(p[gw, :2]).all() or not np.isfinite(half):
            return None
        # landmarks can sit outside the frame; clamp so the padded crop is never empty
        cx = int(np.clip(p[gw, 0] * w, 0, w - 1))
        cy = int(np.clip(p[gw, 1] * h, 0, h - 1))
        pad = cv2.copyMakeBorder(frames[t], half, half, half, half, cv2.BORDER_REPLICATE)
        crop = pad[cy:cy + 2 * half, cx:cx + 2 * half]
        crop = cv2.resize(crop, (CROP, CROP))
        if throws == "L":
            crop = crop[:, ::-1]
        out.append(crop)
    return np.stack(out)


_HOG = cv2.HOGDescriptor((CROP, CROP), (32, 32), (16, 16), (16, 16), 9)


def crop_features(crops: np.ndarray) -> dict[str, float]:
    out = {}
    for i, c in enumerate(crops):
        ph = "set" if i < len(crops) // 2 else "windup"
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        for j, v in enumerate(_HOG.compute(g).ravel()):
            out[f"glovepix_hog{j}|{ph}|{i}"] = float(v)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [6, 4], [0, 180, 0, 256]).ravel()
        hist /= hist.sum() + 1e-9
        for j, v in enumerate(hist):
            out[f"glovepix_hsv{j}|{ph}|{i}"] = float(v)
    return out


def binary_target(y: np.ndarray, positive: set[str] = FASTBALLS) -> np.ndarray:
    return np.isin(y, list(positive)).astype(int)


def _clf(seed=0):
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3.0,
                              random_state=seed, verbose=-1, n_jobs=1)


def _cv_proba(X: np.ndarray, yb: np.ndarray, fold: np.ndarray, seed=0) -> np.ndarray:
    p = np.full(len(yb), yb.mean())
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if len(np.unique(yb[tr])) < 2:
            continue
        p[te] = _clf(seed).fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
    return p


def _folds(games: np.ndarray, n_folds=5, seed=1) -> np.ndarray:
    uniq = np.unique(games)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, n_folds, len(uniq)), index=uniq).loc[games].to_numpy()


def _purity(yb, c):
    ct = pd.crosstab(c, yb)
    return float(ct.max(axis=1).sum() / ct.values.sum())


@dataclass
class TellResult:
    positive_label: str
    negative_label: str
    n: int
    n_pos: int
    n_perm: int = 0
    sets: dict = field(default_factory=dict)      # feature set -> metrics
    phase_auc: dict = field(default_factory=dict)  # glove pose by phase
    top_features: list = field(default_factory=list)
    verdict: str = ""
    mean_crops: dict = field(default_factory=dict)  # label -> (n, CROP, CROP, 3) uint8


def _bootstrap_auc_ci(yb: np.ndarray, p: np.ndarray, games: np.ndarray, n_boot: int = 1000,
                      seed: int = 0) -> tuple[float, float]:
    """2.5-97.5% interval for the out-of-fold AUC, resampling whole games with replacement."""
    uniq = np.unique(games)
    rows = [np.flatnonzero(games == g) for g in uniq]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = np.concatenate([rows[j] for j in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(yb[i])) == 2:
            aucs.append(roc_auc_score(yb[i], p[i]))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return float(lo), float(hi)


def _paired_auc_diff(yb: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, games: np.ndarray,
                     n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUC(p_a) - AUC(p_b) on the same pitches, with a 2.5-97.5% interval from resampling
    whole games and recomputing both AUCs on each resample (a paired comparison)."""
    diff = float(roc_auc_score(yb, p_a) - roc_auc_score(yb, p_b))
    uniq = np.unique(games)
    rows = [np.flatnonzero(games == g) for g in uniq]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = np.concatenate([rows[j] for j in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(yb[i])) == 2:
            d.append(roc_auc_score(yb[i], p_a[i]) - roc_auc_score(yb[i], p_b[i]))
    if not d:
        return diff, float("nan"), float("nan")
    lo, hi = np.quantile(d, [0.025, 0.975])
    return diff, float(lo), float(hi)


def _perm_null(Xv, yb, fold, perms, seed) -> np.ndarray:
    from joblib import Parallel, delayed
    return np.array(Parallel(n_jobs=-1)(
        delayed(lambda yp: roc_auc_score(yp, _cv_proba(Xv, yp, fold, seed)))(yp) for yp in perms))


def test_feature_set(X: pd.DataFrame, yb: np.ndarray, games: np.ndarray, n_perm: int = 200,
                     seed: int = 0, free_shuffle: bool = True, n_boot: int = 1000,
                     return_proba: bool = False):
    """perm_p is the within-game permutation p-value (shuffle labels inside each game_pk).
    perm_p_free is the old free shuffle across all pitches; set free_shuffle=False to skip it.
    With return_proba=True, returns (metrics, out-of-fold probabilities)."""
    Xv = X.fillna(X.median()).to_numpy()
    fold = _folds(games)
    p = _cv_proba(Xv, yb, fold, seed)
    auc = float(roc_auc_score(yb, p))
    acc = float(((p > 0.5) == yb).mean())
    auc_lo, auc_hi = _bootstrap_auc_ci(yb, p, games, n_boot, seed) if n_boot else (float("nan"),) * 2
    # Permutation tests: shuffle labels, keep folds, recompute AUC.
    rng = np.random.default_rng(seed)
    pv = lambda null: float((1 + np.sum(null >= auc)) / (1 + n_perm)) if n_perm else float("nan")
    null = _perm_null(Xv, yb, fold, [_permute_within(yb, games, rng) for _ in range(n_perm)], seed)
    null_free = (_perm_null(Xv, yb, fold, [rng.permutation(yb) for _ in range(n_perm)], seed)
                 if free_shuffle else np.array([]))
    # Unsupervised: 2-cluster GMM on standardized, PCA-reduced features, no labels.
    Z = StandardScaler().fit_transform(Xv)
    Z = PCA(min(8, Z.shape[1], len(Z) - 1), random_state=0).fit_transform(Z)
    c = GaussianMixture(2, covariance_type="full", random_state=0, n_init=5).fit_predict(Z)
    # Same within-game null for the label-free clusters, so both halves carry a p-value.
    cl_p = alignment_perm_p(yb, c, games, n_perm, seed)
    out = {
        "auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "accuracy": acc, "majority": float(max(yb.mean(), 1 - yb.mean())),
        "perm_p": pv(null), "perm_p_free": pv(null_free) if len(null_free) else float("nan"),
        "null_auc_95": float(np.quantile(null, 0.95)) if len(null) else float("nan"),
        "ari": float(adjusted_rand_score(yb, c)), "nmi": float(normalized_mutual_info_score(yb, c)),
        "purity": _purity(yb, c), **cl_p, "n_features": int(X.shape[1]),
    }
    return (out, p) if return_proba else out


def run_tell_test(glove_pose: pd.DataFrame, glove_pix: pd.DataFrame | None, body: pd.DataFrame,
                  y: np.ndarray, games: np.ndarray, positive: set[str] = FASTBALLS,
                  negative: set[str] | None = None, crops: dict | None = None,
                  n_perm: int = 200, names: dict | None = None) -> TellResult:
    """body: the regular pose features; glove-related columns are dropped to form the control."""
    keep = np.isin(y, list(positive | negative)) if negative else np.ones(len(y), bool)
    y, games = y[keep], games[keep]
    yb = binary_target(y, positive)
    gp = glove_pose[keep]
    nm = names or {}
    lab = lambda s: "/".join(nm.get(x, x) for x in sorted(s))
    res = TellResult(lab(positive), lab(negative) if negative else "everything else",
                     int(len(yb)), int(yb.sum()), n_perm)
    glove_cols = [c for c in body.columns if c.split("|")[0] in ("l_wrist", "l_elbow", "hands_sep")]
    control = body[keep].drop(columns=glove_cols)
    sets = {"Glove (pose)": gp, "Body without glove (control)": control}
    if glove_pix is not None and len(glove_pix):
        gx = glove_pix.reindex(gp.index)
        ok = gx.notna().all(axis=1).to_numpy()
        if ok.sum() >= 30:
            # HOG gives hundreds of columns; compress label-free with PCA so the
            # permutation test stays fast and the classifier doesn't chase noise.
            gx = gx[ok]
            Zp = PCA(min(20, gx.shape[1], len(gx) - 1), random_state=0).fit_transform(
                StandardScaler().fit_transform(gx.to_numpy()))
            sets["Glove (pixels)"] = pd.DataFrame(Zp, index=gx.index,
                                                  columns=[f"glovepix_pc{j}||" for j in range(Zp.shape[1])])
    proba = {}
    for name, X in sets.items():
        m = np.isin(gp.index, X.index)
        res.sets[name], proba[name] = test_feature_set(X, yb[m], games[m], n_perm, return_proba=True)
    # Paired glove - body AUC difference on the same pitches. Glove pose and the control share
    # rows and folds; the pixel set uses the control's out-of-fold predictions on its own rows.
    ctrl = "Body without glove (control)"
    for name, X in sets.items():
        if name == ctrl or not name.startswith("Glove"):
            continue
        m = np.isin(gp.index, X.index)
        d, lo, hi = _paired_auc_diff(yb[m], proba[name], proba[ctrl][m], games[m])
        res.sets[name].update(auc_diff=d, auc_diff_lo=lo, auc_diff_hi=hi)
    for ph in ("set", "windup"):
        cols = [c for c in gp.columns if f"|{ph}|" in c]
        if cols:
            res.phase_auc[ph] = test_feature_set(gp[cols], yb, games, n_perm=0, n_boot=0)["auc"]
    # Which glove measurements separate the classes most (standardized mean difference).
    Xs = gp.fillna(gp.median())
    d = (Xs[yb == 1].mean() - Xs[yb == 0].mean()) / (Xs.std() + 1e-9)
    res.top_features = [[k, float(v)] for k, v in d.abs().sort_values(ascending=False).head(8).items()]
    res.top_features = [[k, float(d[k])] for k, _ in res.top_features]
    if crops:
        idx = [i for i in gp.index if i in crops]
        lab_of = dict(zip(gp.index, yb))
        for cls, label in ((1, res.positive_label), (0, res.negative_label)):
            arr = [crops[i].astype(np.float32) for i in idx if lab_of[i] == cls]
            if arr:
                res.mean_crops[label] = np.mean(arr, 0).astype(np.uint8)
    res.verdict = verdict(res)
    return res


def verdict(res: TellResult) -> str:
    g = res.sets.get("Glove (pose)", {})
    c = res.sets.get("Body without glove (control)", {})
    px = res.sets.get("Glove (pixels)", {})
    best = max([g, px], key=lambda m: m.get("auc", 0)) if px else g
    if not best:
        return "Not enough clips to test."
    sig = best["perm_p"] < 0.01 and best["auc"] >= 0.60
    # "Better than the body" only when the paired interval for glove - body excludes zero.
    lo = best.get("auc_diff_lo", float("nan"))
    beats = lo > 0 if np.isfinite(lo) else best["auc"] > c.get("auc", 0) + 0.03
    gap = (f", AUC difference {best['auc_diff']:+.2f}, 95% CI {lo:+.2f} to {best['auc_diff_hi']:+.2f}"
           if np.isfinite(lo) else "")
    if sig and beats:
        return (f"Supported: the glove alone separates {res.positive_label} from {res.negative_label} "
                f"(AUC {best['auc']:.2f}, 95% CI {best.get('auc_lo', float('nan')):.2f}-{best.get('auc_hi', float('nan')):.2f}, "
                f"within-game permutation p={best['perm_p']:.3f}), better than the body-only "
                f"control (AUC {c.get('auc', float('nan')):.2f}{gap}).")
    if sig:
        return (f"Partly supported: the glove carries signal (AUC {best['auc']:.2f}, p={best['perm_p']:.3f}), "
                f"but it is not distinguishable from the body-only control (AUC {c.get('auc', float('nan')):.2f}{gap}), "
                "so the tell may be the whole delivery, not just the glove.")
    if best["perm_p"] < 0.05 and best["auc"] >= 0.55:
        return (f"Weak evidence: glove-only AUC {best['auc']:.2f} (p={best['perm_p']:.3f}) beats chance, "
                "but not strongly enough to call it a reliable tell. More clips (or more permutations) "
                "would settle it.")
    return (f"Not supported in these clips: glove-only AUC {best['auc']:.2f} (p={best['perm_p']:.3f}). "
            "Either the tell was fixed, it is too subtle for broadcast video, or it is in a part of the "
            "delivery the camera does not show well.")


def crop_panel(res: TellResult, path: Path) -> Path | None:
    """Mean glove crop per class (rows) across the 6 moments, plus an amplified difference row."""
    if len(res.mean_crops) < 2:
        return None
    a, b = res.mean_crops[res.positive_label], res.mean_crops[res.negative_label]
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    diff = (255 * diff / (diff.max() + 1e-9)).astype(np.uint8)
    diff = np.stack([cv2.applyColorMap(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_VIRIDIS) for d in diff])
    up = lambda row: np.hstack([cv2.resize(x, (128, 128), interpolation=cv2.INTER_NEAREST) for x in row])
    cv2.imwrite(str(path), np.vstack([up(a), up(b), up(diff)]))
    return path


GLOVE_LABELS = {
    "wrist_x": "Glove side-to-side", "wrist_y": "Glove height (lower = larger)", "angle": "Glove angle",
    "spread": "Glove spread", "thumb_open": "Thumb opening", "tilt": "Glove tilt",
    "to_chest": "Glove distance from chest", "to_throw_hand": "Glove to throwing hand",
    "elbow_flare": "Glove elbow flare", "rest_y": "Resting glove height", "motion": "Glove movement",
}


def label(feature: str) -> str:
    """'glove_wrist_y|set|1' -> 'Glove height (lower = larger) · set, step 2'."""
    name, ph, b = feature.split("|")
    base = GLOVE_LABELS.get(name.replace("glove_", ""), name)
    return f"{base} · {ph}" + (f", step {int(b) + 1}" if b != "" else "")
