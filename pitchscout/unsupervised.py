"""Unsupervised layer: discover delivery clusters from pose alone, then measure how
well they line up with the real pitch types.

The clustering never sees pitch labels. Labels are used only afterwards, to
(a) score alignment (purity, ARI, NMI) and (b) map clusters to a pitch
distribution on training folds so the result can be scored with the same
accuracy / log loss as the supervised model on the same game-held-out folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, adjusted_rand_score, log_loss,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .fusion import N_BINS
from .pose import JOINTS


@dataclass
class UnsupervisedResult:
    k: int
    bic: dict
    metrics: dict
    alignment: dict
    composition: pd.DataFrame     # cluster x pitch type share
    cluster_sizes: pd.Series
    embedding: np.ndarray         # (n, 2) PCA map for plotting
    clusters: np.ndarray          # cluster id per clip (full-data fit)
    mean_pose: dict               # cluster -> (N_BINS, joints, 2)
    proba: np.ndarray
    proba_with_stats: np.ndarray


def _reducer(n_feat: int, n_rows: int, max_comp: int = 12) -> object:
    return make_pipeline(StandardScaler(), PCA(n_components=min(max_comp, n_feat, n_rows - 1), random_state=0))


def choose_k(Z: np.ndarray, k_range=range(2, 13), seed: int = 0) -> tuple[int, dict]:
    bic = {}
    for k in k_range:
        if k >= len(Z):
            break
        g = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=3).fit(Z)
        bic[k] = float(g.bic(Z))
    return min(bic, key=bic.get), bic


def _fit(Xtr: np.ndarray, k: int | None, seed: int = 0):
    red = _reducer(Xtr.shape[1], len(Xtr)).fit(Xtr)
    Z = red.transform(Xtr)
    if k is None:
        k, _ = choose_k(Z, seed=seed)
    gmm = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=3).fit(Z)
    return red, gmm


def _purity(y, c) -> float:
    ct = pd.crosstab(c, y)
    return float(ct.max(axis=1).sum() / ct.values.sum())


def fit_unsupervised(pose_df: pd.DataFrame, y: np.ndarray, games: np.ndarray, classes: list[str],
                     prior: np.ndarray | None = None, n_folds: int = 5, k: int | None = None,
                     alpha: float = 1.0) -> UnsupervisedResult:
    X = pose_df.fillna(pose_df.median()).to_numpy()
    n, K = len(X), len(classes)
    cidx = {c: i for i, c in enumerate(classes)}

    # Full-data fit (labels unused) for the descriptive outputs.
    red, gmm = _fit(X, k)
    Z = red.transform(X)
    k_used = gmm.n_components
    _, bic = choose_k(Z) if k is None else (k, {})
    clusters = gmm.predict(Z)
    emb = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(X))

    alignment = {
        "adjusted_rand": float(adjusted_rand_score(y, clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, clusters)),
        "purity": _purity(y, clusters),
        "majority_class_share": float(pd.Series(y).value_counts(normalize=True).iloc[0]),
        "silhouette": float(silhouette_score(Z, clusters)) if 1 < k_used < n else float("nan"),
        "pca_components": int(red[-1].n_components_),
        "pca_variance_explained": float(red[-1].explained_variance_ratio_.sum()),
    }
    comp = pd.crosstab(clusters, y, normalize="index").reindex(columns=classes).fillna(0)
    sizes = pd.Series(clusters).value_counts().sort_index()

    # Mean normalized pose per cluster from the joint coordinate columns.
    mean_pose = {}
    for c in range(k_used):
        m = pose_df[clusters == c].mean()
        arr = np.zeros((N_BINS, len(JOINTS), 2))
        for b in range(N_BINS):
            for j, jn in enumerate(JOINTS):
                arr[b, j, 0] = m.get(f"{jn}|x|{b}", np.nan)
                arr[b, j, 1] = m.get(f"{jn}|y|{b}", np.nan)
        mean_pose[c] = arr

    # Cross-validated predictive use: same game-grouped folds as the supervised model.
    uniq = np.unique(games)
    rng = np.random.default_rng(1)
    fold = pd.Series(rng.integers(0, n_folds, len(uniq)), index=uniq).loc[games].to_numpy()
    proba = np.zeros((n, K))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0:
            continue
        r, g = _fit(X[tr], k_used)          # clustering on train rows, no labels
        post_tr = g.predict_proba(r.transform(X[tr]))
        Y = np.zeros((tr.sum(), K)); Y[np.arange(tr.sum()), [cidx[v] for v in y[tr]]] = 1
        M = post_tr.T @ Y + alpha          # soft cluster -> pitch counts (labels used only here)
        M /= M.sum(1, keepdims=True)
        proba[te] = g.predict_proba(r.transform(X[te])) @ M
    proba /= proba.sum(1, keepdims=True)

    mix = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(1e-6).to_numpy()
    if prior is not None:
        # Combine with the stats prior (treat cluster evidence as a likelihood ratio vs the base mix).
        ws = prior * proba / mix
        ws /= ws.sum(1, keepdims=True)
    else:
        ws = proba

    def met(p):
        return {"log_loss": float(log_loss(y, p, labels=classes)),
                "accuracy": float(accuracy_score(y, np.asarray(classes)[p.argmax(1)]))}

    metrics = {"clusters_only": met(proba)}
    if prior is not None:
        metrics["stats_plus_clusters"] = met(ws)
    return UnsupervisedResult(k_used, bic, metrics, alignment, comp, sizes, emb, clusters,
                              mean_pose, proba, ws)
