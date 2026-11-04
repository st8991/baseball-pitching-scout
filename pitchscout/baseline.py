"""Stats-only pitch prediction and tendency tables."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .features import situation_features, time_split

COUNTS = [f"{b}-{s}" for b in range(4) for s in range(3)]


def tendencies(df: pd.DataFrame) -> dict:
    """Pitch mix overall, by count, and by batter handedness (row-normalized shares)."""
    pt = df["pitch_type"]
    count = df["balls"].clip(0, 3).astype(str) + "-" + df["strikes"].clip(0, 2).astype(str)
    by_count = pd.crosstab(count, pt, normalize="index").reindex(COUNTS).fillna(0)
    n_count = count.value_counts().reindex(COUNTS).fillna(0).astype(int)
    by_hand = pd.crosstab(df["stand"], pt, normalize="index")
    ab = df.groupby(["game_pk", "at_bat_number"])["pitch_type"]
    prev = ab.shift(1)
    after = pd.crosstab(prev.dropna(), pt[prev.notna()], normalize="index")
    return {
        "mix": pt.value_counts(normalize=True),
        "by_count": by_count,
        "n_by_count": n_count,
        "by_hand": by_hand,
        "after_prev": after,
    }


@dataclass
class BaselineResult:
    classes: list[str]
    model: lgb.LGBMClassifier
    train_idx: np.ndarray
    test_idx: np.ndarray
    metrics: dict = field(default_factory=dict)
    importance: pd.Series | None = None
    oof_proba: np.ndarray | None = None  # out-of-fold probs on all rows (for stacking)


def _metrics(y, proba, classes) -> dict:
    return {
        "log_loss": float(log_loss(y, proba, labels=classes)),
        "accuracy": float(accuracy_score(y, np.asarray(classes)[proba.argmax(1)])),
    }


def _model(seed=0) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, verbose=-1,
    )


def _fit_es(X, y, groups, seed=0) -> lgb.LGBMClassifier:
    """Pick the number of trees by early stopping on the latest ~15% of the given games,
    then refit on all of them. Keeps the model from memorizing noise on small samples."""
    order = pd.unique(groups)
    val_games = set(order[-max(1, len(order) * 15 // 100):])
    va = np.isin(groups, list(val_games))
    m = _model(seed).set_params(n_estimators=1000)
    m.fit(X[~va], y[~va], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    best = max(20, int(m.best_iteration_ or 300))
    return _model(seed).set_params(n_estimators=best).fit(X, y)


def fit_baseline(df: pd.DataFrame, n_folds: int = 5) -> BaselineResult:
    X = situation_features(df)
    y = df["pitch_type"].to_numpy()
    classes = sorted(np.unique(y))
    tr, te = time_split(df)

    # Naive prior: the pitcher's overall mix from training games.
    mix = pd.Series(y[tr]).value_counts(normalize=True).reindex(classes).fillna(1e-6).to_numpy()
    naive = np.tile(mix, (len(te), 1))
    # Count-only prior: mix by count from training games.
    cnt = X["count"].astype(str).to_numpy()
    ct = pd.crosstab(cnt[tr], y[tr]).reindex(columns=classes).fillna(0) + 1
    ct = ct.div(ct.sum(axis=1), axis=0)
    count_only = np.vstack([ct.loc[c].to_numpy() if c in ct.index else mix for c in cnt[te]])

    g = df["game_pk"].to_numpy()
    model = _fit_es(X.iloc[tr], y[tr], g[tr])
    proba = model.predict_proba(X.iloc[te])

    # Out-of-fold probabilities over all rows, grouped by game, for the fusion model.
    oof = np.zeros((len(df), len(classes)))
    games = df["game_pk"].unique()
    rng = np.random.default_rng(0)
    fold_of_game = dict(zip(games, rng.integers(0, n_folds, len(games))))
    folds = df["game_pk"].map(fold_of_game).to_numpy()
    for k in range(n_folds):
        m = _fit_es(X[folds != k], y[folds != k], g[folds != k], seed=k)
        p = m.predict_proba(X[folds == k])
        # Align classes in case a fold is missing one.
        oof[np.ix_(folds == k, [classes.index(c) for c in m.classes_])] = p

    imp = pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
    return BaselineResult(
        classes=classes, model=model, train_idx=tr, test_idx=te,
        metrics={
            "naive_mix": _metrics(y[te], naive, classes),
            "count_only": _metrics(y[te], count_only, classes),
            "stats_model": _metrics(y[te], proba, classes),
            "n_train": int(len(tr)), "n_test": int(len(te)),
        },
        importance=(imp / imp.sum()).sort_values(ascending=False),
        oof_proba=oof,
    )
