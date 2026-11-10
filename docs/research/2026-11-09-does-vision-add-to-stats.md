# 2026-11-09: How do we show that vision adds anything to the stats model?

**Question.** On the 474 real clips, stats alone get 58.9% accuracy and stats + vision 60.1% (README / notebook 05). `fusion.fit_fusion` builds that number by feeding the stats model's log-probabilities as three extra columns into one LightGBM, next to several hundred pose features, and then compares accuracy. For the paper's "does computer vision help?" claim: is that setup able to show a gain, and how should the gain be tested?

## Findings

1. **Accuracy is the wrong yardstick for a one-point gain.**
   Log loss is a strictly proper scoring rule: it rewards a forecast for being right *and* for being calibrated, and it is minimized only by the true probabilities (Gneiting & Raftery, 2007). Accuracy only looks at the arg-max, so a one-point change on 474 pitches is about five pitches flipping. `fit_fusion` already computes log loss for every model. The paper should lead with it and treat accuracy as secondary.

2. **The comparison is paired, so test it as paired.**
   Stats-only and stats + vision are scored on the same pitches. The right question is whether the per-pitch loss difference `d_i = logloss_stats_i − logloss_fused_i` has a positive mean. Diebold & Mariano (1995) built the standard test for "do two forecasts differ in accuracy" on exactly this loss differential, allowing correlated errors. Our errors are correlated *within games*, so the simplest valid version is a bootstrap that resamples games, not pitches, and reports a 95% interval on mean `d`. This matches the game-level reasoning in the Nov 4 note.

3. **Putting the prior in as a feature can make a calibrated prior worse.**
   I simulated our setup: 474 pitches, 3 classes, a calibrated stats prior, 300 pose features, and the exact `_model()` settings from `fusion.py` (250 trees, learning rate 0.03, 7 leaves). Mean out-of-fold log loss over 20 runs (lower is better):

   | Vision signal in 2 of 300 pose features | Stats prior only | Prior as a feature (current) | Prior as offset, 50 trees | Stacked logistic regression |
   |---|---|---|---|---|
   | none | **0.938** | 1.025 | 0.950 | 0.957 |
   | weak (0.3 logit per SD) | **0.946** | 1.027 | 0.955 | 0.962 |

   With no vision signal at all, the current fusion loses about 0.09 nats against the prior it was given, because the trees spend their budget fitting noise in the pose columns and partly overwrite the prior. Treating the prior as an **offset** (LightGBM `init_score`, so trees only model what the prior misses) or **stacking** (a small multinomial logistic regression on the prior's log-odds plus the out-of-fold vision log-odds; Wolpert 1992 is the classic reference) cuts most of that loss. Neither beats the prior when the vision signal is this weak and this spread out, which is the honest answer to expect. The point is that with the current setup, a real but small vision gain could be hidden by the cost of the fusion itself.

   One practical note: the LightGBM docs list `init_score` for `fit` (shape `[n_samples, n_classes]` for multiclass), but `predict` has no `init_score` argument, so predictions do not include the offset. Add it back yourself: `softmax(model.predict(X, raw_score=True) + log_prior)`. That is what the simulation does. A unit test should pin this down.

4. **What this means for the 60.1% vs 58.9% number.**
   Because the current fusion is biased *against* adding anything, the observed +1.2 points is not inflated by the design. But it is also not tested, and on log loss the fused model may be worse than stats alone. Until the paired game-bootstrap is in, the paper should say "vision adds about one point of accuracy, not yet shown to be significant" rather than "vision helps".

## Concrete suggestions for the repo

- **Paired log-loss test.** In `fit_fusion`, keep the per-pitch log loss for `stats_only` and `stats_plus_vision`, and add `delta_logloss` with a 95% game-bootstrap interval (and the same for accuracy). Show it in the report. → #23.
- **Offset and stacked fusion.** Add `stats_offset_vision` (LightGBM with `init_score=log(prior)`, fewer trees or early stopping on an inner fold) and `stats_stacked_vision` (multinomial logistic regression on the prior and nested out-of-fold vision log-probs) next to the current concat model. Keep the concat model for comparison. → #24.
- **Paper (#7/#8).** In methods, describe the fusion as a residual model over a fixed stats prior. In results, lead with the paired log-loss difference and its interval.

## Simulation script (abridged)

```python
# 474 pitches, 3 classes, calibrated prior, 300 pose features; signal vs in 2 of them.
pl = rng.normal(size=(n, K)) * 0.8; X = rng.normal(size=(n, P))
true = pl.copy(); true[:, 0] += vs * X[:, 0]; true[:, 1] += vs * X[:, 1]
y = [rng.choice(K, p=p) for p in softmax(true)]; lp = np.log(softmax(pl))
# per fold f:
concat = model().fit(np.c_[lp[tr], X[tr]], y[tr]).predict_proba(np.c_[lp[te], X[te]])
offset = softmax(model(50).fit(X[tr], y[tr], init_score=lp[tr]).predict(X[te], raw_score=True) + lp[te])
# stack: inner 4-fold OOF vision probs on train -> LogisticRegression on [lp, log vision]
```

`model()` uses the same hyperparameters as `fusion._model`. The full script runs in a few minutes.

## Sources

- Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102(477), 359–378. Tech-report version: https://apps.dtic.mil/sti/tr/pdf/ADA459827.pdf
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics 13(3), 253–263. https://econpapers.repec.org/RePEc:bes:jnlbes:v:13:y:1995:i:3:p:253-63
- LightGBM 4.6.0 docs, `LGBMModel.fit` (`init_score`) and `predict` (`raw_score`). https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMModel.html
- Wolpert, D. H. (1992). *Stacked generalization.* Neural Networks 5(2), 241–259. (Cited for the stacking idea; I did not reread it this run.)
- Real-data numbers: README "Result: Mason Miller glove tell" and notebook 05 results section (Nov 5, 2026 run).
