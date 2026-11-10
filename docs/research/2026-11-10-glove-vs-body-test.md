# 2026-11-10: Is "the glove beats the body" actually tested?

**Question.** The headline claim for #4 and the paper is that the *glove* gives the pitch away, beyond what the rest of the delivery shows. The current numbers are glove pose AUC 0.598 vs body-without-glove AUC 0.564 on FF vs SL, and with the wrist-only robustness check, 0.576 vs 0.564. `glove.verdict` turns that gap into the sentence "better than the body-only control". Each AUC has its own permutation p-value against chance, but nothing tests the gap itself. What is the right test?

## Two different questions

There are two claims hidden in "the glove beats the body", and they need different tests.

- **(A) Glove alone vs body alone.** Two separate (non-nested) models are scored on the same pitches. Is one more accurate than the other?
- **(B) Does the glove add anything once you already know the body?** This is a nested comparison: body + glove vs body. It is the question the paper actually cares about, because glove and body come from the same skeleton and are correlated. Part of any "glove" signal could just be body mechanics seen through the glove arm.

## Findings

1. **For (A), a paired comparison of correlated AUCs is fine.**
   DeLong, DeLong & Clarke-Pearson (1988) give the standard nonparametric test for two correlated AUCs on the same cases. Demler, Pencina & D'Agostino (2012) confirm it is appropriate for *non-nested* models. Our predictions are out-of-fold and correlated within games, so the simplest valid version reuses the game bootstrap from #10: resample games, recompute both AUCs on the same resample, and report the interval of the difference. With a gap of 0.012 (wrist-only) on 441 pitches, I expect that interval to include zero comfortably. Today's README already says the gap is "barely above", but the paper should show the interval.

2. **For (B), testing the AUC difference is the wrong tool.**
   Demler et al. (2012) show that when the added variables carry no information, the AUC difference between nested models is a degenerate statistic: skewed, piled up near zero, and not normal at any sample size. The DeLong test then becomes very conservative and loses most of its power. In one of their settings, power was 0.47 vs 0.94 for a Wald test. Pepe, Kerr, Longton & Wang (2013) go further. "Does the new marker improve performance?" is the same null hypothesis as "the new marker is not associated with the outcome given the old ones", so the efficient move is to test that association directly. Their simulations found ΔAUC tests with a size below 0.005 at a nominal 0.05. Their recommendation: test association, then *estimate* ΔAUC (or Δlog-loss) with a confidence interval, and do not test ΔAUC itself.

3. **With LightGBM there is no coefficient to Wald-test, but a conditional permutation does the same job.**
   The flexible-model analogue of "test β_glove = 0" is a test of glove ⟂ pitch type | body. A plain label shuffle tests the wrong null, and so does shuffling glove rows across pitches, because that also breaks the glove–body correlation. That inflates the apparent value of the real glove columns. The conditional randomization test of Candès, Fan, Janson & Lv (2018) handles this. Draw fake glove features from a model of glove given body, refit, and see how often the fake glove does as well as the real one. A cheap version that fits our code:
   - inside each training fold, regress each glove feature on the body features (ridge);
   - make a null copy by permuting the residuals **within game** and adding them back to the fitted values;
   - refit `body + null glove` with the same folds and record out-of-fold log loss (and AUC);
   - p-value = share of null runs whose loss is at least as good as `body + real glove`.

   This keeps the part of the glove explained by the body and destroys only the glove-specific part, which is exactly the "beyond the body" claim. The linear-Gaussian residual model is an approximation; the note in the paper should say so.

4. **Effect size: report the nested gain with an interval.** After the test, report ΔAUC and Δlog-loss of body + glove over body, with a game-bootstrap interval, as Pepe et al. recommend. This reuses the paired machinery planned in #23 for stats vs vision.

## Concrete suggestions for the repo

- **Paired glove vs body (A).** In `run_tell_test`, compute out-of-fold predictions for both sets on the same folds and report `auc_diff` with a game-bootstrap 95% interval. Make `verdict` say "better than the body-only control" only when that interval excludes zero. → #25.
- **Conditional test of glove beyond body (B).** Add a "Body + glove" set and a conditional-randomization p-value (ridge residuals, within-game permutation, about 200 draws), plus Δlog-loss and ΔAUC intervals vs body only. → #26.
- **Paper (#7/#8).** State the primary claim as (B), report the conditional p-value and the nested effect size, and present (A) as descriptive.

## Sources

- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). *Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.* Biometrics 44(3), 837–845. https://scholars.duke.edu/publication/712661
- Demler, O. V., Pencina, M. J., & D'Agostino, R. B. Sr. (2012). *Misuse of DeLong test to compare AUCs for nested models.* Statistics in Medicine 31(23), 2577–2587. https://pmc.ncbi.nlm.nih.gov/articles/PMC3684152
- Pepe, M. S., Kerr, K. F., Longton, G., & Wang, Z. (2013). *Testing for improvement in prediction model performance.* Statistics in Medicine 32(9), 1467–1482. https://pmc.ncbi.nlm.nih.gov/articles/PMC3625503
- Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). *Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection.* JRSS B 80(3), 551–577. https://arxiv.org/abs/1610.02351 (cited for the conditional randomization test it introduces; I did not reread the full text this run.)
- Current numbers: README "Result: Mason Miller glove tell" (Nov 5 run, wrist-only check added Nov 8) and `pitchscout/glove.py` (`run_tell_test`, `verdict`).
