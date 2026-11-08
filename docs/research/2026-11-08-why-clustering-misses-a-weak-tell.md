# 2026-11-08: Is ARI ≈ 0 evidence against the glove tell?

**Question.** On real data (notebook 05, Nov 5 run) the supervised glove-pose model gets AUC 0.60 (p ≤ 0.004) on fastball vs slider, while the unsupervised k=2 GMM on the same features gets ARI 0.006–0.010. The README reads that as "unsupervised clustering finds nothing". For the supervised vs unsupervised comparison in #5 and the paper (#8): what ARI should we expect if the AUC-0.60 signal is real, and how do we compare the two sides fairly?

## Findings

1. **Even a perfect clusterer would score ARI ≈ 0.02 on a signal this weak.**
   I simulated it (script below). Take two balanced classes that differ along one direction by d′ = √2·Φ⁻¹(AUC), which is the separation where the best possible classifier gets that AUC. Then split the 474 points at the midpoint along that direction, the best 2-way partition there is. Over 200 runs:

   | Best-possible AUC | Oracle split ARI (95% range) | Oracle NMI | GMM after PCA, signal in 1 of 40 dims |
   |---|---|---|---|
   | 0.55 | 0.005 (−0.002 to 0.024) | 0.005 | 0.000 |
   | **0.60** | **0.020 (0.002 to 0.048)** | 0.016 | 0.000 |
   | 0.70 | 0.082 (0.044 to 0.130) | 0.061 | 0.000 |
   | 0.80 | 0.201 (0.130 to 0.277) | 0.152 | 0.008 |
   | 0.90 | 0.403 (0.318 to 0.496) | 0.316 | 0.011 |

   So our observed ARI of 0.006–0.010 sits inside the range an *oracle* would get at AUC 0.60. ARI ≈ 0 is not evidence against the tell. It is what a real but weak tell looks like on this metric. ARI and NMI shrink roughly quadratically with separation, while AUC moves linearly, so the two numbers are not on comparable scales.

2. **A GMM picks the largest variance, not the label.**
   The last column is a toy model of our setup: one label direction plus 39 nuisance dimensions (five with double variance, a stand-in for camera angle, park, and game), standardized, PCA to 8 components, then a 2-component GMM. It recovers nothing even at AUC 0.90. PCA keeps the high-variance directions, and the GMM then splits along whatever dominates. Jin & Wang (2016) make the same point formally for high-dimensional clustering: when only a few features carry class information, classical PCA-plus-k-means struggles, and they first *select* informative features (by KS score with a Higher Criticism threshold) before the SVD. In our data, game-to-game camera differences are a plausible dominant direction, and the season split (AUC 0.59 in 2025 vs 0.51 in 2024) suggests nonstationary nuisance as well.

3. **"Does clustering recover pitch type" is the wrong question for #5.**
   von Luxburg, Williamson & Guyon (2012) argue that a clustering can only be judged against what it is for. Here the purpose is "does the glove carry pitch information that a model can find without labels?" That splits into two testable questions: (a) does the unsupervised *representation* keep the information, and (b) is it prominent enough to become a cluster. Our ARI only answers (b). The standard way to answer (a) is a linear probe (Alain & Bengio, 2016): freeze the unsupervised embedding (here the scaler + PCA, fit without labels) and train a simple linear classifier on it with the same game-grouped CV. If the probe AUC is near the LightGBM AUC, the information survives unsupervised compression and just isn't dominant. If it collapses to 0.5, PCA has thrown it away.

4. **Put both sides on the same metric.**
   The supervised model's out-of-fold probabilities can be thresholded into a 2-way partition and scored with ARI/NMI too. That gives a "supervised ARI" to set against the GMM's, so the paper compares like with like instead of AUC vs ARI. Combined with the permutation null for ARI already planned in #11, this makes the comparison table symmetric.

## Concrete suggestions for the repo

- **Supervised ARI/NMI and an oracle ceiling.** In `test_feature_set`, add `sup_ari`/`sup_nmi` from the out-of-fold `p` thresholded at 0.5, plus `ari_ceiling` from the closed-form d′ given the observed AUC (or by the simulation below). Show them in the report and README table. → #20.
- **Linear probe and nuisance removal on the unsupervised side.** Add (i) a game-grouped logistic-regression probe on the label-free PCA embedding, and (ii) a variant that centers features within each `game_pk` (and optionally season) before the scaler, PCA and GMM. Report probe AUC and GMM ARI for both. → #21.
- **README wording.** Change "Unsupervised clustering finds nothing" to say ARI is at the level expected for an AUC-0.60 signal, so the unsupervised result is uninformative rather than negative. I left this for the #20 PR so the numbers come with it.
- **Paper (#8).** Present the supervised vs unsupervised comparison as "information retained (probe AUC)" vs "information dominant (ARI)", with ceilings and permutation nulls for both.

## Simulation script

```python
import numpy as np
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rng, n = np.random.default_rng(0), 474
for auc in [0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
    d = np.sqrt(2) * norm.ppf(auc)          # 1-D separation giving this Bayes AUC
    oa, on, ga = [], [], []
    for r in range(200):
        y = rng.integers(0, 2, n)
        s = rng.normal(size=n) + d * y      # the one label-relevant direction
        oracle = (s > d / 2).astype(int)    # best 2-way split along it
        oa.append(ari(y, oracle)); on.append(nmi(y, oracle))
        if r < 40:                          # embed among 39 nuisance dims, 5 with 2x sd
            X = np.c_[s, rng.normal(size=(n, 39)) * np.r_[np.full(5, 2.0), np.ones(34)]]
            Z = PCA(8, random_state=0).fit_transform(StandardScaler().fit_transform(X))
            ga.append(ari(y, GaussianMixture(2, random_state=0, n_init=3).fit_predict(Z)))
    print(auc, round(np.mean(oa), 3), np.quantile(oa, [.025, .975]).round(3),
          round(np.mean(on), 3), round(np.mean(ga), 3))
```

## Sources

- von Luxburg, U., Williamson, R. C., & Guyon, I. (2012). *Clustering: Science or Art?* Proc. ICML Workshop on Unsupervised and Transfer Learning, PMLR 27, 65–79. https://proceedings.mlr.press/v27/luxburg12a.html
- Jin, J., & Wang, W. (2016). *Influential Features PCA for high dimensional clustering.* Annals of Statistics 44(6). https://arxiv.org/abs/1407.5241
- Alain, G., & Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644. https://arxiv.org/abs/1610.01644 (I did not reread the full text this run; cited for the linear-probe idea it introduced.)
- Real-data numbers: `notebooks/05_mason_miller_glove_tell.ipynb`, results section (Nov 5, 2026 run).
