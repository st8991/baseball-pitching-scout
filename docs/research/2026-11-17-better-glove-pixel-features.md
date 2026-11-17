# 2026-11-17: Can better pixel features see what HOG missed in the glove?

**Question.** The glove-pixel arm of the tell test summarizes six 64×64 crops around the glove hand with HOG plus an HSV histogram, compresses them with PCA, and scores AUC 0.469 (p 0.74) on FF vs rest. That is no signal at all. Meanwhile the pose arm depends on MediaPipe finger points that are mostly guesses inside a glove (Nov 5 note). So the pixels are the only *direct* look at the glove. Is the pixel null a property of the glove, or of the features we chose? What should we try next, given about 450–1,900 clips?

## Findings

1. **HOG on a 64×64 crop is a weak, low-capacity summary.** HOG pools gradient directions into coarse cells. With our settings (32×32 blocks, 16-pixel cells, 9 bins) the whole crop is a few dozen gradient histograms, plus a 24-bin color histogram. That can capture "glove up vs down" but not small changes in glove *shape* (pocket open, fingers spread under the leather, wrist angle). Broadcast resolution makes the glove only a few dozen pixels across, so the crop is already upsampled. A null from HOG says little about whether the glove differs.

2. **Frozen self-supervised features are the standard next step for small labeled sets.** DINOv2 (Oquab et al., 2023) trains vision transformers without labels and reports that its *frozen* features plus a linear probe come close to fine-tuning: 86.5% ImageNet top-1 for the largest model, with fine-tuning adding only about 2 points. It is particularly strong on fine-grained recognition (iNaturalist, FGVC Aircraft, Stanford Cars), where classes differ in small shape details, the kind of difference a glove tell would be. The smaller ViT-S/B/L models are distilled from the large one, and the small one (ViT-S/14) runs on CPU for a few thousand crops. Because the features are label-free, they slot into both arms of the study: the supervised probe and the unsupervised GMM/probe from #21.

3. **The tell might be motion, not a still pose.** Many described pitcher tells are about *how* the glove moves in the set: a wiggle, a pump, a pause, a hand re-gripping inside the glove. The current crops are six still frames, so a timing or motion cue is invisible to HOG and to any per-frame embedding. Two cheap ways to look at motion:
   - *Dense optical flow* inside the glove crop between consecutive frames (OpenCV's Farnebäck flow is already available through `opencv-python-headless`), summarized as mean magnitude and direction per phase. This is the glove-specific version of the existing `glove_motion` pose feature, without relying on finger keypoints.
   - *Video self-supervised features.* VideoMAE (Tong et al., 2022) is a masked-autoencoder video model that the authors show trains well on as few as 3,000–4,000 videos and reports 91.3% on UCF101 without extra data. That is still more than we have, and it brings a heavy dependency, so it is a later option. Use a pretrained checkpoint for frozen features rather than training.

4. **Stronger pixel features make shortcut risk worse.** Any rich image embedding will also encode park, lighting and broadcast crew. Zech et al.'s hospital fingerprint from the Nov 11 note applies directly. A DINOv2 or flow result therefore has to go through the same venue/season checks (#29, #30) and the within-game permutation before it counts. The crop also contains jersey and background pixels, so center the crop on the glove and keep it tight.

## Concrete suggestions for the repo

- **Frozen DINOv2 glove embeddings.** Add an optional "Glove (DINOv2)" feature set: resize each glove crop to 224×224, take the ViT-S/14 CLS embedding (and optionally mean patch tokens), average per phase, PCA to about 20 dimensions, and run it through `test_feature_set` like the HOG set. Make the `torch` dependency optional so CI stays light. → #38.
- **Glove optical-flow features.** From the existing crop frames, compute Farnebäck flow between consecutive frames within each phase and summarize magnitude, dominant direction and "stillness" (fraction of low-flow frames). Add them as a "Glove (motion)" set. → #39.
- **Paper (#7).** Describe the pixel arm as "handcrafted (HOG) vs learned (DINOv2) vs motion (flow)" so that a null result is informative and not just a weak baseline.

## Sources

- Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193. https://ar5iv.labs.arxiv.org/html/2304.07193
- Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.* NeurIPS 2022. https://arxiv.org/abs/2203.12602
- Zech, J. R., et al. (2018). PLOS Medicine 15(11), e1002683 (see the 2026-11-11 note).
- Current pixel pipeline: `pitchscout/glove.py` (`glove_crops`, `crop_features`, `_HOG`). Result from the README and notebook 05 (Nov 5 run): glove pixels AUC 0.469, p 0.74.
