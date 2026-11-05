# 2026-11-05: Can we trust MediaPipe's glove-hand points?

**Question.** `glove_pose_features` in `pitchscout/glove.py` builds the glove "pose" signal from MediaPipe Pose landmarks 15 (wrist), 17 (pinky), 19 (index) and 21 (thumb) on the glove side. Features such as `glove_angle`, `glove_spread`, `glove_tilt` and `glove_thumb_open` depend on the three finger points. When the hand is inside a glove, at broadcast distance, with motion blur, how much of that signal is real, and what should we change before reading anything into issue #4?

## Findings

1. **The finger points are partly guesses, and the model tells us when.**
   The BlazePose GHUM model card says each landmark comes with a visibility score for "in frame and not occluded", and that hidden points were labeled with a *best guess* during training. So when the glove hides the fingers, the model still returns a plausible hand shape learned from bare-hand training data, not what the glove is doing. The card evaluates only at the region level (PCK@0.2 of torso size, Full model 91.8% averaged over 14 regions) and gives no separate accuracy for the pinky, index or thumb points. A 20%-of-torso tolerance is also far coarser than the few-centimeter glove changes a tell would involve.

2. **Our setting is outside the model's stated scope.**
   The same card lists people beyond about 4 m from the camera as out of scope and says that motion and poor lighting add jitter. A center-field broadcast view of the mound fits neither condition. That does not make the wrist useless, but it means the finger points are the weakest link and their noise will be large next to any real glove signal.

3. **Visibility is stored but never used.**
   `pose.detect_people` saves `p.visibility` as the third channel of `PoseTrack.frames`, and `db.py` writes it to `pose_frames.visibility`. But `glove_pose_features` and `glove_crops` use only x and y, and `interp_nans` fills gaps without marking them. Frames where the glove hand was occluded or lost therefore count exactly as much as clean frames. This is the cheapest thing to fix, and it matters for #4: if glove visibility differs by pitch type (for example, a different glove set that hides the hand more), a classifier could learn the *guessing pattern* rather than the glove.

4. **Hand detectors lose recall under occlusion and blur, while staying "confident".**
   Pu, Chong & Lim tested MediaPipe Hands with metamorphic perturbations. Precision stays high (about 90% up to moderate occlusion, above 40% even with the whole hand covered), but recall drops sharply: roughly halved at heavy joint occlusion, about 40% lower on Panoptic when only the thumb is covered, and badly hurt by diagonal motion blur. Their test is on the dedicated hand model, not the Pose hand points, and does not include gloves, but the direction is the same: occluded hands yield confident-looking but wrong points.

5. **Blur on pitching video is a known, fixable problem for body pose.**
   Bright, Chen & Zelek (Waterloo) show that motion blur degrades 2D/3D pitcher pose and that training with synthetic blur on high-optical-flow patches plus pseudo-labeled slow-motion pitching video cut 2D loss from 1.05 to 0.48 (their Table 5). They do not report per-keypoint results for hands. For us, the takeaway is that the windup frames we use (before release) are less blurred than the arm-acceleration frames, so the `cutoff_s` window helps, but the glove hand still moves fast during the leg lift.

6. **A stronger off-the-shelf option for hand points exists.**
   RTMW (Jiang et al., 2024) predicts the 133-point COCO-WholeBody layout, including 21 points per hand, and is released in MMPose. Hand AP on COCO-WholeBody is 66.3 for RTMW-l at 384×288, versus 62.1 for DWPose-l and 57.9 for RTMPose-l. CPU latency is about 48 ms per crop at that size with ONNXRuntime. That is still bare-hand accuracy on close-up photos, so on a gloved hand at broadcast distance it will also guess, but it gives a second, independent opinion on where the glove hand is.

## Concrete suggestions for the repo

- **Gate on visibility.** In `glove_pose_features`, mask glove points whose visibility is below a threshold (start at 0.5), compute the features from the remaining frames, and add `glove_vis_mean|{phase}|` as an explicit feature. Report the share of clips with enough visible glove frames. → #14.
- **Check that visibility is not itself the tell.** Run the tell test on `glove_vis_mean` alone. If it separates fastballs from offspeed, the pose result is suspect and the paper should say so. → #14.
- **Wrist-only ablation.** Rerun the glove pose test using only wrist-based features (`wrist_x/y`, `to_chest`, `to_throw_hand`, `elbow_flare`, `rest_y`, `motion`) and drop the finger-dependent ones. If AUC holds, the finding does not hinge on guessed fingers. → #14.
- **Second keypoint model on a subset.** On ~50 real clips, run RTMW (via MMPose or the `rtmlib` ONNX wrapper) on the pitcher crop and compare its glove-wrist track to MediaPipe's. Large disagreement means the pixel features (`glove_crops`) are the more trustworthy glove evidence for #4/#5. → #15.
- **Paper (#7/#8).** In methods, say that finger landmarks under a glove are model priors, describe the visibility gate, and report the wrist-only ablation as a robustness check.

## Sources

- Google MediaPipe. *Model Card: BlazePose GHUM 3D.* https://dcmpx.remotevs.com/com/googleapis/storage/SL/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf (mirror of the MediaPipe assets bucket; read this run)
- Pu, M., Chong, C. Y., & Lim, M. K. *Robustness Evaluation in Hand Pose Estimation Models using Metamorphic Testing.* arXiv:2303.04566. https://arxiv.org/abs/2303.04566
- Bright, J., Chen, Y., & Zelek, J. *Mitigating Motion Blur for Robust 3D Baseball Player Pose Modeling for Pitch Analysis.* arXiv:2309.01010. https://arxiv.org/abs/2309.01010 (the PDF's text states 58% for the 2D improvement while the abstract says 54.2%; I quote the table values.)
- Jiang, T., Xie, X., & Li, Y. (2024). *RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation.* arXiv:2407.08634. https://arxiv.org/abs/2407.08634

Not verified: I did not find any published work on detecting the glove itself (as opposed to the bare hand) in broadcast baseball video. If one exists it belongs in #6.
