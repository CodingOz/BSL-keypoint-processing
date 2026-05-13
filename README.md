# BSL Keypoint Processing

A lightweight pipeline for recognising British Sign Language (BSL) fingerspelling from MediaPipe hand keypoints. Trains in roughly two minutes on a mid-range laptop CPU, reaches 97.6 percent mean accuracy across eleven BSL classes under stratified five-fold cross-validation, and 98.6 percent under leave-one-signer-out validation.

The project accompanies the undergraduate dissertation *Supervised Deep Learning Approaches to British Sign Language Recognition* (Strong, 2026). See [USER_MANUAL.md](USER_MANUAL.md) for end-to-end operational instructions; this file is the project overview.

## Headline results

| Variant | Features per frame | Total features | Stratified 5-fold accuracy | Leave-one-signer-out accuracy | Parameters |
| --- | --- | --- | --- | --- | --- |
| v0 (raw coordinates) | 84 | 840 | 0.924 | 0.970 | ~31k |
| v5 (12-point distances) | 66 | 660 | **0.976** | **0.986** | ~26k |
| v7 (inter-hand 6x6 only) | **36** | **360** | 0.969 | 0.976 | ~23k |

All figures: V2 keypoints, seed 42, full processing pipeline, augmentation enabled, 327 recordings across 11 classes from 36 signers. The most parsimonious variant (v7) operates on 360 features and reaches 96.9 percent stratified accuracy with under 26,000 parameters; training a full five-fold sweep takes roughly 80 seconds on a 15-watt mobile CPU with no discrete GPU. The headline finding is that inter-hand geometric structure is informationally sufficient for accurate BSL fingerspelling recognition, contrary to the implicit assumption in pixel-based SLR literature that higher dimensionality and deeper networks produce better representations.

## What the project does

Given a video of a person fingerspelling a single BSL letter, the system extracts 21 hand keypoints per hand via MediaPipe Hands, processes those keypoints through a five-stage pipeline, computes a geometric feature vector, and classifies the sign with a small 1D convolutional network. The whole flow runs on consumer hardware without a discrete GPU.

Eleven letters are currently supported: A, B, E, I, J, N, O, P, S, T, U. These were selected to cover the principal phonological categories of BSL fingerspelling: the five contact-point vowels (A, E, I, O, U), palm-based finger-count signs (N, T), the dynamic letter J, and structurally distinct configurations (B, P, S). The vocabulary can be extended without architectural modification.

## Pipeline

The system is decomposed into five sequential stages. Each stage writes a self-contained JSON corpus to disk so its output can be inspected directly and any stage can be re-run in isolation. The five stages are:

**Stage 1: Cleaning.** Detects and removes two recurring MediaPipe failure modes. Hand arrays containing more than 21 landmarks (the "ten-fingered hand" artefact) are flagged structurally. Single-frame label swaps between left and right hands are flagged behaviourally using a union of positional and kinematic signals. Anomalous frames have their hand data emptied and the original frame copied to file metadata for audit. Across the 327-file unprocessed corpus, structural cleaning alone affects 62.1 percent of files.

**Stage 2: Interpolation.** Fills missing keypoints within the signing region using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial), applied independently to each of the 84 (hand x landmark x coordinate) trajectories. PCHIP is preferred over linear interpolation (creates straight-line paths through coordinate space) and cubic spline (Runge-phenomenon overshoot at held-configuration plateaus). A separate routine fills up to five frames of asymmetric end-of-signing-region gaps by constant-velocity extrapolation.

**Stage 3: Temporal cropping.** Crops each recording to its stroke phase (Kendon, 1980; Kita et al., 1998) via a three-stage unsupervised procedure: valley detection on smoothed momentum, refinement via a composite momentum-and-centroid stability score, and proximity-guided boundary expansion using an adaptive inter-hand-distance threshold. Stage ablation analysis identifies temporal cropping as the most consequential stage in the pipeline, with mean accuracy reduction of 8.4 percentage points when it is disabled.

**Stage 4: Temporal normalisation.** Resamples every recording to 10 frames using PCHIP. Ten frames is short enough to keep the per-sample feature vector tractable and to avoid up-sampling-induced smoothness on shorter signs, but long enough to preserve the temporal dynamics that distinguish signs.

**Stage 5: Spatial normalisation.** Applies a shared two-hand transform to all 42 landmarks: translation to the inter-wrist midpoint, then scaling by the mean wrist-to-middle-fingertip distance. Both transforms use a single shared reference frame derived from both hands together rather than per-hand reference frames, since BSL letter identity is encoded in the inter-hand geometry that independent per-hand normalisation would destroy.

After Stage 5, each recording is a (10 frame x 42 landmark x 2 coordinate) array of geometric values invariant to camera position, distance, and hand size. Feature extraction produces one of eight engineered representations from this normalised array; see `feature_extraction.py` and the [User Manual](USER_MANUAL.md) for the full specification.

## Feature variants

Eight engineered feature representations are generated from the same normalised corpus, grouped into four families. The same CNN architecture is used across all eight variants, with only the input-channel count differing, so any difference in cross-validation accuracy is attributable to the feature representation rather than to mismatched model capacity.

| Variant | Features | Family | Composition |
| --- | --- | --- | --- |
| v0 | 840 | Baseline | Wrist-relative (x, y) of all 42 landmarks across 10 frames |
| v1 | 9,090 | Max-info | All pairwise proximities (intra and inter-hand) plus 48 angles |
| v2 | 8,610 | Max-info | All pairwise proximities only |
| v3 | 1,140 | 12 core keypoints | 12-point distances plus 48 full-hand angles |
| v4 | 850 | 12 core keypoints | 12-point distances plus 12-point angles |
| v5 | 660 | 12 core keypoints | 12-point distances only |
| v6 | 850 | Inter-hand | Each hand's index-fingertip and wrist to all 21 opposite-hand landmarks, plus pinkie-pinkie |
| v7 | 360 | Inter-hand | 6x6 inter-hand distance matrix between the 12 linguistic keypoints |

The 12-keypoint subset consists of the wrist and five fingertips per hand, selected because BSL fingerspelling phonology is dominated by fingertip contact and configuration rather than by intermediate joint geometry (Liwicki and Everingham, 2009). Variants v4 and v6 are deliberately matched at 850 features despite encoding fundamentally different geometric priors so that any performance difference between them isolates the contribution of the prior rather than feature count.

## Repository layout

The repository is organised so that each pipeline stage lives in its own module and each major corpus is a sibling directory under `All_keypoint_data/`. Stage outputs follow the naming convention `<root>_s<n>_<stage>` so the order in which they were produced is visible from the directory listing alone.

```
.
├── README.md                          (this file)
├── USER_MANUAL.md                     (end-to-end operational manual)
├── formalProccessingPipeline.py       (top-level orchestrator: clean -> interpolate -> crop -> time-norm -> space-norm)
│
├── data_cleaner.py                    (Stage 1)
├── anomaly_detection.py               (anomaly signals used by data_cleaner.py)
├── keypoint_interpolator.py           (Stage 2)
├── temporal_cropping.py               (Stage 3)
├── temporal_normalisation.py          (Stage 4)
├── spatial_normalisation.py           (Stage 5)
├── feature_extraction.py              (Eight feature variants from normalised keypoints)
├── generate_corpuses.py               (Runs feature extraction across the full corpus)
│
├── 5-fold-CNN.py                      (Single-run trainer: one feature_set x seed x cv_mode combination)
├── Run_sweep.py                       (Sweep harness: drives 5-fold-CNN.py across the experimental matrix)
│
├── Validators/                        (Validation infrastructure)
│   └── keypoint_validator.py          (Sign-length detection, palm centres, momentum, anomaly signals)
│
├── All_keypoint_data/                 (Corpus storage at every pipeline stage)
│   ├── keypoints_V2/                  (raw MediaPipe output, V2 extraction pass)
│   ├── keypoints_V2_s1_cleaned/       (after Stage 1)
│   ├── keypoints_V2_s2_interpolated/  (after Stage 2)
│   ├── keypoints_V2_s3_cropped/       (after Stage 3)
│   ├── keypoints_V2_s4_time_norm/     (after Stage 4)
│   ├── keypoints_V2_s5_space_norm/    (after Stage 5, ready for feature extraction)
│   ├── Pipeline_without_cleaning/     (parallel V2 corpus for ablation analysis)
│   ├── Pipeline_without_cropping/     (parallel V2 corpus for ablation analysis)
│   ├── Pipeline_without_space_norm/   (parallel V2 corpus for ablation analysis)
│   └── V2_extension_cases/            (movement-defined extension signs: and, coffee, or, tea)
│
├── features/                          (Engineered feature corpora)
│   ├── feature_corpuses_V2/           (eight v0..v7 corpora from the full pipeline)
│   ├── feature_corpuses_V2_with_signer_ids/  (same plus signer_id for leave-one-signer-out)
│   └── feature_corpuses_V2_without_<stage>/  (parallel corpora for stage ablation)
│
├── Validation_testing/                (Synthetic anomaly corpora and detector evaluation)
├── Model_Training/                    (Trained model snapshots and per-data-version results)
├── results/                           (Aggregate results.jsonl from sweep runs)
├── BSL-survey/                        (Flask app for the data collection survey website)
└── Unexpected behaviour/              (Documented MediaPipe failure cases by category)
```

Several inputs and outputs are large enough that they are not committed to the repository. The full keypoint corpus, the eight engineered feature corpora, and the trained model snapshots are released as separate archive downloads. See the [User Manual](USER_MANUAL.md) for download locations and the expected disk layout.

## Quick start

These instructions reproduce the headline result (v5, 0.976 stratified accuracy) on a clean checkout. They assume Python 3.10 or newer.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The pipeline requires NumPy, SciPy, scikit-learn, PyTorch, Matplotlib, and PySide6 (for the visualiser only). MediaPipe is required for fresh keypoint extraction from video but is not needed to reproduce the headline result from the released keypoint corpus.

### 2. Obtain the data

Download `keypoints_V2.zip` and unzip it into `All_keypoint_data/`. The directory should contain one subfolder per letter class (A, B, E, I, J, N, O, P, S, T, U), each containing per-recording JSON files.

### 3. Run the processing pipeline

```bash
python formalProccessingPipeline.py
```

This invokes the five stages in order on the V2 keypoint corpus and writes the outputs of each stage to `All_keypoint_data/keypoints_V2_s<n>_<stage>/`. Total runtime is roughly twenty minutes on a mid-range laptop.

### 4. Generate the feature corpora

```bash
python generate_corpuses.py
```

This computes all eight feature representations from the Stage 5 corpus and writes them to `features/feature_corpuses_V2_with_signer_ids/`.

### 5. Train the model

```bash
python 5-fold-CNN.py --feature_set v5 --seed 42
```

Trains v5 under stratified five-fold cross-validation and writes a structured record to `results/results.jsonl`. A full sweep over all eight variants, seeds, and cross-validation modes is driven by `Run_sweep.py`; see the [User Manual](USER_MANUAL.md) for the complete experimental matrix.

## Reproducing the dissertation results

The full set of fifty-six experimental runs reported in the dissertation can be regenerated by:

```bash
python Run_sweep.py --mode all
```

This runs the main cross-feature-set matrix, the three-stage ablation analysis (no_mislabel, no_cropping, no_spatial), the augmentation ablation, the seed-stability analysis on v5 and v7, and the leave-one-signer-out validation pass on v0, v5, and v7. Run-level results are appended to `results/results.jsonl`; the harness is re-runnable and skips combinations already recorded.

Reproducibility is handled by a single `_set_seed` function fanning the seed across `random`, `numpy`, `torch`, and CUDA, with the seed recorded alongside every other hyperparameter in each result record. Re-execution at a given seed produces fold accuracies identical to floating-point precision.

## Hardware requirements

All figures in the dissertation were produced on a 13th Gen Intel Core i7-1355U laptop processor, a 15-watt mobile chip with no discrete GPU. Total five-fold training time spans 67 to 143 seconds across the eight variants; per-sample inference time stays between 0.06 and 0.21 milliseconds. The system is deliberately designed to be trainable and runnable on consumer-grade hardware, consistent with the project's accessibility commitments.

GPU acceleration is supported through PyTorch's CUDA backend if available, but is not required.

## Citation

If you use this work, please cite the accompanying dissertation:

> Strong, O. (2026). *Supervised Deep Learning Approaches to British Sign Language Recognition.* BSc Dissertation, School of Computing, University of Leeds.

## Licence

Released under the Apache Licence 2.0. The licence covers all original code in this repository. Third-party dependencies retain their respective licences; see Appendix B of the dissertation for the complete external materials inventory.

## Acknowledgements

Thank you to the University of Leeds British Sign Language Society, to the Deaf signers who contributed time and feedback to the project's framing, and to the thirty-six participants who recorded signs for the collection corpus. The project was supervised by Dillon Mayhew at the School of Computing, University of Leeds.