# ERA Expert Ensemble

This extension keeps the original AT-DGNN architecture and `forward()` contract.
It adds subject-specific experts, reliability-gated OOF distillation,
rest-anchored relational ERA geometry, and adaptive probability fusion around
the existing five-class model.

## Data

The expert pipeline reads the existing HDF files:

```text
data_eeg_EMO_Axis/
  sub0.hdf
  ...
  sub31.hdf
```

Old HDF files remain supported. When trial metadata is absent, it is derived
from the original `(trial, segment, feature, channel, time)` layout. Rebuilding
data with the updated `prepare_data.py` stores `sample_id`, `trial_id`,
`segment_id`, and `subject_id` explicitly.

## Install

Use the same CUDA/PyTorch environment as the original model, then install the
packages in `requirements.txt`. AT-DGNN specifically requires `mamba_ssm` and
`einops`; preprocessing additionally requires `mne` and `pyedflib`.

## Train Individual Experts

Run from the project root:

```powershell
python main.py --mode individual `
  --data-dir data_eeg_EMO_Axis `
  --checkpoint-dir checkpoints `
  --oof-dir oof
```

To train a subset while testing the pipeline:

```powershell
python main.py --mode individual --subject-list 0,1
```

For every subject this performs trial-grouped OOF training, saves OOF logits
and probabilities, then trains one final expert using the complete subject.

Outputs:

```text
oof/subject0_oof_logits.npy
oof/subject0_oof_probability.npy
oof/subject0_oof_metadata.npz
subject_geometry/subject_0_geometry.npy
checkpoints/Expert_0_final.pth
...
```

## Build Shared ERA Geometry

Individual training writes each subject geometry automatically. It can also be
rebuilt explicitly together with the reliability-weighted shared template:

```powershell
python main.py --mode geometry `
  --oof-dir oof `
  --geometry-dir subject_geometry
```

Output:

```text
subject_geometry/subject_0_geometry.npy
...
subject_geometry/shared_era_geometry.npy
```

Each subject file contains calibrated OOF class centers, Euclidean and JS
relations, and a rest-anchored normalized distance matrix. The shared template
weights subjects using OOF Macro-F1, NLL, and ECE.

## Train Shared Expert

Run Individual Expert training first, then:

```powershell
python main.py --mode global `
  --data-dir data_eeg_EMO_Axis `
  --checkpoint-dir checkpoints `
  --oof-dir oof `
  --geometry-dir subject_geometry `
  --global-objective full
```

Available ablations are:

```text
ce              = CE
kd              = CE + probability KD
reliability_kd  = CE + reliability/confidence/entropy-gated KD
full            = CE + gated KD + rest-anchored geometry loss
```

Run all four ablations sequentially with `--run-global-ablations`. This writes
`Global_ce_final.pth`, `Global_kd_final.pth`,
`Global_reliability_kd_final.pth`, and `Global_full_final.pth`; the full model
is also saved as `Global_final.pth`.

Evaluate the four Global checkpoints with the same data protocol:

```powershell
python -m train.evaluate_global_ablations `
  --model-dir checkpoints `
  --data-dir data_eeg_EMO_Axis
```

This writes `ERA_results/ablations/global_ablation_results.json` with
classification, calibration, and ERA geometry metrics for every objective.

## Inference

```powershell
python inference.py `
  --model-dir checkpoints `
  --global-checkpoint Global_final.pth `
  --dataset RESB `
  --data-dir data_eeg_EMO_Axis `
  --results-dir ERA_results
```

For ablation inference, set `--global-checkpoint` to a file such as
`Global_ce_final.pth` or `Global_reliability_kd_final.pth`. Add
`--unknown-subject --top-k 3` for the Global + top-k similarity path.

The inference process:

1. applies each checkpoint's own training normalization;
2. calibrates probabilities using its saved temperature;
3. extracts a feature from the Global Expert through a hook on the final layer;
4. computes similarity to subject prototypes in the shared feature space;
5. fuses calibrated probabilities using reliability, similarity, and entropy;
6. uses the matched Individual Expert for a known RESB subject;
7. uses only top-k similar Experts for an unknown subject and falls back to
   Global-only when similarity is too low;
8. computes ERA only from the final fused probability.

Outputs:

```text
ERA_results/subject_result.json
ERA_results/sample_result.npy
ERA_results/ERA_center.json
ERA_results/ERA_scatter.png
```

Results include ACC, Macro-F1, Kappa, NLL, ECE, Brier score, five ERA centers,
the ERA center-distance matrix, and within-class variance.

## Important Protocol Notes

- Do not shuffle samples independently for different experts during inference.
- Do not normalize test subjects with their labels or with a different policy.
- Do not average individual ERA coordinates; fuse probabilities first.
- OOF folds use trial groups. Segment-level random K-fold is not an OOF teacher.
- External datasets must use the same channel order, graph layout, sampling
  rate, window length, and preprocessing contract recorded in checkpoints.
