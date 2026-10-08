# MIP dataset

MIP is the OpenNeuro ds004306 multisensory imagination and perception dataset.
The loader reads EEGLAB `.set`/`.fdt` data, keeps all 124 EEG channels, extracts
event-aligned trials, and writes the same per-subject HDF files used by the
other TFDEEG datasets.

## Configuration

- Original sampling rate: 1024 Hz.
- Target sampling rate: 200 Hz.
- The original paper's baseline uses stratified 5-fold cross-validation;
  therefore MIP defaults to `--fold 5`.
- `A`: auditory imagination (0) vs perception (1), guitar and penguin only,
  using 2-second epochs.
- `V`: orthographic imagination (0) vs perception (1), using flower, guitar,
  and penguin trials with 3-second epochs.
- `D`: pictorial imagination (0) vs perception (1), using flower, guitar, and
  penguin trials with 3-second epochs.
- Common-average reference, 50/100/150 Hz notch filters, and a 2 Hz high-pass
filter follow the dataset paper. Resampling provides anti-alias low-pass
filtering. ICA is not repeated during HDF conversion.

The strict 18-class event table is used. Two non-standard `audio_a` event names
in `sub-03` are excluded, giving 99 imagination and 99 perception trials for A.
V and D include all three semantic categories and each contain 150 imagination
and 150 perception trials.

The loader discovers `sub-*` folders numerically and combines all sessions for
the same participant. With the full dataset, use `--subjects 12`.

## Run

```bash
python main.py --dataset MIP --data-path /home/hutao/缝好的/multisensory \
  --label-type A --train_method n_fold
```

```bash
python main.py --dataset MIP --data-path /home/hutao/缝好的/multisensory \
  --label-type V --train_method loso
```

The generated folders are `data_eeg_MIP_A`, `data_eeg_MIP_V`, and
`data_eeg_MIP_D`. Each contains `sub0.hdf` through `sub11.hdf` when all 12
participants are available.
For the bundled one-subject sample, add `--subjects 1`.
