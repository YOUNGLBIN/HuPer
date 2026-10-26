# HuPer

Code supplement for the anonymous submission:

**HuPer: Aligning Brain with Human-Perception-Inspired Visual Representations for fMRI-to-Text Decoding**  

HuPer improves fMRI-to-text decoding by constructing visual alignment targets
that better reflect human perception. It combines context-aware MQT
representations with subject-specific, token-wise fusion over multiple Gaussian
blur levels, and constrains the fusion weights with a low-blur prior.

```text
NSD images -> multi-level Gaussian blur -> MQT features
                                           |
NSD fMRI  -> cross-subject fMRI encoder ----+-> MQT-LLaVA -> captions
```

This repository provides the formal 256-token experiment used for the paper's
main result, including preprocessing, training, caption generation, and COCO
caption evaluation. 

## Main result

Results are averaged across NSD subjects 1, 2, 5, and 7 and reported on the
percentage scale used in the paper.

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDEr | SPICE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64.29 | 45.63 | 31.33 | 21.27 | 20.80 | 47.50 | 71.74 | 13.85 |

The complete generated captions used for this result are included under
`caption_result/`, with 982 captions for each of subjects 1, 2, 5, and 7. After
installing the environment and preparing the evaluation assets, recompute all
eight metrics and their four-subject average with:

```bash
huper evaluate --config configs/paper.yaml --input caption_result
```

The recomputed reports are written to `outputs/evaluation/last/`.

## Installation

The paper experiment was run on a single NVIDIA GeForce RTX 4090. We recommend
a CUDA GPU with 24 GiB memory and at least 230 GiB of free disk space for the
Gaussian feature cache.

Extract the code archive and create the environment from the HuPer repository
root:

```bash
conda env create -f environment.yml
conda activate huper
```

The environment installs HuPer in editable mode so that the repository-local
configuration and reference-caption files remain available. Keep the extracted
repository intact. Standalone wheel installation and `pip install .` are not
supported by this supplementary package. After installation, the `huper`
commands may be run from any working directory.

The environment uses Python 3.10, PyTorch 2.4/CUDA 12.1, and Java 11. Java is
required for METEOR and SPICE evaluation.

## Data and pretrained models

Create a local configuration without modifying the committed paper settings:

```bash
cp configs/paper.yaml configs/local.yaml
```

Set `paths.nsd_root` in `configs/local.yaml` to the NSD directory containing
`webdataset_avg_split/`:

```yaml
paths:
  nsd_root: /path/to/nsd
```

Expected layout:

```text
nsd/
└── webdataset_avg_split/
    ├── train/train_subj0{1,2,5,7}_{0..17}.tar
    ├── val/val_subj0{1,2,5,7}_0.tar
    ├── test/test_subj0{1,2,5,7}_{0..1}.tar
    └── metadata_subj0{1,2,5,7}.json
```

Download the pinned MQT-LLaVA, CLIP-336, and evaluation assets, then check the
environment and data:

```bash
huper download-models --config configs/local.yaml
huper check --config configs/local.yaml
```

## Reproduction

Run each stage separately:

```bash
# 1. Extract the multi-level Gaussian MQT feature cache.
huper preprocess --config configs/local.yaml

# 2. Train HuPer.
huper train --config configs/local.yaml --gpu 0

# 3. Generate captions for subjects 1, 2, 5, and 7.
huper infer --config configs/local.yaml --gpu 0

# 4. Compute BLEU, METEOR, ROUGE, CIDEr, and SPICE.
huper evaluate --config configs/local.yaml
```

Or run the full pipeline:

```bash
huper run-all --config configs/local.yaml --gpu 0
```

On systems where Accelerate reports an RTX 40-series P2P/IB error, run training
with:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  huper train --config configs/local.yaml --gpu 0
```

The paper configuration uses subjects `1, 2, 5, 7`, 12 Gaussian kernels, 256
MQT tokens, batch size 32, 160 epochs, seed 42, MSE alignment loss, and KL
weight 0.1.

## Checkpoints and inference

Training writes checkpoints to `outputs/train/huper_paper_seed42/`. By default,
inference uses `last.pth`. To use another compatible checkpoint:

```bash
huper infer \
  --config configs/local.yaml \
  --checkpoint /path/to/last.pth \
  --gpu 0
```

The checkpoint's original `config.json` must be located beside it. Interrupted
caption generation can be continued with:

```bash
huper infer --config configs/local.yaml --resume --gpu 0
```

## Outputs

```text
outputs/
├── cache/
│   └── gaussian_feature_cache.pt
├── train/huper_paper_seed42/
│   ├── best.pth
│   └── last.pth
├── inference/last/
│   └── sub{01,02,05,07}_dim1024/fmricap.json
└── evaluation/last/
    ├── summary.json
    └── summary.csv
```

The complete Gaussian cache is approximately 214 GiB and has feature shape
`[N, 12, 256, 1024]`. Evaluation uses the bundled 982-entry BrainHub reference
caption set.

## Tests

```bash
pytest
ruff check huper tests
```

## Citation

If this project is useful in your research, please cite:

> *HuPer: Aligning Brain with Human-Perception-Inspired Visual Representations
> for fMRI-to-Text Decoding.*

The manuscript is currently anonymized. Complete citation metadata will be
added after publication.

## Acknowledgements

HuPer builds on MQT-LLaVA, UMBRAE, BrainHub, NSD, and COCO caption evaluation.
Third-party components and their licenses are documented in
`THIRD_PARTY_NOTICES.md` and `third_party/licenses/`.

The original HuPer code in this archive is provided under the anonymous-review
notice in `LICENSE`. Model checkpoints, NSD data, COCO images, and Stanford
CoreNLP JAR files are not redistributed in this supplement.
