# Third-party notices

The original HuPer code in this anonymous review supplement is covered by the
review-only notice in `LICENSE`. Third-party materials are not covered by that
notice and remain subject to the licenses and terms identified below.

## Included or adapted materials

### 1. MQT-LLaVA / LLaVA runtime

- Source: https://github.com/gordonhu608/MQT-LLaVA
- Included under `llava/`.
- License: Apache License 2.0, copied to
  `third_party/licenses/MQT-LLaVA-Apache-2.0.txt`.
- The vendored files were reduced and integrated for the model-loading,
  vision-encoder, query-abstractor, and caption-generation runtime used by
  HuPer. Modified files carry a project-local modification notice; applicable
  upstream copyright and attribution notices are retained.

The MQT-LLaVA checkpoint itself is not included. The configured
`gordonhu/MQT-LLaVA-7b` checkpoint is based on LLaMA/Vicuna and its model card
states that Llama 2 is governed by the Llama 2 Community License:
https://huggingface.co/gordonhu/MQT-LLaVA-7b/blob/main/README.md

### 2. UMBRAE BrainX implementation

- Source: https://github.com/weihaox/UMBRAE
- Author notice retained from the adapted source: Weihao Xia.
- Adapted in `huper/core/brain_encoder.py`.
- License: Apache License 2.0, copied to
  `third_party/licenses/UMBRAE-Apache-2.0.txt`.
- The HuPer file carries a notice identifying that it was changed.

### 3. flamingo-pytorch Perceiver Resampler

- Source: https://github.com/lucidrains/flamingo-pytorch
- Copyright (c) 2022 Phil Wang.
- Adapted in `huper/core/perceiver_resampler.py`, via the implementation used
  by UMBRAE.
- License: MIT, copied to
  `third_party/licenses/flamingo-pytorch-MIT.txt`.
- The HuPer file carries a notice identifying its source and that it was
  changed.

### 4. BrainHub caption references and evaluation protocol

- Source: https://github.com/weihaox/BrainHub
- The processed 982-entry reference file is included as
  `data/brainhub/fmri_cococap.json`.
- BrainHub license: MIT, copied to
  `third_party/licenses/BrainHub-MIT.txt`.
- Caption metrics are computed with the separately installed
  `pycocoevalcap` package; that package is not redistributed in this archive.

### 5. COCO caption annotations contained in the BrainHub references

- Creator: COCO Consortium.
- Source and terms: https://cocodataset.org/#termsofuse
- License: Creative Commons Attribution 4.0 International (CC BY 4.0):
  https://creativecommons.org/licenses/by/4.0/legalcode
- Required attribution and change information are recorded in
  `third_party/licenses/COCO-CC-BY-4.0-NOTICE.txt`.
- HuPer includes the 982-image, 4,913-caption processed subset distributed by
  BrainHub under a project-local path. HuPer does not claim authorship of the
  captions and does not intentionally alter their text.

## Referenced or downloaded materials not redistributed in this archive

### Stanford CoreNLP 3.6.0

HuPer does not redistribute CoreNLP. `huper download-models` downloads the
pinned 2015-12-09 release from Stanford's server into the ignored
`models/spice/` asset directory for SPICE evaluation. CoreNLP is licensed under
GNU GPL version 3 or later. Do not add its JAR files to this review archive.

Source and license information:
https://nlp.stanford.edu/software/index.shtml

### Natural Scenes Dataset (NSD)

No NSD fMRI data, images, or WebDataset shards are included. Users must obtain
NSD through its official access process and comply with the applicable data
access agreement. Do not add NSD data or a representative NSD subset to this
archive.

Source and access information: https://www.naturalscenesdataset.org/

### CLIP-336 and other model assets

No CLIP model weights, Vicuna/Llama weights, MQT-LLaVA weights, or other model
checkpoints are included. The download command retrieves pinned upstream
revisions, and users remain responsible for the corresponding upstream terms.

