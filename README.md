# RS-Flow-VQA

Conditional Flow Matching and FreeFlow distillation for bridging frozen
Scale-MAE remote-sensing features into the embedding space of frozen
Qwen2.5-3B-Instruct.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/rs-flow-vqa.git
cd rs-flow-vqa
uv sync
uv run pytest -q
```

The real GPU profile additionally needs:

```bash
uv sync --extra gpu --extra notebook
```

## Data

The small datasets shipped in `data/` are synthetic smoke fixtures. They are
not research data, and non-smoke runs deliberately reject them.

Place the official RSICD release at:

```text
data/RSICD/
├── dataset_rsicd.json
└── RSICD_images/
```

A complete RSICD installation contains 10,921 images and 54,605 captions.

Place an official RSVQA-LR release at:

```text
data/RSVQA_LR/
├── LR_split_test_questions.json
├── LR_split_test_answers.json
├── LR_split_test_images.json
└── Images_LR/
```

The legacy combined `lr_questions_answers.json` format is also supported. A
complete RSVQA-LR release contains 772 images and tens of thousands of
question-answer pairs.

## Pipeline

Run a CPU smoke test first:

```bash
uv run rs-flow-vqa cache-features --smoke
uv run rs-flow-vqa train-teacher --smoke
uv run rs-flow-vqa distill-freeflow --smoke
uv run rs-flow-vqa evaluate-caption --smoke
uv run rs-flow-vqa evaluate-rsvqa --smoke
```

Then run the real T4 profile after installing the official datasets:

```bash
uv run rs-flow-vqa cache-features --config configs/t4.yaml
uv run rs-flow-vqa train-teacher --config configs/t4.yaml
uv run rs-flow-vqa distill-freeflow --config configs/t4.yaml
uv run rs-flow-vqa evaluate-caption --config configs/t4.yaml
uv run rs-flow-vqa evaluate-rsvqa --config configs/t4.yaml
```

Feature caching uses training-split statistics only to standardize Scale-MAE
conditions and Qwen target embeddings. Teacher training mixes uniform and
high-noise-biased flow times and includes an image-caption alignment
regularizer. Validation reports the loss increase caused by shuffling image
conditions. FreeFlow distillation is blocked if that condition gap is below
the configured minimum.

The Colab notebook is
`notebooks/RS_Flow_VQA_Pipeline.ipynb`. It assumes the working directory is
the repository root and performs the same real-data checks before downloading
models or training.

Smoke-mode metrics validate software integrity only and are never valid
research results.
