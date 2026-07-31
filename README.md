# RS-Flow-VQA

Conditional Flow Matching and FreeFlow distillation for bridging frozen
Scale-MAE remote-sensing features into frozen Qwen2.5-1.5B-Instruct.

The v3 pipeline avoids regressing raw LLM token embeddings. It learns a
language-compatible compact latent and preserves a 4×4 Scale-MAE patch grid:

```text
Scale-MAE [16,1024] -> visual resampler -> latent [8,256]
                                             |
                              CFM teacher -> FreeFlow student
                                             |
                           prompt decoder -> Qwen prefix [16,1536]
```

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
uv run rs-flow-vqa train-prompt-autoencoder --smoke
uv run rs-flow-vqa train-visual-alignment --smoke
uv run rs-flow-vqa train-teacher --smoke
uv run rs-flow-vqa distill-freeflow --smoke
uv run rs-flow-vqa evaluate-caption --smoke
uv run rs-flow-vqa evaluate-rsvqa --smoke
```

Then run the real T4 profile after installing the official datasets:

```bash
uv run rs-flow-vqa cache-features --config configs/t4.yaml
uv run rs-flow-vqa train-prompt-autoencoder --config configs/t4.yaml
uv run rs-flow-vqa train-visual-alignment --config configs/t4.yaml
uv run rs-flow-vqa train-teacher --config configs/t4.yaml
uv run rs-flow-vqa distill-freeflow --config configs/t4.yaml
uv run rs-flow-vqa evaluate-caption --config configs/t4.yaml
uv run rs-flow-vqa evaluate-rsvqa --config configs/t4.yaml
```

Feature caching stores FP16 spatial features and raw Qwen token IDs. The
prompt-autoencoder stage learns the compact language space; visual alignment
then uses a latent warm start followed by caption supervision through frozen
Qwen. Later flow stages train small latent models from cached tensors. Each
stage has a matched-vs-shuffled validation gate and checkpoints independently.

FreeFlow is *target-free conditional distillation*: it samples flow states
only from the Gaussian prior and never reads cached caption targets, but it
does sample real cached image conditions. This is required for conditional
generation.

The Colab notebook is
`notebooks/RS_Flow_VQA_Pipeline.ipynb`. It assumes the working directory is
the repository root and performs the same real-data checks before downloading
models or training.

The default real profile is designed for one 16 GB T4: Qwen is loaded in
4-bit NF4 with micro-batch 1, Scale-MAE and Qwen are never required
simultaneously, and RSVQA evaluates a reproducible 10% subset unless
`RUN_FULL_EVAL=True` is selected in the notebook. Expect roughly 4–7 hours
for a first full run; Drive-backed outputs make it safe to continue across
Colab sessions.

Smoke-mode metrics validate software integrity only and are never valid
research results.
