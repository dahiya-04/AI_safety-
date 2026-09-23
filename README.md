# AAIGM: Adversarial Prompting and Language Model Evaluation

A research-oriented project for studying **language-model behavior under adversarial prompting**, model generation, benchmark evaluation, and token-level likelihood analysis.

The original work was developed as a notebook-based experiment and has been reorganized here into reusable Python modules.

## Project goals

This repository focuses on:

- Loading and running small Hugging Face causal language models.
- Comparing standard text generation across prompts.
- Evaluating model behavior on benchmark datasets such as TruthfulQA and JailbreakBench.
- Measuring answer-token log probabilities.
- Studying the mechanics of gradient-based adversarial suffix optimization at a conceptual/research level.
- Keeping experiments reproducible and separating model loading, generation, evaluation, and analysis.

> **Safety note:** The public version intentionally removes operational harmful prompts and does not include instructions for generating harmful content or bypassing model safeguards. The adversarial-prompting code is presented as a research scaffold for controlled/synthetic experiments.

## Repository structure

```text
AAIGM/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── generation.py
│   ├── logprob.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── truthfulqa.py
│   │   └── jailbreakbench.py
│   └── experiments/
│       ├── __init__.py
│       └── adversarial_suffix.py
├── notebooks/
│   └── AAIGM_cleaned.ipynb
└── results/
    └── .gitkeep
```

## Environment

Recommended:

- Python 3.10 or 3.11
- PyTorch
- Transformers
- Datasets
- Pandas
- Accelerate

A GPU is recommended for running the larger language-model experiments.

## Installation

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AAIGM

python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

Run a simple generation experiment:

```bash
python -m src.generation
```

Run the TruthfulQA sample evaluation:

```bash
python -m src.evaluation.truthfulqa
```

Run the token-level likelihood example:

```bash
python -m src.logprob
```

## Model

The experiments were originally developed around:

```text
Qwen/Qwen3-1.7B
```

The model can be changed from the command line or by editing the model configuration in the relevant module.

Models hosted on Hugging Face may have their own license and access requirements. Check the model card before redistribution or commercial use.

## Experiments

### 1. Baseline generation

`src/generation.py` contains reusable functions for:

- Loading a causal language model.
- Applying a chat template when supported.
- Generating deterministic responses.
- Decoding only newly generated tokens.

### 2. TruthfulQA

`src/evaluation/truthfulqa.py`:

1. Downloads the TruthfulQA generation split through Hugging Face Datasets.
2. Selects a reproducible sample.
3. Generates model responses.
4. Saves the responses as CSV.

Example:

```bash
python -m src.evaluation.truthfulqa --num-samples 30
```

### 3. JailbreakBench

`src/evaluation/jailbreakbench.py` provides a benchmark-loading and response-generation scaffold.

The public implementation deliberately avoids embedding harmful target prompts or instructions. If you are conducting authorized safety research, supply your approved benchmark data through a local/private path and follow the benchmark's license and usage rules.

### 4. Token-level log probabilities

`src/logprob.py` demonstrates how to calculate the average log probability assigned to the tokens of a candidate answer.

For a question and candidate options, the model can assign a score to each option. This can be useful for studying model preferences without relying only on generated text.

### 5. Adversarial suffix research

`src/experiments/adversarial_suffix.py` contains a **controlled research scaffold** showing the mathematical components used in gradient-based suffix optimization:

- Freeze the language model.
- Represent a suffix using token embeddings.
- Compute a target-token loss.
- Differentiate the loss with respect to suffix embeddings.
- Rank candidate token directions.

The repository does not provide harmful targets, jailbreak strings, or a ready-to-run harmful optimization recipe.

## Reproducibility

For repeatable experiments:

- Set a random seed.
- Keep model and Transformers versions fixed.
- Record GPU/CPU information.
- Save generated results under `results/`.
- Avoid committing downloaded model weights or benchmark datasets.

Example:

```text
Model: Qwen/Qwen3-1.7B
Dataset: TruthfulQA generation
Sample size: 30
Seed: 42
Decoding: greedy
```

## Outputs

Generated CSV files should be stored under:

```text
results/
```

Large datasets, model weights, caches, and generated artifacts should **not** be committed to GitHub.

## Research questions

Possible extensions include:

- How does model size affect resistance to adversarial prompts?
- How does reasoning mode change response behavior?
- Can token-level likelihood identify differences between candidate answers?
- How do refusal rates vary across benchmark categories?
- How does optimization loss correlate with generated-response behavior?
- Can controlled adversarial-prompt experiments reveal weaknesses without producing harmful content?

## Citation / acknowledgement

This project uses open-source tooling and public benchmark datasets from the Hugging Face ecosystem. Please cite the original model, benchmark, and relevant research papers when publishing results.

## Disclaimer

This repository is intended for **AI safety, robustness, and language-model evaluation research**. Do not use it to bypass safety controls, generate harmful instructions, or target real systems without authorization.
