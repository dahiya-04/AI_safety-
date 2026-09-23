"""Small TruthfulQA generation evaluation."""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ..generation import generate_text
from ..model_loader import load_model


def load_questions(num_samples: int = 30, seed: int = 42) -> pd.DataFrame:
    """Load a reproducible sample from TruthfulQA."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    sample = dataset["validation"].shuffle(seed=seed).select(range(num_samples))
    return pd.DataFrame(sample)


def evaluate(
    model_name: str,
    num_samples: int,
    output_path: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate answers for a TruthfulQA sample and save them."""
    questions = load_questions(num_samples=num_samples, seed=seed)
    bundle = load_model(model_name)

    rows = []
    for idx, question in enumerate(questions["question"]):
        answer = generate_text(
            bundle.model,
            bundle.tokenizer,
            question,
            max_new_tokens=200,
        )
        rows.append(
            {
                "id": idx,
                "question": question,
                "answer": answer,
            }
        )

    results = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument(
        "--output",
        default="results/truthfulqa_results.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = evaluate(
        model_name=args.model,
        num_samples=args.num_samples,
        output_path=args.output,
        seed=args.seed,
    )
    print(results.head())


if __name__ == "__main__":
    main()
