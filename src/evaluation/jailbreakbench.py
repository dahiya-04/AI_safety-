"""JailbreakBench loading and controlled evaluation scaffold.

The public repository intentionally does not embed harmful benchmark goals.
For authorized safety research, provide benchmark inputs from a permitted
local/private source and use the same generation pipeline.
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ..generation import generate_text
from ..model_loader import load_model


def load_jailbreakbench() -> pd.DataFrame:
    """Load the JailbreakBench behavior dataset."""
    dataset = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        "behaviors",
    )
    return dataset["harmful"].to_pandas()


def generate_controlled_responses(
    prompts: list[str],
    model_name: str = "Qwen/Qwen3-1.7B",
) -> pd.DataFrame:
    """Generate responses for caller-supplied, authorized test prompts."""
    bundle = load_model(model_name)

    rows = []
    for idx, prompt in enumerate(prompts):
        answer = generate_text(
            bundle.model,
            bundle.tokenizer,
            prompt,
            max_new_tokens=200,
        )
        rows.append(
            {"id": idx, "prompt": prompt, "answer": answer}
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download metadata without running generation.",
    )
    parser.add_argument(
        "--output",
        default="results/jailbreakbench_metadata.csv",
    )
    args = parser.parse_args()

    df = load_jailbreakbench()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Only save benchmark metadata, not model outputs.
    df.drop(columns=["Goal"], errors="ignore").to_csv(args.output, index=False)

    print(f"Loaded {len(df)} benchmark records.")
    if args.download_only:
        return


if __name__ == "__main__":
    main()
