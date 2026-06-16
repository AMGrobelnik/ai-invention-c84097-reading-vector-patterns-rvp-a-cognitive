#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///

"""Convert processed QQP and PAWS datasets to exp_sel_data_out.json schema."""

from pathlib import Path
import json
import pandas as pd
import numpy as np

def load_processed_dataset(json_path: str) -> list[dict]:
    """Load processed dataset from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def convert_to_experiment_format(qqp_data: list[dict]) -> dict:
    """Convert QQP dataset to experiment schema format."""
    
    datasets = []
    
    # Process QQP dataset only
    qqp_examples = []
    for i, row in enumerate(qqp_data):
        example = {
            "input": json.dumps({
                "question1": row["question1"],
                "question2": row["question2"]
            }),
            "output": str(row["is_duplicate"]),
            "metadata_fold": i % 5,  # 5-fold cross-validation
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i
        }
        qqp_examples.append(example)
    
    datasets.append({
        "dataset": "quora_question_pairs",
        "examples": qqp_examples
    })
    
    return {"datasets": datasets}

def main():
    # Load processed datasets
    qqp_path = "data_out_qqp_full.json"
    
    print(f"Loading QQP dataset from {qqp_path}")
    qqp_data = load_processed_dataset(qqp_path)
    print(f"Loaded {len(qqp_data)} QQP examples")
    
    # Convert to experiment format (QQP only)
    print("Converting to experiment format...")
    experiment_data = convert_to_experiment_format(qqp_data)
    
    # Save to full_data_out.json
    output_path = "full_data_out.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"Saved to {output_path}")
    print(f"Total datasets: {len(experiment_data['datasets'])}")
    print(f"  - QQP: {len(experiment_data['datasets'][0]['examples'])} examples")

if __name__ == "__main__":
    main()
