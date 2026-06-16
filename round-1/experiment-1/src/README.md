# Near-Duplicate Short Text Detection Experiment

## Experiment Summary

This experiment implements and evaluates two methods for detecting near-duplicate short text snippets:

1. **Baseline Method**: Character n-gram similarity (n=3) with Jaccard similarity threshold
2. **Proposed Method**: MinHash LSH (Locality-Sensitive Hashing) with word k-shingles (k=2)

## Dataset

- **Type**: Synthetic short text pairs with controlled modifications
- **Size**: 500 pairs per trial (70% near-duplicates)
- **Modifications**: Word substitution, deletion, insertion, and reordering
- **Trials**: 5 trials with different random seeds for statistical significance

## Results

### Baseline (Character N-gram Similarity)
- Precision: 1.000 ± 0.000
- Recall: 1.000 ± 0.000
- F1 Score: 1.000 ± 0.000
- Accuracy: 1.000 ± 0.000

### Proposed (MinHash LSH)
- Precision: 1.000 ± 0.000
- Recall: 0.885 ± 0.013
- F1 Score: 0.939 ± 0.007
- Accuracy: 0.921 ± 0.008

### Comparison
- ΔPrecision: +0.000 (same)
- ΔRecall: -0.115 (lower)
- ΔF1 Score: -0.061 (lower)

## Key Findings

1. **Baseline performs perfectly** on this synthetic dataset - character n-gram similarity with Jaccard distance is extremely effective for detecting near-duplicates with small modifications

2. **MinHash LSH has high precision (1.0)** but lower recall (~88.5%), meaning it never makes false positive errors but misses some true duplicates

3. **Threshold tuning matters**: Optimal MinHash threshold was found to be 0.1 (lower than the default 0.5)

4. **MinHash LSH is more scalable** for large datasets due to sub-linear query time with LSH indexing, while pairwise n-gram comparison is O(n²)

## Files

- `method.py`: Main experiment implementation
- `method_out.json`: Full results in exp_gen_sol_out schema format (100 examples)
- `full_method_out.json`: Complete output file
- `mini_method_out.json`: First 3 examples (for quick testing)
- `preview_method_out.json`: First 3 examples with truncated strings (for inspection)
- `test_dataset.json`: Generated test dataset
- `logs/run.log`: Detailed execution log

## Reproducibility

To reproduce this experiment:
```bash
cd /home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run_Z4TPmJBKTsQm/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
source .venv/bin/activate
python method.py
```

All random seeds are controlled (seed=42 + trial_number) for reproducibility.

## Schema Validation

Output validated against `exp_gen_sol_out.json` schema:
```
Format: exp_gen_sol_out
Validation PASSED
```
