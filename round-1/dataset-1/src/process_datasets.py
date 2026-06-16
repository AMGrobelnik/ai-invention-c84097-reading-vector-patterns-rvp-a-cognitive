#!/usr/bin/env python3
"""Process Quora Question Pairs (QQP) and PAWS datasets for duplicate detection experiment."""

from loguru import logger
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def load_and_inspect_dataset(file_path: str, dataset_name: str) -> pd.DataFrame:
    """Load and inspect a dataset from JSON file."""
    logger.info(f"Loading {dataset_name} from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    logger.info(f"{dataset_name} shape: {df.shape}")
    logger.info(f"{dataset_name} columns: {df.columns.tolist()}")
    logger.info(f"{dataset_name} first 3 rows:\n{df.head(3)}")
    
    # Check label distribution
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if label_col:
        logger.info(f"{dataset_name} label distribution:\n{df[label_col].value_counts()}")
    
    # Check for missing values
    logger.info(f"{dataset_name} missing values:\n{df.isnull().sum()}")
    
    return df

@logger.catch(reraise=True)
def standardize_qqp(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize QQP dataset to required format."""
    logger.info("Standardizing QQP dataset")
    
    # Rename columns to standard format
    column_mapping = {}
    for col in df.columns:
        if 'text1' in col.lower() or 'question1' in col.lower():
            column_mapping[col] = 'question1'
        elif 'text2' in col.lower() or 'question2' in col.lower():
            column_mapping[col] = 'question2'
        elif 'label' in col.lower() and 'text' not in col.lower():
            column_mapping[col] = 'is_duplicate'
    
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['question1', 'question2', 'is_duplicate']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")
    
    # Select only required columns
    df = df[required_cols]
    
    # Clean text - remove HTML tags and strip whitespace
    df['question1'] = df['question1'].astype(str).str.replace(r'<[^<>]+>', '', regex=True).str.strip()
    df['question2'] = df['question2'].astype(str).str.replace(r'<[^<>]+>', '', regex=True).str.strip()
    
    # Remove rows with empty questions
    initial_len = len(df)
    df = df[(df['question1'] != "") & (df['question2'] != "") & (df['question1'] != 'nan') & (df['question2'] != 'nan')]
    logger.info(f"Removed {initial_len - len(df)} rows with empty questions")
    
    # Ensure is_duplicate is integer
    df['is_duplicate'] = df['is_duplicate'].astype(int)
    
    # Validate labels
    assert set(df['is_duplicate'].unique()) <= {0, 1}, "Invalid labels found"
    
    logger.info(f"Final QQP dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['is_duplicate'].value_counts()}")
    
    return df

@logger.catch(reraise=True)
def standardize_paws(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize PAWS dataset to required format."""
    logger.info("Standardizing PAWS dataset")
    
    # Rename columns to standard format
    column_mapping = {}
    for col in df.columns:
        if 'sentence1' in col.lower():
            column_mapping[col] = 'question1'
        elif 'sentence2' in col.lower():
            column_mapping[col] = 'question2'
        elif 'label' in col.lower() and 'text' not in col.lower():
            column_mapping[col] = 'is_duplicate'
    
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['question1', 'question2', 'is_duplicate']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")
    
    # Select only required columns
    df = df[required_cols]
    
    # Clean text
    df['question1'] = df['question1'].astype(str).str.replace(r'<[^<>]+>', '', regex=True).str.strip()
    df['question2'] = df['question2'].astype(str).str.replace(r'<[^<>]+>', '', regex=True).str.strip()
    
    # Remove rows with empty questions
    initial_len = len(df)
    df = df[(df['question1'] != "") & (df['question2'] != "") & (df['question1'] != 'nan') & (df['question2'] != 'nan')]
    logger.info(f"Removed {initial_len - len(df)} rows with empty questions")
    
    # Ensure is_duplicate is integer
    df['is_duplicate'] = df['is_duplicate'].astype(int)
    
    # Validate labels
    assert set(df['is_duplicate'].unique()) <= {0, 1}, "Invalid labels found"
    
    logger.info(f"Final PAWS dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['is_duplicate'].value_counts()}")
    
    return df

@logger.catch(reraise=True)
def stratified_sample(df: pd.DataFrame, target_size: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Sample dataset with stratified labeling."""
    logger.info(f"Sampling {target_size} rows from dataset of size {len(df)}")
    
    if len(df) <= target_size:
        logger.info(f"Dataset already smaller than target size, using full dataset")
        return df
    
    # Stratified sampling
    _, sample_df = train_test_split(
        df,
        train_size=target_size,
        stratify=df['is_duplicate'],
        random_state=random_state
    )
    
    logger.info(f"Sampled dataset shape: {sample_df.shape}")
    logger.info(f"Sampled label distribution:\n{sample_df['is_duplicate'].value_counts()}")
    
    return sample_df

@logger.catch(reraise=True)
def create_output_splits(df: pd.DataFrame, output_prefix: str):
    """Create preview, mini, and full splits."""
    logger.info(f"Creating output splits for {output_prefix}")
    
    records = df.to_dict('records')
    
    # Preview: 5 rows
    preview_records = records[:5]
    preview_path = f"data_out_{output_prefix}_preview.json"
    with open(preview_path, 'w') as f:
        json.dump(preview_records, f, indent=2)
    logger.info(f"Saved preview to {preview_path}")
    
    # Mini: 100 rows
    mini_records = records[:100]
    mini_path = f"data_out_{output_prefix}_mini.json"
    with open(mini_path, 'w') as f:
        json.dump(mini_records, f, indent=2)
    logger.info(f"Saved mini to {mini_path}")
    
    # Full: all sampled records
    full_path = f"data_out_{output_prefix}_full.json"
    with open(full_path, 'w') as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved full to {full_path}")
    
    return preview_path, mini_path, full_path

@logger.catch(reraise=True)
def validate_output_files(prefix: str):
    """Validate output files have correct schema."""
    logger.info(f"Validating output files for {prefix}")
    
    required_fields = {'question1', 'question2', 'is_duplicate'}
    
    for split_name, filename in [
        ('preview', f"data_out_{prefix}_preview.json"),
        ('mini', f"data_out_{prefix}_mini.json"),
        ('full', f"data_out_{prefix}_full.json")
    ]:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Validating {split_name} split ({len(data)} rows)")
        
        for i, row in enumerate(data):
            row_fields = set(row.keys())
            assert row_fields == required_fields, f"{split_name}[{i}] missing fields: {required_fields - row_fields}"
            assert isinstance(row['is_duplicate'], int), f"{split_name}[{i}] is_duplicate not int"
            assert row['is_duplicate'] in [0, 1], f"{split_name}[{i}] invalid label"
            assert isinstance(row['question1'], str), f"{split_name}[{i}] question1 not str"
            assert isinstance(row['question2'], str), f"{split_name}[{i}] question2 not str"
        
        logger.info(f"✓ {split_name} validation passed")
    
    logger.info("All validation checks passed!")

@logger.catch(reraise=True)
def generate_metadata(df: pd.DataFrame, original_size: int, sampled_size: int, dataset_name: str, output_file: str):
    """Generate metadata for the dataset."""
    logger.info(f"Generating metadata for {dataset_name}")
    
    import datetime
    
    metadata = {
        "dataset_name": dataset_name,
        "source": "HuggingFace Hub",
        "download_date": datetime.datetime.now().isoformat(),
        "original_size": original_size,
        "sampled_size": sampled_size,
        "sampling_method": "stratified by is_duplicate label" if sampled_size < original_size else "full dataset (no sampling needed)",
        "random_seed": 42,
        "label_distribution": {
            "duplicate (1)": int(df[df['is_duplicate']==1].shape[0]),
            "non-duplicate (0)": int(df[df['is_duplicate']==0].shape[0])
        },
        "average_question_length": {
            "question1": float(df['question1'].str.split().str.len().mean()),
            "question2": float(df['question2'].str.split().str.len().mean())
        },
        "preprocessing": [
            "Renamed columns to standard format",
            "Removed HTML tags",
            "Stripped whitespace",
            "Removed rows with empty questions",
            "Converted is_duplicate to int"
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {output_file}")
    
    return metadata

@logger.catch(reraise=True)
def main():
    """Main processing function."""
    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("temp/datasets").mkdir(parents=True, exist_ok=True)
    
    # Process QQP dataset
    logger.info("="*60)
    logger.info("Processing QQP dataset")
    logger.info("="*60)
    
    qqp_df = load_and_inspect_dataset(
        "temp/datasets/full_SetFit_qqp_default_train.json",
        "QQP"
    )
    qqp_df = standardize_qqp(qqp_df)
    qqp_sampled = stratified_sample(qqp_df, target_size=10000)
    
    # Create output splits for QQP
    create_output_splits(qqp_sampled, "qqp")
    
    # Validate QQP outputs
    validate_output_files("qqp")
    
    # Generate QQP metadata
    generate_metadata(qqp_sampled, len(qqp_df), len(qqp_sampled), "Quora Question Pairs (QQP)", "data_out_qqp_metadata.json")
    
    # Process PAWS dataset
    logger.info("="*60)
    logger.info("Processing PAWS dataset")
    logger.info("="*60)
    
    paws_df = load_and_inspect_dataset(
        "temp/datasets/full_google-research-datasets_paws_labeled_final_train.json",
        "PAWS"
    )
    paws_df = standardize_paws(paws_df)
    paws_sampled = stratified_sample(paws_df, target_size=5000)  # PAWS is smaller, sample 5K
    
    # Create output splits for PAWS
    create_output_splits(paws_sampled, "paws")
    
    # Validate PAWS outputs
    validate_output_files("paws")
    
    # Generate PAWS metadata
    generate_metadata(paws_sampled, len(paws_df), len(paws_sampled), "PAWS (Paraphrase Adversaries from Word Scrambling)", "data_out_paws_metadata.json")
    
    logger.info("="*60)
    logger.info("Dataset processing complete!")
    logger.info("="*60)
    logger.info("Output files created:")
    logger.info("  - data_out_qqp_preview.json")
    logger.info("  - data_out_qqp_mini.json")
    logger.info("  - data_out_qqp_full.json")
    logger.info("  - data_out_qqp_metadata.json")
    logger.info("  - data_out_paws_preview.json")
    logger.info("  - data_out_paws_mini.json")
    logger.info("  - data_out_paws_full.json")
    logger.info("  - data_out_paws_metadata.json")

if __name__ == "__main__":
    main()
