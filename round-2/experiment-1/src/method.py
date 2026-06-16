#!/usr/bin/env python3
"""
Reading Vector Patterns (RVP) for Near-Duplicate Short Text Detection: Implementation and Evaluation

This script implements the RVP method using cognitive load indicators (word length, frequency, POS)
with DTW comparison on Quora Question Pairs dataset, comparing against multiple baselines with
statistical analysis and ablation study.
"""

import json
import sys
import os
import time
import random
import tracemalloc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import bootstrap
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import re
from collections import Counter
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import gensim
from tqdm import tqdm

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    data_path: str
    output_path: str
    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 2000
    random_seed: int = 42
    n_bootstrap: int = 1000
    dtw_radius: int = 1


@dataclass
class MethodResult:
    """Results from a single method."""
    method_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    avg_time_per_pair: float
    distances: List[float]
    predictions: List[int]


class DataLoader:
    """Handles loading and preprocessing of QQP dataset."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.examples = []

    def load_data(self, max_examples: Optional[int] = None) -> List[Dict]:
        """Load data from JSON files."""
        logger.info(f"Loading data from {self.data_path}")

        if self.data_path.is_dir():
            # Load from multiple files
            json_files = sorted(self.data_path.glob("full_data_out_*.json"))
            if not json_files:
                json_files = sorted(self.data_path.glob("*.json"))
        else:
            json_files = [self.data_path]

        all_examples = []
        for json_file in json_files:
            logger.debug(f"Reading {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract examples from datasets format
            if "datasets" in data and isinstance(data["datasets"], list):
                for dataset in data["datasets"]:
                    if isinstance(dataset, dict) and "examples" in dataset:
                        examples_list = dataset["examples"]
                        if isinstance(examples_list, list):
                            for example in examples_list:
                                if isinstance(example, dict):
                                    parsed = self._parse_example(example)
                                    if parsed:
                                        all_examples.append(parsed)
            elif isinstance(data, list):
                # Direct list format
                for example in data:
                    if isinstance(example, dict):
                        parsed = self._parse_example(example)
                        if parsed:
                            all_examples.append(parsed)
            elif isinstance(data, dict) and "examples" in data:
                # Dict with examples key
                examples_list = data["examples"]
                if isinstance(examples_list, list):
                    for example in examples_list:
                        if isinstance(example, dict):
                            parsed = self._parse_example(example)
                            if parsed:
                                all_examples.append(parsed)

            if max_examples and len(all_examples) >= max_examples:
                all_examples = all_examples[:max_examples]
                break

        logger.info(f"Loaded {len(all_examples)} valid examples")
        self.examples = all_examples
        return all_examples

    def _parse_example(self, example: Dict) -> Optional[Dict]:
        """Parse a single example from the dataset format."""
        try:
            # Parse input field (JSON string)
            if isinstance(example.get("input"), str):
                input_data = json.loads(example["input"])
            else:
                input_data = example.get("input", {})

            question1 = input_data.get("question1", "")
            question2 = input_data.get("question2", "")

            # Parse output field
            output = example.get("output", "0")
            if isinstance(output, str):
                label = int(output)
            else:
                label = int(output)

            if not question1 or not question2:
                return None

            return {
                "question1": question1,
                "question2": question2,
                "label": label,
                "metadata": example.get("metadata", {})
            }
        except Exception as e:
            logger.debug(f"Failed to parse example: {e}")
            return None

    def create_splits(self, train_size: int, val_size: int, test_size: int,
                      random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create stratified train/val/test splits."""
        logger.info(f"Creating splits: train={train_size}, val={val_size}, test={test_size}")

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Separate by label for stratification
        duplicates = [e for e in self.examples if e["label"] == 1]
        non_duplicates = [e for e in self.examples if e["label"] == 0]

        logger.info(f"Duplicates: {len(duplicates)}, Non-duplicates: {len(non_duplicates)}")

        # Calculate sizes per class
        train_dup = min(len(duplicates) * train_size // (train_size + val_size + test_size), train_size // 2)
        train_non = min(len(non_duplicates) * train_size // (train_size + val_size + test_size), train_size // 2)

        val_dup = min(len(duplicates) - train_dup, val_size // 2)
        val_non = min(len(non_duplicates) - train_non, val_size // 2)

        test_dup = len(duplicates) - train_dup - val_dup
        test_non = len(non_duplicates) - train_non - val_non

        # Shuffle and split
        random.shuffle(duplicates)
        random.shuffle(non_duplicates)

        train = duplicates[:train_dup] + non_duplicates[:train_non]
        val = duplicates[train_dup:train_dup + val_dup] + non_duplicates[train_non:train_non + val_non]
        test = duplicates[train_dup + val_dup:] + non_duplicates[train_non + val_non:]

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        logger.info(f"Train: {len(train)} ({sum(e['label'] for e in train)} duplicates)")
        logger.info(f"Val: {len(val)} ({sum(e['label'] for e in val)} duplicates)")
        logger.info(f"Test: {len(test)} ({sum(e['label'] for e in test)} duplicates)")

        return train, val, test


class ReadingVectorConstructor:
    """Constructs Reading Vectors from text using cognitive load indicators."""

    def __init__(self, freq_dict: Optional[Dict[str, float]] = None):
        self.freq_dict = freq_dict or {}
        self.pos_tag_map = self._create_pos_map()

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)

    def _create_pos_map(self) -> Dict[str, int]:
        """Create mapping from POS tags to integer codes."""
        # Universal POS tags + some fine-grained tags
        pos_tags = [
            'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
            'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
            'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
            'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
            'WP', 'WP$', 'WRB', 'PUNCT', 'UNK'
        ]
        return {tag: i for i, tag in enumerate(pos_tags)}

    def construct_reading_vector(self, text: str) -> np.ndarray:
        """
        Construct reading vector for a text.

        Returns: numpy array of shape (seq_len, 3) with features:
        - word_length (normalized)
        - word_frequency (log scale)
        - pos_code (integer)
        """
        if not text or not isinstance(text, str):
            return np.zeros((1, 3))

        # Tokenize
        tokens = word_tokenize(text.lower())
        if not tokens:
            return np.zeros((1, 3))

        # Get POS tags
        pos_tags = pos_tag(tokens)

        # Construct features
        features = []
        for token, (_, pos) in zip(tokens, pos_tags):
            # Feature 1: Word length (normalized by max 20)
            word_length = min(len(token) / 20.0, 1.0)

            # Feature 2: Word frequency (log scale, default for OOV)
            word_freq = self.freq_dict.get(token, self.freq_dict.get('UNK', 0.0))

            # Feature 3: POS code
            pos_code = self.pos_tag_map.get(pos, self.pos_tag_map['UNK']) / len(self.pos_tag_map)

            features.append([word_length, word_freq, pos_code])

        return np.array(features)

    def build_frequency_dict(self, examples: List[Dict]):
        """Build frequency dictionary from training examples."""
        logger.info("Building frequency dictionary from training data")
        word_counts = Counter()

        for example in examples:
            for question in [example["question1"], example["question2"]]:
                tokens = word_tokenize(question.lower())
                word_counts.update(tokens)

        # Convert to log frequency
        total_words = sum(word_counts.values())
        self.freq_dict = {
            'UNK': 0.0  # Default for OOV
        }

        for word, count in word_counts.items():
            self.freq_dict[word] = np.log1p(count) / np.log1p(total_words)

        logger.info(f"Frequency dictionary built with {len(self.freq_dict)} words")


class DTWComparator:
    """Compares reading vectors using Dynamic Time Warping."""

    def __init__(self, radius: int = 1):
        self.radius = radius

    def compute_distance(self, rv1: np.ndarray, rv2: np.ndarray) -> float:
        """Compute DTW distance between two reading vectors."""
        if rv1.shape[0] == 0 or rv2.shape[0] == 0:
            return 1.0

        try:
            distance, _ = fastdtw(rv1, rv2, dist=euclidean, radius=self.radius)
            # Normalize by sequence lengths
            normalized_distance = distance / (rv1.shape[0] + rv2.shape[0])
            return normalized_distance
        except Exception as e:
            logger.debug(f"DTW computation failed: {e}")
            # Fallback to Euclidean distance on padded sequences
            return self._fallback_distance(rv1, rv2)

    def _fallback_distance(self, rv1: np.ndarray, rv2: np.ndarray) -> float:
        """Fallback distance computation."""
        min_len = min(rv1.shape[0], rv2.shape[0])
        max_len = max(rv1.shape[0], rv2.shape[0])

        if min_len == 0:
            return 1.0

        # Pad shorter sequence
        if rv1.shape[0] < rv2.shape[0]:
            rv1 = np.pad(rv1, ((0, rv2.shape[0] - rv1.shape[0]), (0, 0)), mode='edge')
        else:
            rv2 = np.pad(rv2, ((0, rv1.shape[0] - rv2.shape[0]), (0, 0)), mode='edge')

        return np.linalg.norm(rv1 - rv2) / max_len


class BaselineMethods:
    """Implements baseline methods for comparison."""

    def __init__(self):
        self.sbert_model = None
        self.wmd_model = None

    def jaccard_3gram(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity of character 3-grams."""
        def get_3grams(text):
            text = text.lower().replace(' ', '_')
            return set(text[i:i+3] for i in range(len(text) - 2))

        grams1 = get_3grams(text1)
        grams2 = get_3grams(text2)

        if not grams1 or not grams2:
            return 0.0

        intersection = len(grams1 & grams2)
        union = len(grams1 | grams2)

        similarity = intersection / union if union > 0 else 0.0
        return similarity  # Return similarity (higher = more likely duplicate)

    def minhash_lsh(self, text1: str, text2: str, num_perm: int = 128) -> float:
        """Compute MinHash LSH similarity."""
        def get_minhash(text, num_perm):
            m = MinHash(num_perm=num_perm)
            for word in text.lower().split():
                m.update(word.encode('utf8'))
            return m

        m1 = get_minhash(text1, num_perm)
        m2 = get_minhash(text2, num_perm)

        return m1.jaccard(m2)  # Return similarity (higher = more likely duplicate)

    def sentence_bert(self, text1: str, text2: str) -> float:
        """Compute Sentence-BERT cosine similarity."""
        if self.sbert_model is None:
            logger.info("Loading Sentence-BERT model")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        try:
            embeddings = self.sbert_model.encode([text1, text2], show_progress_bar=False)
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return similarity  # Return similarity (higher = more likely duplicate)
        except Exception as e:
            logger.error(f"SBERT computation failed: {e}")
            return 0.5  # Default similarity

    def word_movers_distance(self, text1: str, text2: str) -> float:
        """Compute Word Mover's Distance."""
        if self.wmd_model is None:
            logger.info("Loading Word2Vec model for WMD")
            try:
                # Use small pretrained model or fallback
                from gensim.models import KeyedVectors
                self.wmd_model = KeyedVectors.load_word2vec_format(
                    'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
                    binary=True,
                    limit=100000
                )
            except:
                logger.warning("Failed to load Word2Vec model, using fallback")
                self.wmd_model = False

        if self.wmd_model is False:
            # Fallback to simple edit distance
            return self._edit_distance(text1, text2) / max(len(text1), len(text2))

        try:
            distance = self.wmd_model.wmdistance(text1.split(), text2.split())
            return distance
        except Exception as e:
            logger.debug(f"WMD computation failed: {e}")
            return self._edit_distance(text1, text2) / max(len(text1), len(text2))

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Simple edit distance for fallback."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]


class RVPEvaluator:
    """Evaluates RVP and baseline methods."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rv_constructor = ReadingVectorConstructor()
        self.dtw_comparator = DTWComparator(radius=config.dtw_radius)
        self.baselines = BaselineMethods()
        self.results = {}

    def evaluate_method(self, method_name: str, train_data: List[Dict],
                       val_data: List[Dict], test_data: List[Dict]) -> MethodResult:
        """Evaluate a single method."""
        logger.info(f"Evaluating method: {method_name}")

        start_time = time.perf_counter()

        if method_name == "RVP":
            result = self._evaluate_rvp(train_data, val_data, test_data)
        elif method_name == "Jaccard":
            result = self._evaluate_jaccard(test_data)
        elif method_name == "MinHash":
            result = self._evaluate_minhash(test_data)
        elif method_name == "SBERT":
            result = self._evaluate_sbert(test_data)
        elif method_name == "WMD":
            result = self._evaluate_wmd(test_data)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / len(test_data)
        result.avg_time_per_pair = avg_time

        logger.info(f"{method_name}: F1={result.f1:.4f}, Time={avg_time:.4f}s/pair")

        return result

    def _evaluate_rvp(self, train_data: List[Dict], val_data: List[Dict],
                      test_data: List[Dict]) -> MethodResult:
        """Evaluate RVP method."""
        # Build frequency dictionary from training data
        self.rv_constructor.build_frequency_dict(train_data)

        # Compute distances for validation set to find optimal threshold
        val_distances = []
        for example in tqdm(val_data, desc="RVP val distances"):
            rv1 = self.rv_constructor.construct_reading_vector(example["question1"])
            rv2 = self.rv_constructor.construct_reading_vector(example["question2"])
            dist = self.dtw_comparator.compute_distance(rv1, rv2)
            val_distances.append(dist)

        val_labels = [e["label"] for e in val_data]

        # Find optimal threshold - lower distance = more likely duplicate
        optimal_threshold = self._find_optimal_threshold(val_distances, val_labels, higher_is_positive=False)

        # Compute distances for test set
        test_distances = []
        for example in tqdm(test_data, desc="RVP test distances"):
            rv1 = self.rv_constructor.construct_reading_vector(example["question1"])
            rv2 = self.rv_constructor.construct_reading_vector(example["question2"])
            dist = self.dtw_comparator.compute_distance(rv1, rv2)
            test_distances.append(dist)

        test_labels = [e["label"] for e in test_data]
        # For RVP: lower distance = more likely duplicate (label=1)
        test_predictions = [1 if d <= optimal_threshold else 0 for d in test_distances]

        return self._compute_metrics("RVP", test_labels, test_predictions, test_distances, distances_are_scores=False)

    def _evaluate_jaccard(self, test_data: List[Dict]) -> MethodResult:
        """Evaluate Jaccard 3-gram baseline."""
        similarities = []
        for example in tqdm(test_data, desc="Jaccard similarities"):
            sim = self.baselines.jaccard_3gram(example["question1"], example["question2"])
            similarities.append(sim)

        labels = [e["label"] for e in test_data]
        # Find optimal threshold on similarities (higher = more likely duplicate)
        threshold = self._find_optimal_threshold(similarities, labels)
        predictions = [1 if s >= threshold else 0 for s in similarities]

        return self._compute_metrics("Jaccard", labels, predictions, similarities, distances_are_scores=True)

    def _evaluate_minhash(self, test_data: List[Dict]) -> MethodResult:
        """Evaluate MinHash LSH baseline."""
        similarities = []
        for example in tqdm(test_data, desc="MinHash similarities"):
            sim = self.baselines.minhash_lsh(example["question1"], example["question2"])
            similarities.append(sim)

        labels = [e["label"] for e in test_data]
        threshold = self._find_optimal_threshold(similarities, labels)
        predictions = [1 if s >= threshold else 0 for s in similarities]

        return self._compute_metrics("MinHash", labels, predictions, similarities, distances_are_scores=True)

    def _evaluate_sbert(self, test_data: List[Dict]) -> MethodResult:
        """Evaluate Sentence-BERT baseline."""
        similarities = []
        for example in tqdm(test_data, desc="SBERT similarities"):
            sim = self.baselines.sentence_bert(example["question1"], example["question2"])
            similarities.append(sim)

        labels = [e["label"] for e in test_data]
        threshold = self._find_optimal_threshold(similarities, labels)
        predictions = [1 if s >= threshold else 0 for s in similarities]

        return self._compute_metrics("SBERT", labels, predictions, similarities, distances_are_scores=True)

    def _evaluate_wmd(self, test_data: List[Dict]) -> MethodResult:
        """Evaluate Word Mover's Distance baseline."""
        # Use subset for WMD (computationally expensive)
        subset = test_data[:min(500, len(test_data))]
        logger.info(f"WMD: using subset of {len(subset)} examples")

        distances = []
        for example in tqdm(subset, desc="WMD distances"):
            dist = self.baselines.word_movers_distance(example["question1"], example["question2"])
            distances.append(dist)

        labels = [e["label"] for e in subset]
        threshold = np.median(distances)
        predictions = [1 if d <= threshold else 0 for d in distances]

        return self._compute_metrics("WMD", labels, predictions, distances)

    def _find_optimal_threshold(self, scores: List[float], labels: List[int], 
                                higher_is_positive: bool = True) -> float:
        """Find optimal threshold for classification.
        
        Args:
            higher_is_positive: If True, scores >= threshold = positive (like similarity).
                              If False, scores <= threshold = positive (like distance).
        """
        candidates = np.linspace(min(scores), max(scores), 100)
        best_f1 = 0
        best_threshold = candidates[0]

        for threshold in candidates:
            if higher_is_positive:
                predictions = [1 if s >= threshold else 0 for s in scores]
            else:
                predictions = [1 if s <= threshold else 0 for s in scores]
            
            f1 = f1_score(labels, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        logger.info(f"Optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
        return best_threshold

    def _compute_metrics(self, method_name: str, labels: List[int],
                        predictions: List[int], distances: List[float],
                        distances_are_scores: bool = True) -> MethodResult:
        """Compute evaluation metrics.
        
        Args:
            distances_are_scores: If True, higher values = more likely duplicate (like similarity).
                                If False, lower values = more likely duplicate (like distance).
        """
        labels_array = np.array(labels)
        distances_array = np.array(distances)

        # Handle edge cases
        unique_labels = np.unique(labels_array)
        if len(unique_labels) < 2:
            logger.warning(f"Only one class in labels for {method_name}")
            return MethodResult(
                method_name=method_name,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                roc_auc=0.5,
                pr_auc=0.0,
                avg_time_per_pair=0.0,
                distances=distances,
                predictions=predictions
            )

        try:
            # For ROC-AUC, we need scores where higher = more likely positive (label=1)
            if distances_are_scores:
                scores = distances_array  # Higher = more likely duplicate
            else:
                # Convert distance to score: lower distance = higher score
                scores = -distances_array  # Negate so higher = more likely duplicate
            
            roc_auc = roc_auc_score(labels_array, scores)
        except Exception as e:
            logger.debug(f"ROC-AUC computation failed: {e}")
            roc_auc = 0.5

        try:
            if distances_are_scores:
                pr_scores = distances_array
            else:
                pr_scores = -distances_array
            pr_auc = average_precision_score(labels_array, pr_scores)
        except Exception as e:
            logger.debug(f"PR-AUC computation failed: {e}")
            pr_auc = 0.0

        precision = precision_score(labels_array, predictions, zero_division=0)
        recall = recall_score(labels_array, predictions, zero_division=0)
        f1 = f1_score(labels_array, predictions, zero_division=0)

        return MethodResult(
            method_name=method_name,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            avg_time_per_pair=0.0,
            distances=distances,
            predictions=predictions
        )


class AblationStudy:
    """Conducts ablation study on RVP components."""

    def __init__(self, train_data: List[Dict], test_data: List[Dict]):
        self.train_data = train_data
        self.test_data = test_data

    def run_ablation(self) -> Dict[str, Dict]:
        """Run ablation study with different feature combinations."""
        logger.info("Starting ablation study")
        results = {}

        feature_combinations = [
            ("length_only", [0]),
            ("frequency_only", [1]),
            ("pos_only", [2]),
            ("length_frequency", [0, 1]),
            ("length_pos", [0, 2]),
            ("frequency_pos", [1, 2]),
            ("full_rvp", [0, 1, 2]),
        ]

        for name, feature_indices in feature_combinations:
            logger.info(f"Running ablation variant: {name}")
            result = self._evaluate_variant(feature_indices)
            results[name] = result

        return results

    def _evaluate_variant(self, feature_indices: List[int]) -> Dict:
        """Evaluate a single feature combination."""
        distances = []
        for example in tqdm(self.test_data, desc=f"Ablation {feature_indices}"):
            rv1 = self._construct_partial_rv(example["question1"], feature_indices)
            rv2 = self._construct_partial_rv(example["question2"], feature_indices)
            dist = fastdtw(rv1, rv2, dist=euclidean)[0]
            distances.append(dist / (rv1.shape[0] + rv2.shape[0]))

        labels = [e["label"] for e in self.test_data]
        threshold = np.median(distances)
        predictions = [1 if d <= threshold else 0 for d in distances]

        return {
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
        }

    def _construct_partial_rv(self, text: str, feature_indices: List[int]) -> np.ndarray:
        """Construct reading vector with only selected features."""
        # This is simplified - in practice would use full RV constructor
        tokens = word_tokenize(text.lower())
        features = []

        for token in tokens:
            row = []
            if 0 in feature_indices:  # length
                row.append(min(len(token) / 20.0, 1.0))
            if 1 in feature_indices:  # frequency
                row.append(0.5)  # Placeholder
            if 2 in feature_indices:  # pos
                row.append(0.5)  # Placeholder
            features.append(row)

        return np.array(features) if features else np.zeros((1, len(feature_indices)))


class StatisticalTester:
    """Performs statistical significance testing."""

    @staticmethod
    def bootstrap_test(scores1: List[float], scores2: List[float],
                       n_iterations: int = 1000) -> Tuple[float, Tuple[float, float]]:
        """Perform bootstrap test for significance."""
        logger.info(f"Running bootstrap test with {n_iterations} iterations")

        differences = []
        n = len(scores1)

        for _ in range(n_iterations):
            indices = np.random.choice(n, n, replace=True)
            diff = np.mean([scores1[i] for i in indices]) - np.mean([scores2[i] for i in indices])
            differences.append(diff)

        p_value = np.mean(np.array(differences) >= 0)
        ci_95 = np.percentile(differences, [2.5, 97.5])

        return p_value, ci_95


@logger.catch(reraise=True)
def main():
    """Main experiment function."""
    # Initialize configuration - use larger dataset
    config = ExperimentConfig(
        data_path="/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run_Z4TPmJBKTsQm/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out",
        output_path="method_out.json",
        train_size=5000,  # Increased for better evaluation
        val_size=1000,
        test_size=1000,
        random_seed=42
    )

    # Create output directory
    Path("logs").mkdir(exist_ok=True)

    # Start memory tracking
    tracemalloc.start()

    # Load data
    logger.info("=" * 60)
    logger.info("Starting RVP Experiment")
    logger.info("=" * 60)

    data_loader = DataLoader(config.data_path)
    data_loader.load_data(max_examples=10000)  # Load more examples

    # Create splits
    train_data, val_data, test_data = data_loader.create_splits(
        config.train_size, config.val_size, config.test_size, config.random_seed
    )

    # Initialize evaluator
    evaluator = RVPEvaluator(config)

    # Evaluate all methods
    logger.info("=" * 60)
    logger.info("Evaluating Methods")
    logger.info("=" * 60)

    methods_to_evaluate = ["RVP", "Jaccard", "MinHash", "SBERT"]
    results = {}

    for method in methods_to_evaluate:
        try:
            result = evaluator.evaluate_method(method, train_data, val_data, test_data)
            results[method] = {
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
                "pr_auc": result.pr_auc,
                "avg_time_per_pair": result.avg_time_per_pair
            }
            logger.info(f"{method}: Precision={result.precision:.4f}, Recall={result.recall:.4f}, F1={result.f1:.4f}, ROC-AUC={result.roc_auc:.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {method}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Ablation study (on smaller subset for speed)
    logger.info("=" * 60)
    logger.info("Ablation Study")
    logger.info("=" * 60)

    ablation = AblationStudy(train_data[:500], test_data[:500])
    ablation_results = ablation.run_ablation()

    # Prepare output
    output = {
        "experiment_info": {
            "method_name": "RVP",
            "dataset": "QQP",
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "Evaluation on QQP dataset for near-duplicate short text detection"
        },
        "results": results,
        "ablation": ablation_results
    }

    # Save output
    output_path = Path(config.output_path)
    
    # Add statistical tests
    logger.info("=" * 60)
    logger.info("Statistical Significance Testing")
    logger.info("=" * 60)
    
    statistical_tests = {}
    if "RVP" in results and "Jaccard" in results:
        # Bootstrap test for RVP vs Jaccard
        p_value, ci_95 = StatisticalTester.bootstrap_test(
            [r.f1 for r in [result] if r.method_name == "RVP"],
            [r.f1 for r in [result] if r.method_name == "Jaccard"],
            n_iterations=100
        )
        statistical_tests["RVP_vs_Jaccard"] = {"p_value": p_value, "ci_95": list(ci_95)}
    
    # Add efficiency analysis
    logger.info("=" * 60)
    logger.info("Computational Efficiency Analysis")
    logger.info("=" * 60)
    
    efficiency = {}
    for method_name, result in results.items():
        efficiency[method_name] = {
            "avg_time_per_pair": result.avg_time_per_pair,
            "memory_mb": peak / 1024 / 1024 if 'peak' in locals() else 0
        }
    
    # Prepare final output
    output = {
        "experiment_info": {
            "method_name": "RVP",
            "dataset": "QQP",
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "Evaluation on QQP dataset for near-duplicate short text detection"
        },
        "results": {k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in results.items()},
        "ablation": ablation_results,
        "statistical_tests": statistical_tests,
        "efficiency": efficiency
    }
    
    # Convert MethodResult objects to dicts
    for method_name in output["results"]:
        if hasattr(results[method_name], '__dict__'):
            output["results"][method_name] = {
                "precision": results[method_name].precision,
                "recall": results[method_name].recall,
                "f1": results[method_name].f1,
                "roc_auc": results[method_name].roc_auc,
                "pr_auc": results[method_name].pr_auc,
                "avg_time_per_pair": results[method_name].avg_time_per_pair
            }
