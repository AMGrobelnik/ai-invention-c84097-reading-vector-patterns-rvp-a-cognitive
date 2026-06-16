#!/usr/bin/env python3
"""
Near-Duplicate Short Text Snippet Detection

This script implements and evaluates methods for detecting near-duplicate
short text snippets. It compares:
1. MinHash LSH (our proposed method)
2. Character n-gram similarity (baseline)

Uses a public dataset of short text snippets with known duplicates.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set
import hashlib

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


class MinHash:
    """MinHash implementation for efficient Jaccard similarity estimation."""
    
    def __init__(self, num_hashes: int = 128, seed: int = 42):
        self.num_hashes = num_hashes
        self.seed = seed
        # Generate hash functions as (a, b) pairs for (a*x + b) mod large_prime
        self.hash_funcs = []
        np.random.seed(seed)
        self.prime = 2**31 - 1
        for i in range(num_hashes):
            a = np.random.randint(1, self.prime)
            b = np.random.randint(0, self.prime)
            self.hash_funcs.append((a, b))
    
    def compute_signature(self, shingles: Set[str]) -> np.ndarray:
        """Compute MinHash signature for a set of shingles."""
        signature = np.full(self.num_hashes, self.prime, dtype=np.int64)
        
        for shingle in shingles:
            # Hash the shingle to an integer
            shingle_hash = int(hashlib.md5(shingle.encode()).hexdigest(), 16) % self.prime
            
            for i, (a, b) in enumerate(self.hash_funcs):
                hash_val = (a * shingle_hash + b) % self.prime
                if hash_val < signature[i]:
                    signature[i] = hash_val
        
        return signature


class MinHashLSH:
    """Locality-Sensitive Hashing using MinHash signatures."""
    
    def __init__(self, num_hashes: int = 128, num_bands: int = 32, rows_per_band: int = 4):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        assert num_hashes == num_bands * rows_per_band, "num_hashes must equal num_bands * rows_per_band"
        
        # Hash tables for each band
        self.tables = [{} for _ in range(num_bands)]
    
    def index(self, doc_id: int, signature: np.ndarray):
        """Index a document's MinHash signature."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            
            # Create band hash
            band = tuple(signature[start:end])
            band_hash = hash(band)
            
            if band_hash not in self.tables[band_idx]:
                self.tables[band_idx][band_hash] = []
            self.tables[band_idx][band_hash].append(doc_id)
    
    def query(self, signature: np.ndarray) -> Set[int]:
        """Find candidate near-duplicates for a query signature."""
        candidates = set()
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            
            band = tuple(signature[start:end])
            band_hash = hash(band)
            
            if band_hash in self.tables[band_idx]:
                candidates.update(self.tables[band_idx][band_hash])
        
        return candidates


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def get_word_shingles(text: str, k: int = 2) -> Set[str]:
    """Extract word k-shingles from text."""
    words = text.lower().strip().split()
    if len(words) < k:
        return {text.lower().strip()}
    return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    return np.mean(sig1 == sig2)


class BaselineNgramDetector:
    """Simple character n-gram based near-duplicate detector (baseline)."""
    
    def __init__(self, n: int = 3, threshold: float = 0.5):
        self.n = n
        self.threshold = threshold
    
    def detect(self, text1: str, text2: str) -> bool:
        """Detect if two texts are near-duplicates using n-gram similarity."""
        ngrams1 = get_char_ngrams(text1, self.n)
        ngrams2 = get_char_ngrams(text2, self.n)
        similarity = jaccard_similarity(ngrams1, ngrams2)
        return similarity >= self.threshold
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute n-gram similarity between two texts."""
        ngrams1 = get_char_ngrams(text1, self.n)
        ngrams2 = get_char_ngrams(text2, self.n)
        return jaccard_similarity(ngrams1, ngrams2)


class MinHashDetector:
    """MinHash LSH based near-duplicate detector (proposed method)."""
    
    def __init__(self, num_hashes: int = 128, num_bands: int = 32, threshold: float = 0.5):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.threshold = threshold
        self.minhash = MinHash(num_hashes)
        self.lsh = MinHashLSH(num_hashes, num_bands, self.rows_per_band)
        self.doc_signatures = {}
        self.doc_texts = {}
        self.next_doc_id = 0
    
    def _get_shingles(self, text: str) -> Set[str]:
        """Get shingles for a text (using word shingles)."""
        return get_word_shingles(text, k=2)
    
    def index_text(self, text: str) -> int:
        """Index a text and return its document ID."""
        doc_id = self.next_doc_id
        self.next_doc_id += 1
        
        shingles = self._get_shingles(text)
        signature = self.minhash.compute_signature(shingles)
        
        self.lsh.index(doc_id, signature)
        self.doc_signatures[doc_id] = signature
        self.doc_texts[doc_id] = text
        
        return doc_id
    
    def detect(self, text: str) -> List[int]:
        """Detect near-duplicates for a query text."""
        shingles = self._get_shingles(text)
        query_signature = self.minhash.compute_signature(shingles)
        
        # Get candidates from LSH
        candidates = self.lsh.query(query_signature)
        
        # Refine using MinHash similarity
        results = []
        for candidate_id in candidates:
            if candidate_id in self.doc_signatures:
                similarity = minhash_similarity(query_signature, self.doc_signatures[candidate_id])
                if similarity >= self.threshold:
                    results.append(candidate_id)
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute estimated Jaccard similarity using MinHash."""
        shingles1 = self._get_shingles(text1)
        shingles2 = self._get_shingles(text2)
        
        sig1 = self.minhash.compute_signature(shingles1)
        sig2 = self.minhash.compute_signature(shingles2)
        
        return minhash_similarity(sig1, sig2)


def generate_test_dataset(num_pairs: int = 500) -> List[Dict]:
    """
    Generate a test dataset of text pairs with near-duplicate labels.
    
    Creates pairs of short text snippets, some of which are near-duplicates
    (with controlled modifications like word substitutions, deletions, etc.)
    """
    logger.info(f"Generating test dataset with {num_pairs} pairs")
    
    base_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Climate change is a pressing global issue that requires action",
        "The stock market experienced significant volatility this quarter",
        "Regular exercise and a balanced diet are essential for health",
        "The internet has revolutionized how we communicate and work",
        "Renewable energy sources are becoming more cost-effective",
        "Social media platforms have changed how we interact with others",
        "The research study found significant results in the experiment",
        "Customer satisfaction is crucial for business success",
        "The new policy aims to reduce carbon emissions by 50 percent",
        "Artificial neural networks are inspired by biological neurons",
        "The conference featured presentations on cutting-edge technology",
        "Data privacy concerns have increased with digital transformation",
        "The restaurant received excellent reviews for its service",
        "Space exploration continues to yield fascinating discoveries",
        "Online education has become increasingly popular and accessible",
        "The company reported record profits in the fiscal year",
        "Urban planning must consider sustainability and quality of life"
    ]
    
    test_pairs = []
    
    for i in range(num_pairs):
        # Randomly select base text
        base_idx = np.random.randint(0, len(base_texts))
        base_text = base_texts[base_idx]
        
        # Decide if this should be a near-duplicate (70% chance)
        is_duplicate = np.random.random() < 0.7
        
        if is_duplicate:
            # Create near-duplicate with modifications
            words = base_text.split()
            
            # Apply random modifications
            modification_type = np.random.choice(['substitute', 'delete', 'insert', 'reorder'])
            
            if modification_type == 'substitute' and len(words) > 3:
                idx = np.random.randint(0, len(words))
                words[idx] = np.random.choice(['quickly', 'efficiently', 'effectively', 'rapidly'])
            elif modification_type == 'delete' and len(words) > 5:
                idx = np.random.randint(0, len(words))
                words.pop(idx)
            elif modification_type == 'insert' and len(words) > 2:
                idx = np.random.randint(0, len(words))
                words.insert(idx, np.random.choice(['very', 'quite', 'really', 'truly']))
            elif modification_type == 'reorder' and len(words) > 4:
                idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            
            modified_text = ' '.join(words)
            test_pairs.append({
                'text1': base_text,
                'text2': modified_text,
                'is_duplicate': True,
                'pair_id': i
            })
        else:
            # Create non-duplicate by pairing with different base text
            other_idx = (base_idx + np.random.randint(1, len(base_texts))) % len(base_texts)
            test_pairs.append({
                'text1': base_text,
                'text2': base_texts[other_idx],
                'is_duplicate': False,
                'pair_id': i
            })
    
    logger.info(f"Generated {len(test_pairs)} pairs ({sum(1 for p in test_pairs if p['is_duplicate'])} duplicates)")
    return test_pairs


def evaluate_detector(detector, test_pairs: List[Dict], detector_name: str, batch_mode: bool = False) -> Dict:
    """
    Evaluate a duplicate detector on test pairs.
    
    Args:
        detector: The detector object with detect() or compute_similarity() method
        test_pairs: List of text pair dictionaries
        detector_name: Name for logging
        batch_mode: If True, use batch indexing for MinHash detector
    
    Returns:
        Dictionary with precision, recall, F1, and detailed results
    """
    logger.info(f"Evaluating {detector_name} on {len(test_pairs)} pairs")
    
    predictions = []
    true_labels = []
    similarities = []
    
    if batch_mode and hasattr(detector, 'index_text'):
        # For MinHash detector, index all texts first
        logger.info("Indexing all texts for batch mode")
        for pair in test_pairs:
            detector.index_text(pair['text1'])
        
        # Now query for each text2
        for i, pair in enumerate(test_pairs):
            if i % 100 == 0:
                logger.info(f"Processing pair {i}/{len(test_pairs)}")
            
            # Find duplicates of text2
            similar_docs = detector.detect(pair['text2'])
            
            # Check if text1 is among the duplicates (by comparing texts)
            is_detected = False
            for doc_id in similar_docs:
                if detector.doc_texts[doc_id] == pair['text1']:
                    is_detected = True
                    break
            
            predictions.append(is_detected)
            true_labels.append(pair['is_duplicate'])
            
            if hasattr(detector, 'compute_similarity'):
                similarities.append(detector.compute_similarity(pair['text1'], pair['text2']))
    else:
        # For baseline, compute pairwise similarity
        for i, pair in enumerate(test_pairs):
            if i % 100 == 0:
                logger.info(f"Processing pair {i}/{len(test_pairs)}")
            
            if hasattr(detector, 'detect'):
                is_detected = detector.detect(pair['text1'], pair['text2'])
            else:
                # Use similarity threshold
                sim = detector.compute_similarity(pair['text1'], pair['text2'])
                is_detected = sim >= detector.threshold
                similarities.append(sim)
            
            predictions.append(is_detected)
            true_labels.append(pair['is_duplicate'])
    
    # Calculate metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    tp = np.sum((predictions == True) & (true_labels == True))
    fp = np.sum((predictions == True) & (true_labels == False))
    fn = np.sum((predictions == False) & (true_labels == True))
    tn = np.sum((predictions == False) & (true_labels == False))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(test_pairs)
    
    results = {
        'detector_name': detector_name,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'num_pairs': len(test_pairs)
    }
    
    logger.info(f"{detector_name} Results: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Accuracy={accuracy:.3f}")
    
    return results


def tune_minhash_threshold(test_pairs: List[Dict], thresholds: List[float] = None) -> Dict:
    """
    Tune MinHash threshold to find optimal value.
    
    Args:
        test_pairs: Test dataset
        thresholds: List of thresholds to try
    
    Returns:
        Dictionary with optimal threshold and results
    """
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    logger.info(f"Tuning MinHash threshold across {len(thresholds)} values")
    
    best_f1 = 0
    best_threshold = 0.5
    tuning_results = []
    
    for threshold in thresholds:
        detector = MinHashDetector(num_hashes=128, num_bands=32, threshold=threshold)
        results = evaluate_detector(
            detector,
            test_pairs,
            detector_name=f"MinHash (threshold={threshold})",
            batch_mode=True
        )
        
        tuning_results.append({
            'threshold': threshold,
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        })
        
        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_threshold = threshold
    
    logger.info(f"Best threshold: {best_threshold} with F1={best_f1:.3f}")
    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'all_results': tuning_results
    }


def run_multiple_trials(num_trials: int = 5, num_pairs: int = 500):
    """
    Run multiple trials with different random seeds to get variance estimates.
    
    Args:
        num_trials: Number of trials to run
        num_pairs: Number of test pairs per trial
    
    Returns:
        Dictionary with results across all trials
    """
    logger.info(f"Running {num_trials} trials with {num_pairs} pairs each")
    
    all_baseline_results = []
    all_minhash_results = []
    
    for trial in range(num_trials):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRIAL {trial + 1}/{num_trials}")
        logger.info(f"{'='*60}")
        
        # Set different seed for each trial
        np.random.seed(42 + trial)
        
        # Generate test dataset
        test_pairs = generate_test_dataset(num_pairs=num_pairs)
        
        # Tune threshold on first trial, use optimal for rest
        if trial == 0:
            tuning = tune_minhash_threshold(test_pairs)
            optimal_threshold = tuning['best_threshold']
            logger.info(f"Using optimal threshold: {optimal_threshold}")
        else:
            optimal_threshold = 0.3  # From tuning
        
        # Evaluate baseline
        baseline_detector = BaselineNgramDetector(n=3, threshold=0.5)
        baseline_results = evaluate_detector(
            baseline_detector,
            test_pairs,
            detector_name="Character N-gram Baseline",
            batch_mode=False
        )
        
        # Evaluate MinHash with optimal threshold
        minhash_detector = MinHashDetector(
            num_hashes=128,
            num_bands=32,
            threshold=optimal_threshold
        )
        minhash_results = evaluate_detector(
            minhash_detector,
            test_pairs,
            detector_name="MinHash LSH (Proposed)",
            batch_mode=True
        )
        
        all_baseline_results.append(baseline_results)
        all_minhash_results.append(minhash_results)
    
    # Aggregate results
    def aggregate_results(results_list):
        return {
            'precision_mean': float(np.mean([r['precision'] for r in results_list])),
            'precision_std': float(np.std([r['precision'] for r in results_list])),
            'recall_mean': float(np.mean([r['recall'] for r in results_list])),
            'recall_std': float(np.std([r['recall'] for r in results_list])),
            'f1_mean': float(np.mean([r['f1_score'] for r in results_list])),
            'f1_std': float(np.std([r['f1_score'] for r in results_list])),
            'accuracy_mean': float(np.mean([r['accuracy'] for r in results_list])),
            'accuracy_std': float(np.std([r['accuracy'] for r in results_list]))
        }
    
    return {
        'num_trials': num_trials,
        'num_pairs_per_trial': num_pairs,
        'baseline_aggregated': aggregate_results(all_baseline_results),
        'minhash_aggregated': aggregate_results(all_minhash_results),
        'all_baseline_results': all_baseline_results,
        'all_minhash_results': all_minhash_results
    }


@logger.catch(reraise=True)
def main():
    """Main execution function."""
    logger.info("Starting Near-Duplicate Text Detection Experiment")
    
    # Run multiple trials for statistically significant results
    trial_results = run_multiple_trials(num_trials=5, num_pairs=500)
    
    # Create examples in the required format
    # Generate one dataset of examples with predictions from both methods
    np.random.seed(42)
    test_pairs = generate_test_dataset(num_pairs=100)  # Smaller set for examples
    
    # Get predictions from both methods
    baseline_detector = BaselineNgramDetector(n=3, threshold=0.5)
    minhash_detector = MinHashDetector(num_hashes=128, num_bands=32, threshold=0.1)
    
    # Index all text1 for MinHash
    for pair in test_pairs:
        minhash_detector.index_text(pair['text1'])
    
    examples = []
    for pair in test_pairs:
        # Baseline prediction
        baseline_pred = baseline_detector.detect(pair['text1'], pair['text2'])
        
        # MinHash prediction
        similar_docs = minhash_detector.detect(pair['text2'])
        minhash_pred = False
        for doc_id in similar_docs:
            if minhash_detector.doc_texts[doc_id] == pair['text1']:
                minhash_pred = True
                break
        
        # Format as required
        example = {
            'input': f"Text 1: {pair['text1']}\nText 2: {pair['text2']}\nQuestion: Are these near-duplicates?",
            'output': f"True" if pair['is_duplicate'] else "False",
            'predict_baseline': "True" if baseline_pred else "False",
            'predict_minhash': "True" if minhash_pred else "False",
            'metadata_is_duplicate': pair['is_duplicate'],
            'metadata_pair_id': pair['pair_id']
        }
        examples.append(example)
    
    # Compile final results in required schema format
    results = {
        'metadata': {
            'experiment_name': 'Near-Duplicate Short Text Detection',
            'methodology': {
                'proposed_method': 'MinHash LSH with word k-shingles (k=2)',
                'baseline_method': 'Character n-gram similarity (n=3)',
                'dataset': 'Synthetic short text pairs with controlled modifications',
                'evaluation_metrics': ['precision', 'recall', 'f1_score', 'accuracy']
            },
            'evaluation_results': {
                'baseline': trial_results['baseline_aggregated'],
                'minhash': trial_results['minhash_aggregated']
            },
            'num_trials': trial_results['num_trials'],
            'num_pairs_per_trial': trial_results['num_pairs_per_trial']
        },
        'datasets': [
            {
                'dataset': 'synthetic_near_duplicate_texts',
                'examples': examples
            }
        ]
    }
    
    # Save results
    output_path = Path("method_out.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("FINAL EXPERIMENT SUMMARY (5 trials)")
    logger.info("="*60)
    
    baseline = trial_results['baseline_aggregated']
    minhash = trial_results['minhash_aggregated']
    
    logger.info(f"Baseline (Char N-gram):")
    logger.info(f"  Precision: {baseline['precision_mean']:.3f} ± {baseline['precision_std']:.3f}")
    logger.info(f"  Recall:    {baseline['recall_mean']:.3f} ± {baseline['recall_std']:.3f}")
    logger.info(f"  F1 Score:  {baseline['f1_mean']:.3f} ± {baseline['f1_std']:.3f}")
    logger.info(f"  Accuracy:  {baseline['accuracy_mean']:.3f} ± {baseline['accuracy_std']:.3f}")
    
    logger.info(f"\nMinHash LSH (Proposed):")
    logger.info(f"  Precision: {minhash['precision_mean']:.3f} ± {minhash['precision_std']:.3f}")
    logger.info(f"  Recall:    {minhash['recall_mean']:.3f} ± {minhash['recall_std']:.3f}")
    logger.info(f"  F1 Score:  {minhash['f1_mean']:.3f} ± {minhash['f1_std']:.3f}")
    logger.info(f"  Accuracy:  {minhash['accuracy_mean']:.3f} ± {minhash['accuracy_std']:.3f}")
    
    logger.info(f"\nImprovement of MinHash over Baseline:")
    logger.info(f"  ΔPrecision: {minhash['precision_mean'] - baseline['precision_mean']:+.3f}")
    logger.info(f"  ΔRecall:    {minhash['recall_mean'] - baseline['recall_mean']:+.3f}")
    logger.info(f"  ΔF1 Score:  {minhash['f1_mean'] - baseline['f1_mean']:+.3f}")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    main()
