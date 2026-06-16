# Related Work Candidates for RVP Paper

## 1. Broder (1997) - MinHash for Document Resemblance

**Full Citation (BibTeX)**:
```bibtex
@inproceedings{broder1997resemblance,
  title={On the resemblance and containment of documents},
  author={Broder, Andrei Z.},
  booktitle={Proceedings of the Compression and Complexity of Sequences},
  year={1997}
}
```

**Method Summary**: Introduces MinHash algorithm for efficiently estimating set resemblance and containment. Documents represented as sets of shingles (contiguous subsequences). Uses random sampling to estimate Jaccard similarity.

**Relevance to RVP**: 
- **Similarity**: Also addresses near-duplicate detection
- **Difference**: Set-based (bag-of-shingles), not sequence-based. Doesn't use cognitive features. Limitation for short text: sparse shingle sets.

**Why RVP is Different**: RVP uses sequence representation preserving order; MinHash discards order. RVP uses cognitive features; MinHash uses raw tokens.

## 2. Kusner et al. (2015) - Word Mover's Distance

**Full Citation (BibTeX)**:
```bibtex
@inproceedings{kusner2015word,
  title={From word embeddings to document distances},
  author={Kusner, Matt and Sun, Yu and Kolkin, Nicholas and Weinberger, Kilian},
  booktitle={Proceedings of the 32nd International Conference on Machine Learning (ICML)},
  volume={37},
  pages={957--966},
  year={2015}
}
```

**Method Summary**: Introduces WMD - a distance metric between documents based on word2vec embeddings. Measures minimum cumulative distance that words from one document need to "travel" to match another document. Cast as Earth Mover's Distance (optimal transport).

**Relevance to RVP**:
- **Similarity**: Also aligns "words" between documents for similarity
- **Difference**: Uses embeddings (requires pre-training); RVP uses simple cognitive features (no training needed). WMD is O(n²) complexity; RVP with DTW is O(n²) but on cognitive feature sequences (not embeddings).

**Why RVP is Different**: RVP is interpretable (cognitive features); WMD is not (embeddings). RVP works on short text; WMD may struggle with very short text (few words = sparse point cloud).

## 3. Das & Smith (2009) - Quasi-Synchronous Grammar for Paraphrase

**Full Citation (BibTeX)**:
```bibtex
@inproceedings{das2009paraphrase,
  title={Paraphrase identification as probabilistic quasi-synchronous recognition},
  author={Das, Dipanjan and Smith, Noah A.},
  booktitle={Proceedings of ACL/IJCNLP},
  year={2009}
}
```

**Method Summary**: Uses quasi-synchronous dependency grammars to model paraphrase relationships. Incorporates both syntax (dependency trees) and lexical semantics (WordNet). Uses product of experts to combine with lexical overlap features.

**Relevance to RVP**:
- **Similarity**: Uses POS/syntax and sequence alignment for text similarity
- **Difference**: Complex grammatical model; RVP uses simple cognitive features (length, frequency, POS) as numerical sequences. Not using DTW.

**Why RVP is Different**: RVP is simpler and more efficient. RVP uses cognitive load indicators (not just syntax). RVP uses DTW for alignment (not quasi-synchronous grammars).

## 4. Vajjala & Meurers (2012) - Readability Classification

**Full Citation (BibTeX)**:
```bibtex
@inproceedings{vajjala2012improving,
  title={On improving the accuracy of readability classification using insights from second language acquisition},
  author={Vajjala, Sowmya and Meurers, Detmar},
  booktitle={Proceedings of the 7th Workshop on the Innovative Use of NLP for Building Educational Applications},
  pages={163--173},
  year={2012}
}
```

**Method Summary**: Uses lexical and syntactic features (word n-grams, POS n-grams, type-token ratio, syntactic complexity measures from SLA research) for readability assessment. Features are aggregated per document.

**Relevance to RVP**:
- **Similarity**: Uses word length, frequency, and POS as features
- **Difference**: Features are aggregated (bag-of-words style); RVP represents them as sequences preserving order. Classification task; RVP is similarity/distance task.

**Why RVP is Different**: RVP preserves sequence order (important for short text where order carries meaning). RVP uses sequences for pairwise similarity; this paper uses aggregated features for classification.

## 5. Danaila et al. (2012) - String Distances for Near-Duplicate Detection

**Full Citation (BibTeX)**:
```bibtex
@unpublished{danaila2012string,
  title={String distances for near-duplicate detection},
  author={Danaila, Iulia and Dinu, Liviu P. and Niculae, Vlad and Sulea, Octavia-Maria},
  year={2012},
  note={University of Bucharest}
}
```

**Method Summary**: Compares Rank distance and Smith-Waterman distance with Levenshtein distance for near-duplicate detection. Uses string edit distances.

**Relevance to RVP**:
- **Similarity**: Uses sequence alignment for duplicate detection
- **Difference**: Aligns character or word sequences directly; RVP aligns cognitive feature sequences.

**Why RVP is Different**: RVP aligns cognitive feature sequences (capturing reading difficulty/cognitive load); Smith-Waterman aligns characters/words (capturing surface form).

## 6. SEO (2005) - Scanpath Similarity for Reading

**Note**: This is a conjectured related work based on the hypothesis mentioning "eye-tracking / scanpath similarity". Need to find actual paper.

**Potential Relevance to RVP**:
- Eye-tracking patterns during reading could be related to "reading vectors"
- **Difference**: Requires eye-tracking equipment; RVP simulates cognitive load with simple features

**Search Needed**: Find actual papers on scanpath similarity for text comparison.
