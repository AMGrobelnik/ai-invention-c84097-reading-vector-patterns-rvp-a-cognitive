# Literature Review: Cognitive Features and DTW for Near-Duplicate Text Detection

## Summary

This literature review examined whether the Reading Vector Patterns (RVP) approach—combining cognitive load indicators (word length, log frequency, part-of-speech) represented as sequences with Dynamic Time Warping (DTW) alignment for near-duplicate short text detection—represents a novel contribution. Through 15+ web searches and 5 detailed paper examinations, the research found: (1) Individual cognitive features are well-established in readability assessment [4] but used as aggregated features, not sequences. (2) DTW is primarily applied to speech/audio processing, not text similarity with cognitive features [Searches 5, 11]. (3) Existing near-duplicate detection methods (MinHash [1], WMD [2], edit distances [5]) use set-based or embedding-based representations, not cognitive feature sequences. (4) The specific combination of cognitive features AS SEQUENCES with DTW alignment for duplicate detection was NOT FOUND in the literature. The assessment is PARTIAL NOVELTY: the combination appears novel (70-80% confidence), but individual components exist. The review produced 5 deliverable files: papers_surveyed.json (5 papers examined), novelty_assessment.md (aspect-by-aspect analysis), related_work_candidates.md (6 candidates with BibTeX), motivation_citations.md (6 citation categories), and search_log.md (15 searches documented).

## Research Findings

The literature review investigated whether the Reading Vector Patterns (RVP) approach—combining cognitive load indicators (word length, log frequency, part-of-speech) as sequences with Dynamic Time Warping (DTW) alignment for near-duplicate short text detection—represents a novel contribution. 

## Novelty Assessment

**PARTIAL NOVELTY** - The RVP approach appears to combine existing components in a novel way, with 70-80% confidence.

### Individual Components (NOT Novel):
- Word length, frequency, and POS tags are well-established features in readability assessment and text classification [4]
- Dynamic Time Warping (DTW) is a known algorithm, primarily used for speech/audio processing and time series analysis [Searches 5, 11]
- Sequence alignment for text has been explored (e.g., Smith-Waterman for near-duplicate detection [5], quasi-synchronous grammars for paraphrase [3])

### The Combination (LIKELY Novel):
- **No evidence found** of using word length + frequency + POS TOGETHER AS SEQUENCES (preserving order) for text similarity [Searches 1-15]
- **No evidence found** of applying DTW to align cognitive feature sequences (word length, frequency, POS sequences) [Searches 5, 11, 12]
- **No evidence found** of this specific combination for near-duplicate detection [Searches 8, 9]

### Closest Existing Methods (Not Direct Competitors):
1. **Word Mover's Distance (WMD) [2]**: Also aligns 'words' between documents, but uses embeddings + Earth Mover's Distance (optimal transport), not cognitive features + DTW. Requires pre-trained embeddings; RVP uses simple, interpretable features.
2. **Readability Assessment [4]**: Uses word length, frequency, POS, but as aggregated features (bag-of-words style), not as sequences. Classification task, not similarity/distance.
3. **Quasi-Synchronous Grammar [3]**: Uses syntax and alignment for paraphrase, but complex grammatical model. Not using cognitive load indicators.
4. **Smith-Waterman for Duplicates [5]**: Uses sequence alignment, but aligns characters/words directly, not cognitive feature sequences.

## Limitations of Existing Methods for Short Text

1. **MinHash [1]**: Set-based (shingles), discards order. For short text, shingle sets may be too sparse for meaningful similarity estimation. Designed for long documents (30M web pages evaluation).
2. **WMD [2]**: Acknowledges challenges with short documents - 'effect might be artificially magnified due to the short document lengths' [2, Section 4]. Requires sufficient words to form meaningful point cloud.
3. **Edit distances (Levenshtein, Smith-Waterman) [5]**: Capture surface form, not semantic/cognitive similarity. May not capture 'same meaning, different words' paraphrases.

## Cognitive Features Capture Unique Information

- Readability assessment papers [4] show that cognitive features (word length, frequency, POS) improve performance over traditional features alone (93.3% accuracy vs. baselines).
- Psycholinguistic research shows word length and frequency affect reading effort [Searches 4, 7] - supports using these as cognitive load indicators.
- Sequence information matters for similarity [3] - alignment between text parts improves paraphrase detection.

## Evidence Gaps (Limitations of This Review)

1. **Limited search depth**: May have missed papers using different terminology. No systematic search of ACL Anthology, EMNLP, SIGIR proceedings.
2. **Access limitations**: Some papers could not be fetched (HTTP 403 errors) [Searches 2, 6].
3. **No direct comparison**: Haven't implemented RVP to empirically compare with existing methods.
4. **Short text focus**: Found motivation that short text is challenging [5, 2], but need stronger evidence that cognitive features specifically help for short text duplicate detection.

## Recommendations

1. **For novelty strengthening**: Conduct systematic search in ACL Anthology, EMNLP, SIGIR proceedings. Search for 'cognitive features' + 'sequence alignment' more broadly.
2. **For motivation strengthening**: Find papers showing cognitive features improve duplicate detection (not just readability). Find papers showing sequence alignment improves text similarity.
3. **For empirical validation**: Implement RVP and compare against MinHash, WMD, edit distances on short text datasets (Quora Question Pairs, MSRP).

## Conclusion

The RVP approach appears to represent a **novel combination** of existing components: cognitive features (word length, frequency, POS) represented AS SEQUENCES (not aggregated) with DTW alignment for near-duplicate short text detection. While individual components exist, their specific combination for this task was NOT FOUND in the literature. Confidence: 70-80%.

**Key supporting evidence**:
- No paper found using cognitive features as sequences for similarity [Searches 1-15]
- No paper found applying DTW to cognitive feature sequences [Searches 5, 11, 12]
- 'Reading Vector Patterns' term does not exist in literature [Search 10]

**Caveats**:
- Novelty is primarily in combination and application, not individual components
- Need more thorough search to increase confidence
- Should compare against simple baselines (cosine of cognitive feature vectors, edit distance on POS sequences) to validate DTW adds value

## Sources

[1] [On the resemblance and containment of documents](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf) — Broder (1997) introduces MinHash for estimating set resemblance and containment. Documents represented as sets of shingles. Set-based, not sequence-based. Limitation for short text: sparse shingle sets.

[2] [From Word Embeddings To Document Distances](https://proceedings.mlr.press/v37/kusnerb15.html) — Kusner et al. (2015) introduces Word Mover's Distance (WMD) using word2vec embeddings and Earth Mover's Distance. O(n^2) complexity. Acknowledges challenges with short documents. Not using DTW or cognitive features.

[3] [Paraphrase Identification as Probabilistic Quasi-Synchronous Recognition](http://www.cs.cmu.edu/~nasmith/papers/das+smith.acl09.pdf) — Das & Smith (2009) uses quasi-synchronous dependency grammars for paraphrase identification. Uses POS and syntax but not as simple sequences. Not using cognitive load indicators or DTW.

[4] [On Improving the Accuracy of Readability Classification using Insights from Second Language Acquisition](https://aclanthology.org/W12-2019.pdf) — Vajjala & Meurers (2012) uses word length, frequency, POS for readability assessment. But features are aggregated per document (not sequences). Classification task, not similarity. Shows cognitive features capture unique information.

[5] [String distances for near-duplicate detection](https://nlp.unibuc.ro/papers/danaila12.pdf) — Danaila et al. (2012) compares Rank distance and Smith-Waterman for near-duplicate detection. Uses sequence alignment but on characters/words, not cognitive features. Shows short text presents challenges for traditional methods.

## Follow-up Questions

- Does representing cognitive features (word length, frequency, POS) as sequences (preserving order) instead of aggregated features improve text similarity/distance measurement for short text?
- Does DTW alignment of cognitive feature sequences capture similarity that is missed by lexical overlap (edit distance) and embedding-based methods (WMD)?
- On what datasets and tasks does the RVP approach outperform or underperform existing methods (MinHash, WMD, edit distances) for near-duplicate short text detection?

---
*Generated by AI Inventor Pipeline*
