# Novelty Assessment for Reading Vector Patterns (RVP)

## Aspect-by-Aspect Novelty

| Aspect | Existing Work | Novel in RVP? |
|--------|--------------|---------------|
| Word length as feature | Many papers (readability, text classification) [4] | No - individual cognitive features are well-known |
| Word frequency as feature | Many papers (readability, information retrieval) [4] | No - word frequency is a standard feature |
| POS as feature | Many papers (POS n-grams, syntactic analysis) [3] | No - POS tags are commonly used |
| Sequence (not bag) representation of cognitive features | Some papers use sequences for syntax (Das & Smith 2009) [3] | **Partial** - using cognitive features AS SEQUENCES (not aggregated) appears novel |
| DTW for alignment of cognitive feature sequences | DTW used for speech, time series; NOT for cognitive feature sequences [Search result: only speech/audio applications found] | **Likely YES** - no evidence of DTW applied to cognitive feature sequences |
| Cognitive features + DTW for text similarity | No papers found combining these | **Likely YES** - this specific combination appears absent from literature |
| Applied to duplicate detection | Many methods (MinHash, WMD, edit distances) [1, 2, 5] | **Partial** - DTW-based approach for duplicate detection is uncommon |
| Focus on short text | Some methods address short text challenges [5] | **Partial** - explicit focus on short text with cognitive features is less common |

## Closest Existing Methods

1. **Word Mover's Distance (Kusner et al., 2015)** [2]:
   - **Why close?** Also uses a distance metric that aligns words between documents
   - **What's different?** WMD uses embeddings and optimal transport (Earth Mover's Distance); RVP uses cognitive features and DTW. WMD requires pre-trained embeddings; RVP uses simple, interpretable cognitive features.

2. **Readability Classification (Vajjala & Meurers, 2012)** [4]:
   - **Why close?** Uses word length, frequency, and POS as features
   - **What's different?** Features are aggregated per document (bag-of-words style); RVP represents them as sequences preserving order. Classification vs. similarity/distance task.

3. **Quasi-Synchronous Grammar for Paraphrase (Das & Smith, 2009)** [3]:
   - **Why close?** Uses syntax (POS, dependency trees) and sequence alignment
   - **What's different?** Complex grammatical model; RVP uses simple cognitive features (length, frequency, POS) as numerical sequences.

4. **Smith-Waterman for Near-Duplicate Detection (Danaila et al., 2012)** [5]:
   - **Why close?** Uses sequence alignment for duplicate detection
   - **What's different?** Aligns character or word sequences directly; RVP aligns cognitive feature sequences (word length, frequency, POS sequences).

## Specific Novelty Claims Verification

- [x] Any paper uses word length + frequency + POS TOGETHER? **YES** - Vajjala & Meurers (2012) [4]
- [x] Any paper uses these as a SEQUENCE (not aggregated)? **NO** - not found in literature (Vajjala aggregates features)
- [x] Any paper uses DTW on cognitive feature sequences? **NO** - not found (DTW used for speech, time series, but not cognitive features for text)
- [ ] Any paper applies this to DUPLICATE DETECTION? **Unknown** - need more search (closest: Danaila et al. use Smith-Waterman for duplicates, but not with cognitive features)
- [x] Any paper focuses on SHORT text? **Partial** - WMD evaluated on short text; Das & Smith on sentences

## Novelty Verdict

**PARTIAL NOVELTY** - The RVP approach appears to combine existing components in a novel way:

1. **Novel combination**: Using word length, frequency, and POS together AS SEQUENCES (preserving order) with DTW alignment for near-duplicate detection has not been found in the literature.

2. **Novel representation**: Representing text as "reading vectors" (sequences of cognitive load indicators) is a unique way to capture text similarity that is complementary to lexical and embedding-based methods.

3. **Novel alignment method for this domain**: Applying DTW to align cognitive feature sequences (rather than acoustic features or time series) appears to be a new application.

**Confidence Level**: MEDIUM-HIGH (70-80% confident in novelty)

**Caveats**:
- Limited search depth - may have missed papers combining these elements
- Cognitive features are well-established individually
- DTW is a known algorithm (just not applied to this specific problem)
- The novelty is primarily in the COMBINATION and APPLICATION, not in individual components

## Recommendations for Strengthening Novelty Claims

1. Conduct more thorough search in ACL Anthology, EMNLP, SIGIR proceedings
2. Search for "cognitive features" + "sequence alignment" more broadly
3. Verify that DTW has not been applied to ANY numerical text representations (beyond speech)
4. Compare against simple baselines (cosine of cognitive feature vectors, edit distance on POS sequences) to show DTW adds value
