# Search Log

## Search 1: Verify Hypothesis References - Broder 1997 (MinHash)
- **Date**: 2025-01-01
- **Query**: "On the resemblance and containment of documents Broder 1997 MinHash"
- **Results**: 5 papers found
- **Relevant**: 1 paper (Broder 1997)
- **URLs**: 
  - https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf (fetched)
- **Findings**: Confirmed Broder 1997 uses set-based resemblance (shingles), not sequence-based. MinHash is for long documents. Limitation for short text: shingle sets may be too sparse.

## Search 2: Verify Hypothesis References - Kusner 2015 (WMD)
- **Date**: 2025-01-01
- **Query**: "Word Mover's Distance Kusner 2015 arXiv"
- **Results**: 5 papers found
- **Relevant**: 1 paper (Kusner et al. 2015)
- **URLs**:
  - https://proceedings.mlr.press/v37/kusnerb15.html (fetched)
  - http://proceedings.mlr.press/v37/kusnerb15.pdf (fetched)
- **Findings**: Confirmed WMD uses word embeddings and Earth Mover's Distance (optimal transport). O(n²) complexity. Requires pre-trained embeddings. Not using DTW. Acknowledges challenges with short documents.

## Search 3: Verify Hypothesis References - Das & Smith 2009
- **Date**: 2025-01-01
- **Query**: "POS n-gram paraphrase Das Smith"
- **Results**: 5 papers found
- **Relevant**: 1 paper (Das & Smith 2009)
- **URLs**:
  - http://www.cs.cmu.edu/~nasmith/papers/das+smith.acl09.pdf (fetched)
- **Findings**: Paper is about quasi-synchronous dependency grammars for paraphrase identification, not simple POS n-grams. Uses syntax and WordNet. Not using cognitive load indicators (word length, frequency). Not using DTW.

## Search 4: Cognitive Features for Text Similarity
- **Date**: 2025-01-01
- **Query**: "cognitive features text similarity psycholinguistic"
- **Results**: 5 papers found
- **Relevant**: 0 papers directly relevant
- **URLs**: None fetched
- **Findings**: Search returned general psycholinguistics references, not specific to text similarity. Need different search terms.

## Search 5: DTW Applied to Text (Not Speech)
- **Date**: 2025-01-01
- **Query**: "Dynamic Time Warping text document similarity not speech"
- **Results**: 5 papers found
- **Relevant**: 1 paper (ResearchGate - DTW for string similarity)
- **URLs**:
  - https://www.researchgate.net/publication/374617740_An_Investigation_of_the_Usage_of_Dynamic_Time_Warping_in_String_Similarity_Estimation (fetch failed - HTTP 403)
- **Findings**: DTW is primarily used for speech/audio. Limited application to text. The ResearchGate paper might be relevant but couldn't access.

## Search 6: POS n-gram for Text Similarity
- **Date**: 2025-01-01
- **Query**: "POS n-gram text similarity duplicate detection"
- **Results**: 5 papers found
- **Relevant**: 1 paper (Sciencedirect - POS tag n-grams for plagiarism detection)
- **URLs**:
  - https://www.sciencedirect.com/science/article/abs/pii/S0957417422001610 (fetch failed - HTTP 403)
- **Findings**: POS n-grams used for plagiarism detection. Not using cognitive features as sequences. Not using DTW.

## Search 7: Cognitive Load Features for Text Similarity
- **Date**: 2025-01-01
- **Query**: "cognitive load word length frequency text similarity"
- **Results**: 5 papers found
- **Relevant**: 0 papers directly relevant
- **URLs**: None fetched
- **Findings**: Search returned cognitive psychology references, not NLP applications. Word length and frequency effects are well-studied in psycholinguistics but not for text similarity.

## Search 8: Near-Duplicate Short Text Detection Challenges
- **Date**: 2025-01-01
- **Query**: "near-duplicate short text detection challenges MinHash limitations"
- **Results**: 5 papers found
- **Relevant**: 2 papers (Sectional MinHash, Near-Duplicate Detection survey)
- **URLs**:
  - https://www.sciencedirect.com/science/article/abs/pii/S0957417418300149 (not fetched)
  - https://nlp.unibuc.ro/papers/danaila12.pdf (fetched)
- **Findings**: Short text presents challenges for traditional methods. Danaila et al. (2012) uses Smith-Waterman for near-duplicate detection. Confirms need for methods tailored to short text.

## Search 9: Readability Features for Text Classification
- **Date**: 2025-01-01
- **Query**: "cognitive readability features SVM text classification"
- **Results**: 5 papers found
- **Relevant**: 1 paper (Vajjala & Meurers 2012)
- **URLs**:
  - https://aclanthology.org/W12-2019.pdf (fetched)
- **Findings**: Readability assessment uses word length, frequency, POS as features. But features are aggregated (bag-of-words style), not represented as sequences. Supports using cognitive features but not sequence representation.

## Search 10: Check if "Reading Vector Patterns" Already Exists
- **Date**: 2025-01-01
- **Query**: "\"reading vector\" text similarity"
- **Results**: 5 results found
- **Relevant**: 0 papers (all about vector databases, not "reading vectors" as cognitive features)
- **URLs**: None relevant
- **Findings**: "Reading Vector Patterns" term does not appear to exist in literature. Supports novelty of RVP name.

## Search 11: DTW for NLP (Not Speech)
- **Date**: 2025-01-01
- **Query**: "\"Dynamic Time Warping\" NLP text similarity not speech"
- **Results**: 5 papers found
- **Relevant**: 0 papers directly relevant
- **URLs**: None fetched
- **Findings**: DTW in NLP context is rare. Primarily used for speech processing. No evidence of DTW applied to cognitive feature sequences.

## Search 12: Sequence Alignment of Cognitive Features in NLP
- **Date**: 2025-01-01
- **Query**: "sequence alignment cognitive features text NLP"
- **Results**: 5 papers found
- **Relevant**: 0 papers directly relevant
- **URLs**: None fetched
- **Findings**: No papers found combining sequence alignment with cognitive features for text similarity.

## Summary of Search Results

**Total Searches**: 12
**Papers Found**: ~15 unique papers
**Papers Fetched**: 5 (Broder 1997, Kusner 2015, Das & Smith 2009, Vajjala & Meurers 2012, Danaila et al. 2012)
**Relevant Papers**: 5 (all fetched)

**Key Findings**:
1. No paper found using cognitive features (word length, frequency, POS) AS SEQUENCES with DTW for text similarity
2. DTW is not commonly applied to text (primarily speech/audio)
3. Cognitive features are used for readability/classification but as aggregated features (not sequences)
4. Near-duplicate detection for short text is challenging for traditional methods (MinHash, WMD)

**Confidence in Novelty**: MEDIUM-HIGH (70-80%)

**Limitations of Search**:
1. Limited access to some papers (HTTP 403 errors)
2. Search primarily in English literature
3. May have missed papers using different terminology
4. No systematic search of ACL Anthology, EMNLP, SIGIR proceedings

**Recommended Additional Searches**:
1. Search ACL Anthology directly: "cognitive features" "text similarity"
2. Search for "DTW" + "NLP" or "DTW" + "text classification"
3. Search for "reading ease" + "duplicate detection"
4. Search for "word length sequence" + "similarity"
