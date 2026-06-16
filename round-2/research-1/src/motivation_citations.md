# Motivation Citations for RVP Paper

## 1. Cognitive Features Capture Unique Information

**Citation**: Vajjala & Meurers (2012) - "On Improving the Accuracy of Readability Classification using Insights from Second Language Acquisition" [4]

**Quote**: "We show that the developmental measures from Second Language Acquisition (SLA) research when combined with traditional readability features such as word length and sentence length provide a good indication of text readability across different grades. The resulting classifiers significantly outperform the previous approaches on readability classification, reaching a classification accuracy of 93.3%."

**How it Supports RVP**: Shows that cognitive features (word length, frequency, POS) capture information about text that is complementary to traditional features. Supports using cognitive load indicators for text representation.

## 2. Existing Methods Fail for Short Text

**Citation**: Danaila et al. (2012) - "String distances for near-duplicate detection" [5]

**Quote**: "For short texts such as text messages, [9] indicated the fundamental differences that must be taken into account when doing term weighting, for example. For short messages, larger differences need to be tolerated, and as much semantic information needs to be taken into account."

**How it Supports RVP**: Supports the claim that short text presents unique challenges for duplicate detection. Traditional methods (like MinHash) that work well for long documents may not work for short text.

**Additional Citation**: Kusner et al. (2015) - WMD paper [2]

**Quote**: "An additional example D3 highlights the flow when the number of words does not match. D3 has term weights dj=0.33 and excess flow is sent to other similar words. This increases the distance, although the effect might be artificially magnified due to the short document lengths as longer documents may contain several similar words."

**How it Supports RVP**: WMD (a state-of-the-art method) acknowledges challenges with short documents. Supports need for methods specifically designed for short text.

## 3. Sequence Information Matters for Similarity

**Citation**: Das & Smith (2009) - "Paraphrase Identification as Probabilistic Quasi-Synchronous Recognition" [3]

**Quote**: "Although paraphrase identification is defined in semantic terms, it is usually solved using statistical classifiers based on shallow lexical, n-gram, and syntactic 'overlap' features. Such overlap features give the best-published classification accuracy for the paraphrase identification task, but do not explicitly model correspondence structure (or 'alignment') between the parts of two sentences."

**How it Supports RVP**: Supports the importance of modeling alignment/sequence structure between texts for similarity tasks. RVP uses DTW to align sequences; this paper uses quasi-synchronous grammars for alignment.

## 4. Limitations of Set-Based Methods (MinHash) for Short Text

**Citation**: Broder (1997) - "On the resemblance and containment of documents" [1]

**Quote**: "The resemblance has the additional property that d(A, B) = 1 − r(A, B), is a metric (obeys the triangle inequality), which is useful for the design of algorithms intended to cluster a collection of documents into sets of closely resembling documents."

**How it Supports RVP (Contradicting Evidence)**: MinHash is designed for long documents (30M web documents in evaluation). The shingle sets for short text may be too sparse for meaningful similarity estimation. RVP addresses this by using cognitive features that provide signal even for short text.

## 5. Need for Interpretable Methods

**Citation**: Kusner et al. (2015) [2]

**Quote**: "The WMD distance has several intriguing properties: 1. it is hyper-parameter free and straight-forward to understand and use; 2. it is highly interpretable as the distance between two documents can be broken down and explained as the sparse distances between few individual words"

**How it Supports RVP**: Even state-of-the-art methods (WMD) emphasize interpretability. RVP is even more interpretable: cognitive features (word length, frequency, POS) are directly meaningful to humans.

## 6. Cognitive Load Affects Reading Comprehension

**Citation**: Cognitive Load Theory (not directly searched, but relevant)

**Note**: Need to find papers on cognitive load and text processing. Search for "cognitive load text comprehension" or "reading effort word frequency".

**Proposed Citation**: Search for papers connecting word frequency/length to reading effort.

**Search Query**: "word frequency effect reading effort eye-tracking"

**Expected Finding**: Word frequency and length affect reading time (eye-tracking studies). Supports using these as cognitive load indicators.

## Gaps in Motivation Citations

1. **Direct evidence that cognitive features help for NEAR-DUPLICATE detection**: Not found yet. Need to find papers showing that cognitive features improve duplicate detection.

2. **Evidence that sequence alignment (DTW) helps for text**: Not found yet. Need to find papers showing sequence alignment improves text similarity.

3. **Evidence that MinHash/LSH fail for short text**: Implied but not directly shown. Need to find papers demonstrating this failure.

## Next Steps for Strengthening Motivation

1. Search for "MinHash short text failure" or "limitations of MinHash for short documents"
2. Search for "cognitive features duplicate detection" or "readability near-duplicate"
3. Search for "sequence alignment text similarity improvement" or "DTW NLP application"
