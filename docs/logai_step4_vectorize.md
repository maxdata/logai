# Step 4: Vectorize parsed or raw loglines into numeric embeddings

Convert text into numerical representations for downstream ML tasks.

## Key types
- `logai.information_extraction.log_vectorizer.VectorizerConfig`
- `logai.information_extraction.log_vectorizer.LogVectorizer`

## Configuration
```python
from logai.information_extraction.log_vectorizer import VectorizerConfig

cfg = VectorizerConfig.from_dict({
    "algo_name": "tfidf",  # options: tfidf | word2vec | fasttext | logbert | forecast_nn
    "algo_param": {},
})
```

## Usage
```python
from logai.information_extraction.log_vectorizer import LogVectorizer

vec = LogVectorizer(cfg)
vec.fit(parsed_df["parsed_logline"])  # or cleaned_series / raw logline
emb_series = vec.transform(parsed_df["parsed_logline"])  # Series of vectors
```

## Ordering: vectorize before or after grouping?
- Vectorize → Group (default for event-level features): produce one vector per line, then aggregate within each event (mean by default) via `FeatureExtractor.convert_to_feature_vector`. Yields one fixed-length vector per event.
- Group → Vectorize (session/sequences): create sequences/windows first (e.g., via `FeatureExtractor.convert_to_sequence`) and then vectorize the concatenated text or pooled per-line vectors.
- Counter-based time-series AD (`convert_to_counter_vector`) doesn’t need text vectors at all.

## Notes
- Traditional features (TF-IDF) are fast and non-neural; Word2Vec/FastText require training or pre-trained models.
- LogBERT/NN models require optional deep-learning extras (`pip install "logai[deep-learning]"`).
- Vectorized Series can be padded and framed into a DataFrame via `FeatureExtractor` for clustering or anomaly detection.
