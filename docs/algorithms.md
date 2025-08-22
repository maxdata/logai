# LogAI Algorithms Catalog

This document summarizes the algorithms available in LogAI, grouped by task, with their registry names (used in configs) and brief notes.

## Parsing algorithms
- drain (`parsing`): Prefix-tree template miner; groups lines by structure, replacing variables with wildcards.
- ael (`parsing`): Alignment-based parser using token bins and merging; extracts templates.
- iplom (`parsing`): IPLoM heuristic parser for event template discovery.

## Vectorization algorithms
- tfidf (`vectorization`): TF-IDF features over tokens.
- word2vec (`vectorization`): Word2Vec embeddings aggregated to line vectors.
- fasttext (`vectorization`): FastText embeddings aggregated to line vectors.
- semantic (`vectorization`): Semantic vectorizer (wrapper for sentence/transformer-like encoders when available).
- logbert (`vectorization`): LogBERT embedding generation (requires deep-learning extras).
- forecast_nn (`vectorization`): Forecast-oriented neural features for time-series models.
- sequential (`vectorization`): Sequence vectorization of loglines (utility vectorizer).

## Clustering algorithms
- kmeans (`clustering`): K-Means clustering of features.
- dbscan (`clustering`): Density-based clustering.
- birch (`clustering`): BIRCH hierarchical clustering.

## Anomaly detection algorithms
- one_class_svm (`detection`): One-Class SVM outlier detection.
- isolation_forest (`detection`): Isolation Forest anomaly detector.
- lof (`detection`): Local Outlier Factor detector.
- distribution_divergence (`detection`): Divergence-based scoring between distributions.
- ets (`detection`): Exponential smoothing (time-series) for counter vectors.
- dbl (`detection`): Dynamic Baseline (Merlion) time-series detector.
- lstm (`detection`): LSTM-based sequence anomaly detection (DL extras).
- cnn (`detection`): CNN-based anomaly detection (DL extras).
- transformer (`detection`): Transformer-based anomaly detection (DL extras).
- logbert (`detection`): LogBERT-based anomaly detection (DL extras).

## Usage notes
- Registry names appear in configs, e.g. `LogParserConfig(parsing_algorithm="drain")`, `VectorizerConfig(algo_name="tfidf")`, `ClusteringConfig(algo_name="kmeans")`, `AnomalyDetectionConfig(algo_name="one_class_svm")`.
- Deep-learning algorithms require optional installs: `pip install "logai[deep-learning]"`.
- Time-series detectors (`ets`, `dbl`) operate on counter vectors (`FeatureExtractor.convert_to_counter_vector`). Others use feature vectors (`convert_to_feature_vector`).
