# Step 6: Feed features into LogAI applications

Use event- or line-level features with clustering, anomaly detection, or summarization workflows.

## Clustering
```python
from logai.applications.application_interfaces import WorkFlowConfig
from logai.applications.log_clustering import LogClustering

config_dict = {
    "data_loader_config": { ... },
    "preprocessor_config": { ... },
    "log_parser_config": { "parsing_algorithm": "drain" },
    "log_vectorizer_config": { "algo_name": "tfidf" },
    "categorical_encoder_config": { "name": "label_encoder" },
    "clustering_config": { "algo_name": "kmeans", "algo_params": {"n_clusters": 8} },
}
wf = WorkFlowConfig.from_dict(config_dict)
app = LogClustering(wf)
app.execute()
clusters = app.logline_with_clusters  # includes cluster_id per line
```

## Anomaly detection (classical)
```python
from logai.applications.application_interfaces import WorkFlowConfig
from logai.applications.log_anomaly_detection import LogAnomalyDetection

config_dict = {
    "data_loader_config": { ... },
    "preprocessor_config": { ... },
    "log_parser_config": { "parsing_algorithm": "drain" },
    "log_vectorizer_config": { "algo_name": "word2vec" },
    "categorical_encoder_config": { "name": "label_encoder" },
    "feature_extractor_config": { "group_by_category": ["Level"], "group_by_time": "1s" },
    "anomaly_detection_config": { "algo_name": "one_class_svm" },  # or counter-based ETS/DBL
}
wf = WorkFlowConfig.from_dict(config_dict)
app = LogAnomalyDetection(wf)
app.execute()
anomalies = app.anomaly_results
```

## Summarization (template-based)
- Use `LogParser` directly to mine templates and group by `parsed_logline`.
- Or run the GUI (`python gui/application.py`) for interactive summarization.

## Notes
- Some deep-learning models require optional installs: `pip install "logai[deep-learning]"`.
- Counter-based time-series detection (ETS/DBL) doesn’t require text embeddings; it uses `FeatureExtractor.convert_to_counter_vector`.
