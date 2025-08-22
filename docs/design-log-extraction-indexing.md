## Log Extraction and Indexing Design

### Purpose
This document explains how LogAI ingests raw logs, extracts structure, and builds vectorized/indexable representations for downstream analytics (clustering, anomaly detection, summarization).

### Scope
- Extraction: loading, preprocessing, and pattern parsing.
- Indexing: vectorization, grouping into event-level indices, and feature construction.
- Configuration and extensibility notes.

## High-level Architecture
1. Load raw logs into a unified data model (`LogRecordObject`).
2. Preprocess and normalize text (optional custom regex-based cleaning).
3. Parse unstructured loglines into templates/patterns and extract dynamic parameters.
4. Vectorize parsed loglines (or raw/cleaned text) into numeric embeddings.
5. Group logs into events (by attributes/time/windows) to create event-level indices, counters, and features.
6. Feed event- or line-level features into apps (clustering, anomaly detection, summarization).

Key modules:
- `logai.dataloader`: ingestion and data model.
- `logai.preprocess`: text cleaning, grouping helpers.
- `logai.information_extraction`: parsing, vectorization, feature extraction.
- `logai.algorithms.*`: algorithm implementations selected via a factory.
- `logai.applications.*`: end-to-end workflows stitching the above.

## Data Model
File: `logai/dataloader/data_model.py`

- `LogRecordObject`
  - Fields (all are `pd.DataFrame` unless noted): `timestamp`, `attributes`, `resource`, `trace_id`, `span_id`, `severity_text`, `severity_number`, `body`, `labels`; plus `_index` (computed from `body.index`).
  - `body[constants.LOGLINE_NAME]` holds the raw/cleaned logline text.
  - `to_dataframe()` joins all non-empty fields to a single `DataFrame` aligned by index.

Constants used across the pipeline (`logai.utils.constants`):
- `LOGLINE_NAME`, `PARSED_LOGLINE_NAME`, `PARAMETER_LIST_NAME`, `LOG_TIMESTAMPS`, `LOG_COUNTS`, etc.

## Extraction Pipeline

### 1) Data Loading
File: `logai/dataloader/data_loader.py`

- `FileDataLoader(DataLoaderConfig)` supports `csv`, `tsv`, `json`, and free-form text like `.log`.
- For free-form logs, pass `reader_args.log_format` using angle-bracketed fields (e.g., `<timestamp> <level> <msg>`). The loader compiles this into a regex and extracts named columns.
- `dimensions` map input columns to `LogRecordObject` fields. If omitted, all columns are concatenated into `body[LOGLINE_NAME]`.
- Timestamp parsing: if `infer_datetime=True`, converts `timestamp` to datetime using `datetime_format`.

Output: a `LogRecordObject` whose `body` contains the loglines and optional structured fields.

### 2) Preprocessing
File: `logai/preprocess/preprocessor.py`

- `Preprocessor(PreprocessorConfig)`
  - `custom_delimiters_regex`: list/dict of regex patterns replaced with spaces.
  - `custom_replace_list`: list of `(pattern, replacement)` pairs applied to loglines; also returns a `terms` table of extracted matches.
  - `clean_log(loglines)` returns `(cleaned_loglines, terms)`.
  - `group_log_index(attributes, by=[...])` groups attributes and returns a `DataFrame` mapping group keys to arrays of original indices.

### 3) Parsing (Template Extraction and Parameters)
File: `logai/information_extraction/log_parser.py`

- `LogParser(LogParserConfig)` dispatches to a parsing algorithm via `logai.algorithms.factory`.
- Supported algorithms (files under `logai/algorithms/parsing_algo/`):
  - `Drain`: tree-based online parser.
  - `AEL`: frequent pattern mining.
  - `IPLoM`: partition-based parser.
- `parse(loglines)` returns a `DataFrame` with columns:
  - `LOGLINE_NAME`: original (cleaned) text
  - `PARSED_LOGLINE_NAME`: template with wildcards
  - `PARAMETER_LIST_NAME`: list of dynamic tokens not part of the template
- Models can be saved/loaded (`save`, `load`).

## Indexing and Feature Construction

Indexing in LogAI is realized through two complementary layers:
1) Line-level numeric index (the DataFrame index, preserved throughout).
2) Event-level grouping indices produced by `FeatureExtractor`, enabling retrieval/aggregation per event/session/time window.

### 4) Vectorization (Embeddings)
File: `logai/information_extraction/log_vectorizer.py`

- `LogVectorizer(VectorizerConfig)` selects a vectorizer via the factory:
  - Statistical: `tfidf` (`logai/algorithms/vectorization_algo/tfidf.py`)
  - Neural: `word2vec`, `fasttext`, `logbert` (see `logai/algorithms/vectorization_algo/` and `logai/algorithms/nn_model/`)
- `fit(loglines)` trains the model; `transform(loglines)` returns a `pd.Series` of `numpy.ndarray` vectors aligned to input indices.
- Example: TF-IDF builds a vocabulary and returns dense vectors per logline; vocabulary size defines the embedding dimension.

### 5) Event Grouping and Feature Extraction
File: `logai/information_extraction/feature_extractor.py`

- `FeatureExtractor(FeatureExtractorConfig)` combines vectors, attributes, and timestamps to produce event-level indices and features.
- Grouping controls:
  - `group_by_category`: list of attribute columns (e.g., host, service, severity).
  - `group_by_time`: resample by time frequency (e.g., "1min").
  - `sliding_window` and `steps`: windowed sequence construction (sequence mode).
- Outputs and modes:
  - `convert_to_feature_vector(log_vectors, attributes, timestamps)` returns:
    - `event_index_list`: `DataFrame` keyed by group columns with `event_index` as a list of original line indices (line-level index → event-level index mapping).
    - `block_list`: aggregated numeric features per event (mean over vectors by default). Feature columns are padded and named `feature_0..N`.
  - `convert_to_counter_vector(log_pattern, attributes, timestamps)` returns group counts and `event_index` for each group.
  - `convert_to_sequence(...)` builds sliding windows over indices and concatenated text sequences, yielding event windows and their `event_index` lists.

These structures constitute the in-repo indexing: each event row retains the list of original line indices for retrieval, and the derived features serve as compact representations for ML tasks.

## Application Workflows (End-to-End)

### Auto Log Summarization
File: `logai/applications/auto_log_summarization.py`
- Load → Preprocess → Parse → Join attributes/timestamps → Expose parsed results.
- Notes: The docstring mentions storing an index; the implementation maintains DataFrame indices and parsed results for search/filtering. No external search backend is included.

### Log Clustering
File: `logai/applications/log_clustering.py`
- Load → Preprocess → Parse → Vectorize → Pad vectors → Join encoded attributes → Cluster.
- Produces `cluster_id` per line; indices preserved for back-references.

### Log Anomaly Detection
File: `logai/applications/log_anomaly_detection.py`
- Load → Preprocess → Parse.
- Two paths: counter-based (event counts) or vector-based features via `FeatureExtractor`.
- Trains detector and predicts anomaly scores; marks anomalies back on the original line-level index (`_id` column mirrors the DataFrame index).

## Configuration
Top-level `WorkFlowConfig` (see `logai/applications/application_interfaces.py`) composes:
- `data_loader_config` (or `open_set_data_loader_config`)
- `preprocessor_config`
- `log_parser_config`
- `log_vectorizer_config`
- `feature_extractor_config`
- `clustering_config` / `anomaly_detection_config` for downstream apps

All configs use `Config.from_dict(...)` and algorithm selection is routed through `logai.algorithms.factory.factory` allowing plug-in expansion.

## Extensibility
- Add a new parsing or vectorization algorithm by:
  1) Implementing the algorithm class under `logai/algorithms/...`
  2) Defining its `Params` dataclass
  3) Registering via `@factory.register("parsing"|"vectorization", "algo_name", ParamsClass)`
  4) Refer to it in the corresponding config (`parsing_algorithm` or `algo_name`).

## Outputs and “Index” Summary
- Line-level: pandas index preserved across transformations (`_id` when materialized to a column).
- Event-level: `event_index` lists mapping grouped events back to the original line indices.
- Features: padded numeric vectors (`feature_*`) from vectorization/aggregation; counters (`LOG_COUNTS`); sequences (sliding windows).
- No external search index (e.g., Elastic, FAISS) is bundled; the produced vectors and indices are suitable for exporting to an external index if needed.

## Minimal Usage Example (Conceptual)
```python
from logai.applications.application_interfaces import WorkFlowConfig
from logai.applications.log_clustering import LogClustering

config = WorkFlowConfig.from_dict({
  "data_loader_config": {
    "filepath": "./examples/datasets/HDFS_5000.log",
    "log_type": "log",
    "reader_args": {"log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>"},
    "dimensions": {"body": ["Content"], "timestamp": ["Date"]},
    "infer_datetime": True,
    "datetime_format": "%Y-%m-%d"
  },
  "preprocessor_config": {
    "custom_delimiters_regex": [r"`+", r"\s+"],
    "custom_replace_list": [(r"(\d+)", "<NUM>")]
  },
  "log_parser_config": {"parsing_algorithm": "drain"},
  "log_vectorizer_config": {"algo_name": "tfidf", "algo_param": {"ngram_range": [1,2]}},
  "feature_extractor_config": {"group_by_time": "1min", "max_feature_len": 128},
  "clustering_config": {"algo_name": "kmeans", "algo_params": {"n_clusters": 20}}
})

app = LogClustering(config)
app.execute()
# app.logline_with_clusters, app._index let you map clusters back to original lines
```

## Notes
- All intermediate artifacts are standard pandas objects; persist them as needed.
- For semantic search or nearest-neighbor retrieval, export vectors plus line/event indices to an ANN library (e.g., FAISS) or a search engine.


