# Step 5: Group logs into events and build indices, counters, and features

Aggregate lines by attributes/time/windows to form event-level views for counting and feature creation.

## Key types
- `logai.information_extraction.feature_extractor.FeatureExtractorConfig`
- `logai.information_extraction.feature_extractor.FeatureExtractor`

## Configuration
```python
from logai.information_extraction.feature_extractor import FeatureExtractorConfig

cfg = FeatureExtractorConfig(
    group_by_category=["Level"],     # any columns present in attributes/body
    group_by_time="1s",             # pandas offset alias, e.g. "1s", "1min"
    sliding_window=0,                # 0 -> no windowing
    steps=1,
    max_feature_len=100,
)
```

## Usage
```python
from logai.information_extraction.feature_extractor import FeatureExtractor
from logai.utils import constants

fe = FeatureExtractor(cfg)
# Counter vectors (per-group counts with event_index lists)
counter_df = fe.convert_to_counter_vector(
    timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS],
    attributes=logrecord.attributes.astype(str),
)
# Feature vectors (aggregate/padded vectors with event_index lists)
index_df, feature_df = fe.convert_to_feature_vector(
    log_vectors=emb_series,
    attributes=logrecord.attributes.astype(str),
    timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS],
)
```

## How “chunk per vector” works
- Input is a Series of per-line vectors (`emb_series`) aligned to the original DataFrame index.
- Vectors are padded to `max_feature_len` and expanded into columns.
- Attributes/timestamps are joined, then grouped by `group_by_time` and/or `group_by_category`.
- Per-group aggregation (mean by default) yields one vector per chunk/event.
- An `event_index` column holds the list of original line indices belonging to the chunk for back-mapping.

## Sequences (optional)
Use `convert_to_sequence` to build fixed-length chunks of consecutive lines via sliding windows. Each sequence carries its `event_index` window and, if needed, you can pool per-line vectors within each window to obtain a single sequence vector.

## Notes
- When `group_by_time` is set, timestamps are floored before grouping to form time buckets.
- For time-series AD, prefer `convert_to_counter_vector`; for semantic clustering/AD, prefer `convert_to_feature_vector`.
