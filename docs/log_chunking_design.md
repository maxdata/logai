## Log chunking and sessionization design

This document explains how LogAI partitions raw logs into sequences using sliding windows, time/category grouping, and session windows. It covers the primary components, algorithms, configuration, and example usage.

### Goals

- Transform per-line logs into sequences for downstream tasks (e.g., anomaly detection, forecasting, representation learning).
- Support multiple grouping strategies: by time, by categorical attributes, or by session ids.
- Provide consistent outputs via `LogRecordObject` for easy pipeline composition.

## Data model

- `logai.dataloader.data_model.LogRecordObject` holds the log data:
  - Body: `body["logline"]` (unstructured text)
  - Timestamp: `timestamp["timestamp"]`
  - Attributes: `attributes[...]` (structured fields)
  - Session: `span_id["span_id"]`
  - Labels: `labels["labels"]`
  - Methods: `to_dataframe()` merges available fields; used by partitioners.

## Components

### Partitioner (`logai.preprocess.partitioner.Partitioner`)

Responsible for sliding-window partitioning and grouping by time and/or categorical attributes.

- Config (`PartitionerConfig`):
  - `group_by_category: list | None` — attribute columns to group by (e.g., `"Level"`, `"host"`).
  - `group_by_time: str | None` — Pandas offset alias (e.g., `"15s"`, `"1min"`) to bucket timestamps.
  - `sliding_window: int` — fixed window size. `<= 0` means no windowing (returns original lines or joined group).
  - `sep_token: str` — delimiter to join lines (default `"[SEP]"`).
  - `exclude_last_window: bool` — drops the trailing partial window by using `closed="left"` semantics.
  - `exclude_smaller_windows: bool` — filters windows shorter than `sliding_window`.

- Key methods:
  - `sliding_window(loglines: pd.Series) -> pd.Series`
  - `group_counter(df: pd.DataFrame) -> pd.DataFrame` — group, then count rows as `counts`.
  - `group_sliding_window(df: pd.DataFrame, logline_col_name="logline") -> pd.DataFrame`

- Implementation details:
  - Time grouping uses `pd.Grouper` with `freq` and `label="left"`.
  - Sliding windows use `Series.rolling(window, min_periods=window, closed=...)` and then join with `sep_token`.

### OpenSetPartitioner (`logai.preprocess.openset_partitioner.OpenSetPartitioner`)

High-level wrapper specialized for open datasets. It supports two strategies:

1) Sliding windows per session id
   - Internally constructs a `Partitioner` with `group_by_category=[span_id]` and applies `group_sliding_window`.
   - Adds:
     - `attributes["next_logline"]`: next logline aligned with each window (for sequence prediction tasks).
     - `labels["labels"]`: aggregated binary label per window — positive if any element in the window or its next element is positive.

2) Session windows (sessionization)
   - Uses `FeatureExtractor` to group by session id and build one sequence per session by joining lines with `logsequence_delim`.
   - Aggregates labels over the session (positive if any element is positive).

- Config (`OpenSetPartitionerConfig`):
  - `sliding_window: int` — if `> 0`, enable sliding windows per session id (excludes last/smaller windows by default).
  - `session_window: bool` — if `True` and `sliding_window == 0`, build one sequence per session.
  - `logsequence_delim: str` — delimiter for sequence joins (default `"[SEP]"`).

### FeatureExtractor (`logai.information_extraction.feature_extractor.FeatureExtractor`)

Groups logs by attributes/time buckets and converts to:

- Counter vectors (`convert_to_counter_vector`): counts per group.
- Feature vectors (`convert_to_feature_vector`): aggregates numeric features per group and returns event indices.
- Sequences (`convert_to_sequence`):
  - If `sliding_window > 0`, uses numpy stride tricks to create overlapping fixed-length subsequences with step size `steps`.
  - If `sliding_window == 0`, concatenates one sequence per group.
  - Maintains `event_index` per sequence; can carry timestamp lists if time is not a grouping factor.

## Dataset-specific preprocessors

Open-set preprocessors derive `span_id` and `labels` before partitioning:

- `HDFSPreprocessor`
  - Builds serial ids from block ids in the log body, preserving a mapping to original block ids.
  - Computes labels by intersecting each line’s block-id set with the anomaly labels CSV.

- `BGLPreprocessor`
  - Constructs session ids by bucketing timestamps into fixed intervals (e.g., minute buckets).
  - Derives labels from dataset-specific label column.

## Outputs and fields

- Sliding-window outputs (`generate_sliding_window`):
  - `body["logline"]`: concatenated sequences.
  - `attributes["next_logline"]`: next-element target aligned with each window.
  - `labels["labels"]`: window-level binary labels.
  - `span_id["span_id"]`: group/session id for each sequence.

- Session-window outputs (`generate_session_window`):
  - `body["logline"]`: one joined sequence per session.
  - `labels["labels"]`: aggregated session label.
  - `span_id["span_id"]`: session id.

## Usage examples

### Group by time bucket and Level, window length 2

```python
from logai.preprocess.partitioner import Partitioner, PartitionerConfig
from logai.utils import constants

cfg = PartitionerConfig(
    group_by_time="15s",
    group_by_category=["Level"],
    sliding_window=2,
    sep_token="[SEP]",
    exclude_last_window=True,
    exclude_smaller_windows=True,
)
p = Partitioner(cfg)
df = logrecord.to_dataframe()
sequences_df = p.group_sliding_window(df, constants.LOGLINE_NAME)
```

### Per-session sliding windows (size 5) with next-logline

```python
from logai.preprocess.openset_partitioner import OpenSetPartitioner, OpenSetPartitionerConfig

cfg = OpenSetPartitionerConfig(sliding_window=5, session_window=False)
osp = OpenSetPartitioner(cfg)
seq_logrecord = osp.generate_sliding_window(logrecord)
```

### One sequence per session (sessionization)

```python
from logai.preprocess.openset_partitioner import OpenSetPartitioner, OpenSetPartitionerConfig

cfg = OpenSetPartitionerConfig(sliding_window=0, session_window=True)
osp = OpenSetPartitioner(cfg)
session_logrecord = osp.generate_session_window(logrecord)
```

## Edge cases and behavior

- `sliding_window <= 0` in `Partitioner`: returns original lines or a simple join without windowing.
- `exclude_last_window=True`: suppresses the trailing partial window via `closed="left"`.
- `exclude_smaller_windows=True`: filters windows with length `< sliding_window`.
- Time grouping: timestamp column is bucketed via `pd.Grouper` (Partitioner) or `dt.floor` (FeatureExtractor).

## Design choices and trade-offs

- Two windowing paths are supported:
  - `Partitioner`: Pandas-native rolling for text concatenation with flexible grouping.
  - `FeatureExtractor`: stride-based subsequences with explicit `steps`, retaining event indices (useful for modeling).
- Open-set wrapper centralizes common patterns (per-session windows, next-element labeling) while preserving configurability.

## Extensibility

- Additional grouping keys can be added via `group_by_category` without changing algorithms.
- Alternative window semantics (e.g., tumbling windows, custom step sizes) can be introduced by extending `Partitioner`.
- Richer label aggregation strategies can be added in `OpenSetPartitioner`.

## Relevant files

- `logai/preprocess/partitioner.py`
- `logai/preprocess/openset_partitioner.py`
- `logai/information_extraction/feature_extractor.py`
- `logai/preprocess/hdfs_preprocessor.py`
- `logai/preprocess/bgl_preprocessor.py`


