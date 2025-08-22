# LogAI Log Chunking and Sessionization - Design Document

## Overview

This document explains how LogAI partitions raw logs into sequences ("chunks") for downstream tasks such as anomaly detection, clustering, and sequence modeling. It covers sliding-window chunking, time/category grouping, and session windows.

Key modules:
- `logai/preprocess/partitioner.py`: Core grouping and sliding-window concatenation
- `logai/preprocess/openset_partitioner.py`: High-level partitioning for open datasets (HDFS, BGL), including session windows and per-session sliding windows
- `logai/information_extraction/feature_extractor.py`: Grouping by attributes/time and building sequences or feature vectors
- `logai/preprocess/hdfs_preprocessor.py`, `logai/preprocess/bgl_preprocessor.py`: Dataset-specific id/label derivation for sessions
- `logai/dataloader/data_model.py`: `LogRecordObject` schema for data interchange

## Goals

- Provide flexible, composable chunking strategies:
  - Sliding window over log lines (fixed-length windows, optional exclusion of last/short windows)
  - Time and/or category grouping before windowing
  - Session windows that create one sequence per session id
- Preserve alignment between sequences and labels, and support next-logline prediction tasks
- Work seamlessly with LogAI’s `LogRecordObject` and downstream algorithms

## Non-goals

- Domain-specific session extractors beyond HDFS/BGL examples
- Advanced labeling schemes (e.g., multi-class, weighted labels) beyond current binary reductions

## Data Model

All chunking operates on a `LogRecordObject` whose principal fields include:
- `body["logline"]`: unstructured/semi-structured raw log text per line
- `timestamp["timestamp"]`: datetime per line (optional depending on strategy)
- `span_id["span_id"]`: session key per line (dataset or preprocessor-specific)
- `labels["labels"]`: binary label per line (optional or inferred from dataset artifacts)

`LogRecordObject.to_dataframe()` yields a DataFrame that merges available fields for grouping and windowing.

## Architecture and Responsibilities

### Partitioner

Configuration (`PartitionerConfig`):
- `group_by_category: list[str] | None`
- `group_by_time: str | None` (Pandas offset alias, e.g., "15s", "1min")
- `sliding_window: int` (<= 0 disables windowing)
- `sep_token: str` (default "[SEP]")
- `exclude_last_window: bool` (drop trailing partial window)
- `exclude_smaller_windows: bool` (filter windows shorter than size)

Capabilities:
- `sliding_window(loglines: pd.Series) -> pd.Series`:
  Uses `Series.rolling(window, min_periods=window, closed="left" if exclude_last_window)` to generate fixed-length windows; each window is joined by `sep_token`.
- `group_counter(df) -> df`:
  Group by time/category and return counts per group.
- `group_sliding_window(df, logline_col_name="logline") -> df`:
  Group by time/category first; within each group, either create sliding windows or a single concatenation when `sliding_window <= 0`.

### OpenSetPartitioner

Configuration (`OpenSetPartitionerConfig`):
- `sliding_window: int` — when > 0, performs per-session sliding windows using `Partitioner` with `group_by_category=[span_id]`, and sets `exclude_last_window=True`, `exclude_smaller_windows=True`.
- `session_window: bool` — when true and `sliding_window == 0`, creates a single sequence per session.
- `logsequence_delim: str` — delimiter for sequence joins (default "[SEP]").

Per-session sliding windows (`generate_sliding_window`):
- Groups by `span_id`, builds windows of `logline`, and aligns the “next logline” for each window (used for next-step prediction tasks).
- Window labels are aggregated from per-line labels using a simple reduction: a window is labeled anomalous if any constituent line is anomalous; the label is also OR-ed with the label of the immediate next line after each window.

Session windows (`generate_session_window`):
- Uses `FeatureExtractor.convert_to_counter_vector` with `group_by_category=[span_id]` to group lines per session.
- Builds one sequence per session by joining group loglines with `logsequence_delim` and aggregates labels across the session (any positive -> session is positive).

### FeatureExtractor

Configuration (`FeatureExtractorConfig`):
- `group_by_category: list[str] | None`
- `group_by_time: str | None`
- `sliding_window: int` (0 for no sliding)
- `steps: int` (stride for sliding windows)
- `max_feature_len: int` (for feature padding in vector mode)

Capabilities:
- `convert_to_counter_vector(log_pattern, attributes, timestamps) -> df`:
  Groups by attributes/time and returns an event index list with counts per group.
- `convert_to_feature_vector(log_vectors, attributes, timestamps) -> (event_index_list, block_list)`:
  Aggregates vector features by group (mean) and carries event indices.
- `convert_to_sequence(log_pattern, attributes, timestamps) -> (event_index_list, event_sequence)`:
  - If `sliding_window > 0`, uses numpy stride tricks to produce overlapping subsequences with step `steps`, maintaining `event_index` for each subsequence and concatenating text within each window.
  - Else, concatenates grouped lines to form one sequence per group.

## Data Flow

1. Ingestion: Raw logs are parsed/cleaned into a `LogRecordObject` (via dataset-specific preprocessors or generic preprocessors).
2. Grouping: Depending on configuration, group by `span_id` and/or time buckets (`timestamp.floor(freq=...)`) and/or categorical attributes.
3. Chunking: Apply sliding-window or session-window logic within each group to produce sequences.
4. Label Aggregation: Reduce per-line labels to per-window/per-session labels (logical OR over lines; with sliding windows, also OR with the label of the next line).
5. Output: A new `LogRecordObject` or DataFrame with sequence text, attributes (time bucket, categories, `span_id`), optional `next_logline`, and labels.

## Label Semantics

Binary labels are propagated forward:
- Sliding windows: `window_label = any(line_label) OR next_line_label`
- Session windows: `session_label = any(line_label in session)`

This convention emphasizes early detection of emerging anomalies and aligns with next-step prediction objectives.

## Configuration Examples

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
seq_df = p.group_sliding_window(df, constants.LOGLINE_NAME)
```

```python
from logai.preprocess.openset_partitioner import OpenSetPartitioner, OpenSetPartitionerConfig

# Per-session sliding windows
cfg = OpenSetPartitionerConfig(sliding_window=5, session_window=False)
osp = OpenSetPartitioner(cfg)
seq_lr = osp.generate_sliding_window(logrecord)

# One sequence per session
cfg = OpenSetPartitionerConfig(sliding_window=0, session_window=True)
osp = OpenSetPartitioner(cfg)
session_lr = osp.generate_session_window(logrecord)
```

## Edge Cases and Behavior

- `sliding_window <= 0`: No windowing; group concatenation or original lines returned
- `exclude_last_window=True`: trailing partial window is dropped (pandas `closed="left"`)
- `exclude_smaller_windows=True`: windows shorter than the configured size are filtered out
- Missing timestamps: time-based grouping is skipped if `group_by_time` is unset or `timestamp` column is absent
- Empty groups: groups with insufficient lines may be removed when exclusion flags are set

## Performance Considerations

- Pandas `rolling` is used in `Partitioner` for simplicity; numpy stride tricks in `FeatureExtractor` allow `steps` control and efficient window generation
- Grouping before windowing confines operations to smaller partitions (by time, category, or session), which scales better for large datasets

## Extensibility

- New sessionization strategies can be introduced by deriving ids in preprocessors and reusing either `Partitioner` or `FeatureExtractor`
- Label reducers can be generalized (e.g., majority vote, weighted, time-decayed) by replacing the current OR-based reductions
- Additional grouping keys (e.g., host, service, container) can be appended to `group_by_category`

## Testing

- Unit tests validate grouping/window behavior and open-set partitioning:
  - `tests/logai/preprocess/test_partition.py`
  - `tests/logai/preprocess/test_openset_partition.py`

## File Map

- `logai/preprocess/partitioner.py`
- `logai/preprocess/openset_partitioner.py`
- `logai/information_extraction/feature_extractor.py`
- `logai/preprocess/hdfs_preprocessor.py`
- `logai/preprocess/bgl_preprocessor.py`
- `logai/dataloader/data_model.py`

## Future Work

- Unify step-size support in `Partitioner` (currently only in `FeatureExtractor`)
- Support alternative window label semantics (e.g., last-line label, window majority)
- Add Sphinx Markdown (e.g., MyST) if we want this page in the published docs site
