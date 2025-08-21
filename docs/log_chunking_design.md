# LogAI Log Chunking Design Document

## Overview

LogAI implements sophisticated log chunking mechanisms to partition log data into manageable sequences for processing by machine learning algorithms. The chunking system supports both sliding window and session-based partitioning strategies to handle various log analysis scenarios.

## Architecture

### Core Components

#### 1. Partitioner (`logai/preprocess/partitioner.py`)
The base partitioner provides fundamental chunking capabilities:

- **Sliding Window Partitioning**: Creates overlapping sequences of log entries with configurable window size
- **Time-based Grouping**: Groups logs by temporal intervals using pandas time frequency notations
- **Category-based Grouping**: Groups logs by categorical fields (e.g., span_id, user_id)
- **Counter Vector Generation**: Aggregates log events into statistical counters

**Key Configuration Parameters:**
```python
@dataclass
class PartitionerConfig:
    group_by_category: list = None          # Fields to group by
    group_by_time: str = None               # Time frequency for grouping
    sliding_window: int = 0                 # Window size for sliding window
    sep_token: str = "[SEP]"                # Token separator for sequences
    exclude_last_window: bool = False       # Exclude incomplete final window
    exclude_smaller_windows: bool = False   # Exclude windows smaller than target size
```

#### 2. OpenSetPartitioner (`logai/preprocess/openset_partitioner.py`)
Specialized partitioner for open log datasets with enhanced features:

- **Session Window Partitioning**: Groups logs by session boundaries
- **Sliding Window with Next-data Support**: Includes prediction target data
- **Label Propagation**: Handles anomaly labels across partitioned sequences

**Key Configuration Parameters:**
```python
@dataclass
class OpenSetPartitionerConfig:
    sliding_window: int = 0                 # Size of sliding window
    session_window: bool = True             # Use session-based partitioning
    logsequence_delim: str = "[SEP]"        # Delimiter for log sequences
```

#### 3. FeatureExtractor (`logai/information_extraction/feature_extractor.py`)
Provides advanced chunking with feature extraction capabilities:

- **Counter Vector Conversion**: Transforms log sequences into statistical features
- **Multi-dimensional Grouping**: Supports complex grouping strategies
- **Sliding Window Processing**: Configurable window size and step parameters

## Chunking Algorithms

### 1. Sliding Window Algorithm

**Implementation Location**: `partitioner.py:188-212`

The sliding window algorithm creates overlapping sequences of log entries:

```python
def _sliding_window(self, loglines: pd.Series) -> list:
    if self.config.sliding_window <= 0:
        return list(loglines)
    
    windows = loglines.rolling(
        window=self.config.sliding_window,
        min_periods=self.config.sliding_window,
        closed=closed,
    )
    
    windows = list(map(lambda x: self.config.sep_token.join(x), windows))
    return windows
```

**Features:**
- Uses pandas rolling window for efficient processing
- Configurable minimum window size
- Optional exclusion of incomplete windows
- Concatenates log entries with separator tokens

### 2. Session Window Algorithm

**Implementation Location**: `openset_partitioner.py:119-142`

Session windows group logs by session boundaries using feature extraction:

```python
def generate_session_window(self, logrecord):
    partitioned_data = self.feature_extractor.convert_to_counter_vector(
        log_pattern=logrecord.body[constants.LOGLINE_NAME],
        attributes=logrecord.span_id.join(logrecord.labels),
        timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS],
    )
    
    partitioned_loglines = partitioned_data[constants.LOGLINE_NAME].apply(
        lambda x: self.config.logsequence_delim.join(x)
    )
```

**Features:**
- Groups logs by session identifiers (span_id)
- Maintains temporal ordering within sessions
- Preserves anomaly labels at session level
- Uses configurable delimiters for sequence joining

### 3. Counter Vector Generation

**Implementation Location**: `partitioner.py:61-101`

Transforms partitioned logs into statistical counters:

```python
def group_counter(self, logrecord_df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = (
        logrecord_df.groupby(by=group_bys, as_index=False)
        .size()
        .rename(columns={"size": constants.LOG_COUNTS})
    )
    return grouped_df
```

**Features:**
- Aggregates log events by occurrence count
- Supports multi-dimensional grouping
- Time-aware aggregation with configurable intervals

## Data Flow

### Input Processing
1. **Raw Logs** → `FileDataLoader` → `LogRecordObject`
2. **LogRecordObject** → `Partitioner/OpenSetPartitioner` → **Chunked Sequences**
3. **Chunked Sequences** → `FeatureExtractor` → **Feature Vectors**

### Chunking Pipeline
```
Raw Log Data
    ↓
[Data Loader] (data_loader.py)
    ↓
LogRecordObject (body, attributes, labels, timestamps, span_id)
    ↓
[Partitioner] (partitioner.py / openset_partitioner.py)
    ↓ (sliding_window OR session_window)
Partitioned LogRecordObject
    ↓
[Feature Extractor] (feature_extractor.py)
    ↓
Feature Vectors / Counter Vectors
```

### Memory Management
- **Streaming Processing**: Large logs processed in chunks to manage memory
- **Lazy Evaluation**: pandas operations optimized for memory efficiency
- **Configurable Window Sizes**: Adjustable based on available resources

## Use Cases

### 1. Anomaly Detection
- **Sliding Windows**: Detect anomalies in log sequences
- **Next-data Prediction**: Include target data for supervised learning
- **Label Propagation**: Maintain anomaly labels across windows

### 2. Log Clustering
- **Session Windows**: Group similar log sessions
- **Counter Vectors**: Statistical features for clustering algorithms

### 3. Sequential Analysis
- **Temporal Partitioning**: Time-based log analysis
- **Pattern Mining**: Discover sequential patterns in log data

## Configuration Examples

### Sliding Window Configuration
```python
config = PartitionerConfig(
    sliding_window=10,
    sep_token="[SEP]",
    exclude_last_window=True,
    exclude_smaller_windows=True
)
```

### Session Window Configuration
```python
config = OpenSetPartitionerConfig(
    session_window=True,
    logsequence_delim="[SEP]",
    sliding_window=0  # Disable sliding window
)
```

### Time-based Grouping Configuration
```python
config = PartitionerConfig(
    group_by_time="1H",  # Group by hour
    group_by_category=["span_id", "user_id"]
)
```

## Performance Considerations

### Optimization Strategies
1. **Pandas Vectorization**: Leverages pandas optimized operations
2. **Memory-efficient Rolling**: Uses pandas rolling windows
3. **Configurable Exclusions**: Skip unnecessary small/incomplete windows
4. **Batch Processing**: Process logs in configurable chunks

### Scalability
- **Horizontal Scaling**: Partition large datasets across multiple processes
- **Memory Bounds**: Configurable window sizes prevent memory overflow
- **Streaming Support**: Process logs incrementally for real-time analysis

## Constants and Data Models

### Key Constants (`logai/utils/constants.py`)
- `LOGLINE_NAME`: Standard column name for log content
- `SPAN_ID`: Session identifier column
- `LABELS`: Anomaly/classification labels
- `LOG_TIMESTAMPS`: Timestamp column
- `LOG_COUNTS`: Counter column for aggregated data

### Data Model (`logai/dataloader/data_model.py`)
```python
@dataclass
class LogRecordObject:
    body: pd.DataFrame      # Main log content
    attributes: pd.DataFrame # Structured attributes
    labels: pd.DataFrame    # Classification labels
    timestamp: pd.DataFrame # Temporal information
    span_id: pd.DataFrame   # Session identifiers
```

## Future Enhancements

### Planned Features
1. **Adaptive Window Sizing**: Dynamic window size based on log patterns
2. **Distributed Processing**: Support for distributed log chunking
3. **Real-time Streaming**: Enhanced streaming capabilities
4. **Custom Chunking Strategies**: Plugin architecture for custom algorithms

### Research Directions
1. **ML-driven Chunking**: Use machine learning to optimize window boundaries
2. **Semantic Chunking**: Chunk based on log content semantics
3. **Multi-modal Chunking**: Combine temporal, structural, and semantic features