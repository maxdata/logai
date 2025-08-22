# Step 1: Load raw logs into a unified data model (LogRecordObject)

This step ingests logs from files and produces a `LogRecordObject`, a column-aligned container compatible with OpenTelemetry’s log record fields.

## Key types
- `logai.dataloader.data_model.LogRecordObject`
- `logai.dataloader.data_loader.DataLoaderConfig`
- `logai.dataloader.data_loader.FileDataLoader`

## Configuration
```python
from logai.dataloader.data_loader import DataLoaderConfig

cfg = DataLoaderConfig(
    filepath="/path/to/log.txt",
    log_type="text",  # csv | tsv | json | text
    reader_args={
        # For free-form text, provide a log_format using angle-bracket fields
        # Example: "<timestamp> | <Level> | <SpanId> | <logline>"
        "log_format": "<timestamp> | <Level> | <SpanId> | <logline>",
    },
    dimensions={
        # Map file columns to LogRecordObject fields
        "timestamp": ["timestamp"],
        "body": ["logline"],
        "span_id": ["SpanId"],
    },
    infer_datetime=True,
    datetime_format="%Y-%m-%dT%H:%M:%S.%f",
)
```

## Usage
```python
from logai.dataloader.data_loader import FileDataLoader

logrecord = FileDataLoader(cfg).load_data()
# Access fields
body = logrecord.body  # DataFrame with column "logline"
ts = logrecord.timestamp  # DataFrame with column "timestamp"
attrs = logrecord.attributes  # optional structured cols
```

## Notes
- For `log_type="text"`, `reader_args.log_format` is required. The parser builds a regex from the format and extracts named groups.
- If `dimensions` is empty, all columns are concatenated into a single `logline`.
- The object preserves index alignment across fields for safe joins/feature extraction.
