# Step 3: Parse unstructured loglines into templates and parameters

Mine log templates and extract dynamic parameters with a configurable parser (e.g., Drain, AEL, IPLoM).

## Key types
- `logai.information_extraction.log_parser.LogParserConfig`
- `logai.information_extraction.log_parser.LogParser`

## Configuration
```python
from logai.information_extraction.log_parser import LogParserConfig

cfg = LogParserConfig.from_dict({
    "parsing_algorithm": "drain",  # "ael" | "iplom" also available
    "parsing_algo_params": {
        "sim_th": 0.5,
        "depth": 5,
    },
})
```

## Usage
```python
from logai.information_extraction.log_parser import LogParser

parser = LogParser(cfg)
parsed_df = parser.parse(cleaned_series.dropna())
# Columns: ["logline", "parsed_logline", "parameter_list"]
```

## Notes
- Drain builds a prefix-tree of templates; differing tokens become wildcards.
- `parsed_df["parsed_logline"]` is the template string; use `value_counts()` to get representative segments.
- `parameter_list` holds dynamic tokens detected per line (ordered by position).
