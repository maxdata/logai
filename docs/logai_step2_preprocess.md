# Step 2: Preprocess and normalize text (regex-based cleaning)

Normalize whitespace, redact tokens, or standardize variable substrings before parsing/vectorization.

## Key types
- `logai.preprocess.preprocessor.PreprocessorConfig`
- `logai.preprocess.preprocessor.Preprocessor`

## Configuration
```python
from logai.preprocess.preprocessor import PreprocessorConfig

cfg = PreprocessorConfig(
    custom_delimiters_regex=[r"\[\d+\]", r"\(pid=\d+\)"],
    custom_replace_list=[
        (r"0x[0-9a-fA-F]+", "<HEX>"),
        (r"\b[0-9]{4,}\b", "<NUM>"),
    ],
)
```

## Usage
```python
from logai.preprocess.preprocessor import Preprocessor

pre = Preprocessor(cfg)
cleaned_series, extracted_terms = pre.clean_log(logrecord.body["logline"])  # returns (Series, DataFrame)
```

## Notes
- `custom_delimiters_regex`: patterns replaced by a single space (tokenization aid).
- `custom_replace_list`: list of (pattern, replacement) to normalize dynamic tokens.
- `extracted_terms` contains columns for each replacement label with matched tokens per line (optional features).
