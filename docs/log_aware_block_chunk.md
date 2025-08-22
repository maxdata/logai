## Log-aware block chunking design

### Overview
Traditional log chunking treats each log line independently. That works for short, single-line messages, but falls short for build/test logs, multi-line errors, and stack traces where the meaningful unit is a block of related lines. Log-aware block chunking groups lines into semantically coherent blocks (for example, “Running step X…” followed by its output) and then applies standard token/word chunking within those blocks to respect LLM context limits.

### Line-based vs. block-based
- **Line-based**: each log line is one record. If a line is too long, it is further split by tokens/words. This is simple and fast, but it can scatter multi-line errors and stack traces across chunks, which harms downstream semantic tasks.
- **Block-based**: lines are grouped until a boundary marker is encountered. Each block (header + body) is treated as a single unit for downstream operations. This keeps steps and their outputs intact, improving summarization, root-cause analysis, and retrieval.

### Design goals
- **Preserve semantics**: keep step output, stack traces, and multi-line messages together wherever possible.
- **Composable**: add a lightweight pre-processing stage before existing semantic operators; do not disrupt downstream APIs.
- **Configurable**: support multiple boundary markers and regexes for different CI/log formats.
- **Streaming-friendly**: operate in O(n) time with bounded memory; yield blocks as they are discovered.

### Boundary detection
Blocks begin at well-defined boundaries. Common markers include:
- **CI sections**: `##[section] Starting:`, `##[group]`, `##[endgroup]` (Azure DevOps, GitHub Actions)
- **Build tools**: `[INFO] --- maven-compiler-plugin`, `Task :compileJava` (Maven/Gradle)
- **Test runners**: `--- FAIL:`, `=== RUN` (Go), `FAILURES`/`ERRORS` (pytest, JUnit)

Recommended strategy:
- Prefer explicit start markers when available.
- Optionally allow end markers (e.g., `##[section] Finishing:`), but do not rely on them.
- Fall back to secondary heuristics when explicit markers are absent: indentation runs, timestamp resets, severity prefixes, or known header regexes.

### Implementation in Fenic
Implement block grouping as a simple pre-processing step that runs before semantic operators. The output is a two-column dataset: `header` (block header/step) and `block` (the joined lines). Then apply Fenic’s token/word/recursive chunkers within each block.

```python
from typing import Iterable, Iterator, Dict, List, Optional, Pattern
import re

from fenic.api.functions import io, text, col
from fenic.api.session import Session


def group_blocks(
    lines: Iterable[str],
    start_markers: Optional[List[str]] = None,
    start_regexes: Optional[List[Pattern[str]]] = None,
) -> Iterator[Dict[str, str]]:
    """
    Group lines into blocks by start markers/regexes.

    Yields dicts with keys: {"header": str, "block": str}.
    Uses a streaming single pass with O(1) additional memory per line.
    """
    start_markers = start_markers or ["##[section] Starting:"]
    start_regexes = start_regexes or []

    def is_start(line: str) -> bool:
        return any(line.startswith(m) for m in start_markers) or any(
            r.search(line) is not None for r in start_regexes
        )

    current_header: Optional[str] = None
    current_lines: List[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        if is_start(line):
            if current_header is not None:
                yield {"header": current_header, "block": "\n".join(current_lines).strip()}
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_header is not None:
        yield {"header": current_header, "block": "\n".join(current_lines).strip()}


# 1) Read raw lines
sess = Session()
logs = io.read.text("build.log", session=sess).alias("line")

# 2) Collect and group
raw_lines = [r["line"] for r in logs.collect()]
blocks = list(
    group_blocks(
        raw_lines,
        start_markers=[
            "##[section] Starting:",  # Azure DevOps
            "[INFO] ---",            # Maven
            "--- FAIL:",             # Go test
            "=== RUN",               # Go test
        ],
        start_regexes=[
            re.compile(r"^Task\s+:\S+"),     # Gradle tasks
            re.compile(r"^\[\w+\]\s+"),    # [INFO]/[WARN]/[ERROR]
        ],
    )
)

# 3) Create a block-level dataset and chunk within blocks
df = io.read.rows(blocks, session=sess, columns=["header", "block"])
chunked = (
    df.select(
        "header",
        text.token_chunk(col("block"), chunk_size=800, overlap=50).alias("chunks"),
    )
    .explode("chunks")
    .with_column_renamed("chunks", "chunk")
)
```

Notes:
- `group_blocks` is intentionally minimal and streaming-safe. If input size is large, prefer iterating the file and yielding blocks directly to `io.read.rows` rather than materializing `blocks` in memory.
- Choose `chunk_size` and `overlap` based on downstream model context and desired recall.
- Replace `text.token_chunk` with `text.word_chunk` or a recursive chunker if tokens are not available.

### When to prefer block chunking
- **Build/test logs with multi-line errors or stack traces**: keeps the error context intact.
- **CI systems with explicit sections**: Azure DevOps, GitHub Actions, Jenkins, etc.
- **Root-cause per step**: you want summarization/QA per step instead of per line.

### Edge cases and fallbacks
- **Huge single blocks**: if a block exceeds the model context, chunk within the block with overlap; consider splitting on secondary cues (blank-line groups, repeated timestamp changes) before token chunking.
- **Nested sections**: if the log format has nesting, either flatten by taking only outer headers or extend the header to reflect a path (e.g., `build > test > unit`).
- **Marker drift**: some tools prefix timestamps/severity. Normalize lines (trim timestamps/prefixes) for boundary detection while keeping raw text in the block body.

### Configuration
Expose a small, declarative configuration for different environments:

```yaml
start_markers:
  - "##[section] Starting:"
  - "[INFO] ---"
  - "=== RUN"
  - "--- FAIL:"
start_regexes:
  - "^Task\\s+:\\S+"   # Gradle
  - "^\\[\\w+\\]\\s+"  # Severity prefix
chunk_size: 800
overlap: 50
```

### Performance considerations
- Single pass over lines; boundary checks are O(1) per line for marker prefixes and O(r) for r regexes.
- To reduce regex cost, pre-compile patterns and short-circuit with marker-prefix checks first.
- For very large logs, stream the file and write blocks/chunks incrementally to the sink.

### Testing checklist
- Synthetic logs with known sections to validate exact block boundaries.
- Real CI logs (Azure DevOps, GitHub Actions) to confirm markers.
- Logs with multi-line stack traces to ensure they remain within the same block.
- Oversized blocks to verify intra-block chunking and overlap.

### Summary
Block chunking preserves the semantic integrity of logs by grouping related lines before tokenizing. The approach is simple, configurable, and streaming-friendly. By inserting a small pre-processing stage and then applying Fenic’s chunkers within each block, you keep steps and their outputs together without exceeding LLM context windows.


