#!/usr/bin/env python3
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Try to import optional logdetective utilities
try:
	from logdetective.utils import get_chunks as ld_get_chunks  # type: ignore
	has_logdetective = True
except Exception:
	has_logdetective = False
	ld_get_chunks = None  # type: ignore

from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
from logai.information_extraction.log_parser import LogParser, LogParserConfig

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHUNKS_DIR = ROOT / "chunks"
OUTPUTS_DIR = CHUNKS_DIR / "outputs"


def find_first_log() -> Path:
	candidates = sorted(DATA_DIR.glob("*.log.txt"))
	if not candidates:
		raise SystemExit("No .log.txt files found in data directory")
	return candidates[0]


def simple_chunk_lines(lines: List[str], max_chunk_lines: int = 200) -> List[Dict[str, Any]]:
	chunks = []
	buf = []
	start = 0
	for idx, line in enumerate(lines):
		buf.append(line)
		if len(buf) >= max_chunk_lines:
			chunks.append({
				"start_line": start,
				"end_line": idx,
				"text": "".join(buf)
			})
			buf = []
			start = idx + 1
	# tail
	if buf:
		chunks.append({
			"start_line": start,
			"end_line": start + len(buf) - 1,
			"text": "".join(buf)
		})
	return chunks


def read_log_as_df(path: Path) -> pd.DataFrame:
	# Split by pipe with spaces around it
	df = pd.read_csv(
		path,
		sep=r"\s+\|\s+",
		engine="python",
		names=["timestamp", "Level", "SpanId", "logline"],
		na_filter=False,
		skip_blank_lines=True,
	)
	# Drop blank lines
	df = df[df["logline"].astype(str).str.len() > 0].reset_index(drop=True)
	return df


def extract_patterns(loglines: pd.Series) -> Dict[str, Any]:
	pre = Preprocessor(PreprocessorConfig(custom_delimiters_regex=[], custom_replace_list=[]))
	cleaned, _ = pre.clean_log(loglines)
	parser = LogParser(LogParserConfig.from_dict({
		"parsing_algorithm": "drain",
		"parsing_algo_params": {"sim_th": 0.5, "depth": 5},
	}))
	parsed = parser.parse(cleaned.dropna())
	stats = {
		"parsed_rows": int(len(parsed)),
		"unique_patterns": int(parsed["parsed_logline"].nunique()),
		"top_patterns": parsed["parsed_logline"].value_counts().head(25).to_dict(),
	}
	return {"parsed": parsed, "stats": stats}


def main():
	CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
	OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

	log_path = find_first_log()
	text = Path(log_path).read_text(errors="ignore")
	lines = text.splitlines(keepends=True)

	# Chunking
	if has_logdetective and ld_get_chunks is not None:
		raw_chunks = ld_get_chunks(text)  # Expecting a list of dict-like chunks
		# Normalize output
		norm_chunks = []
		for ch in raw_chunks:
			if isinstance(ch, dict) and "text" in ch:
				norm_chunks.append({
					"start_line": ch.get("start_line"),
					"end_line": ch.get("end_line"),
					"text": ch.get("text"),
				})
			else:
				# Fallback normalization for unexpected shapes
				norm_chunks.append({
					"start_line": None,
					"end_line": None,
					"text": str(ch),
				})
	else:
		norm_chunks = simple_chunk_lines(lines)

	# Extract patterns/segments using Drain
	df = read_log_as_df(log_path)
	pat = extract_patterns(df["logline"])
	parsed = pat["parsed"]
	stats = pat["stats"]

	# Derive representative segments: top templates with sample examples
	vc = parsed["parsed_logline"].value_counts()
	top_templates = list(vc.head(20).index)
	segments = []
	for tpl in top_templates:
		rows = parsed[parsed["parsed_logline"] == tpl].head(5)
		examples = rows["logline"].tolist()
		segments.append({
			"template": tpl,
			"count": int(vc[tpl]),
			"examples": examples,
		})

	# Write outputs
	(raw_chunks_path := OUTPUTS_DIR / "raw_chunks.json").write_text(json.dumps(norm_chunks, indent=2))
	(seg_json_path := OUTPUTS_DIR / "segments.json").write_text(json.dumps(segments, indent=2))
	with open(OUTPUTS_DIR / "segments.txt", "w") as f:
		for s in segments:
			f.write(f"[count={s['count']}] {s['template']}\n")
			for ex in s["examples"]:
				f.write(f"  - {ex}\n")
			f.write("\n")

	# Also save a compact run summary
	summary = {
		"input_log": str(log_path),
		"total_lines": len(lines),
		**stats,
	}
	(OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
	print("Wrote:")
	print(" -", raw_chunks_path)
	print(" -", seg_json_path)
	print(" -", OUTPUTS_DIR / "segments.txt")
	print(" -", OUTPUTS_DIR / "summary.json")


if __name__ == "__main__":
	main()
