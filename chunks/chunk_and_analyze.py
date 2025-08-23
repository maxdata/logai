#!/usr/bin/env python3
import json
import sys
import re
import argparse
import argparse
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
	"""Legacy auto chunking with 1-based line numbers and metadata."""
	chunks = []
	buf = []
	start_idx = 0
	for idx, line in enumerate(lines):
		buf.append(line)
		if len(buf) >= max_chunk_lines:
			start_line = start_idx + 1
			end_line = idx + 1
			text = "".join(buf)
			chunks.append({
				"id": len(chunks) + 1,
				"start_line": start_line,
				"end_line": end_line,
				"num_lines": end_line - start_line + 1,
				"text": text,
				"metadata": {"range": f"{start_line}-{end_line}"}
			})
			buf = []
			start_idx = idx + 1
	# tail
	if buf:
		start_line = start_idx + 1
		end_line = start_idx + len(buf)
		text = "".join(buf)
		chunks.append({
			"id": len(chunks) + 1,
			"start_line": start_line,
			"end_line": end_line,
			"num_lines": end_line - start_line + 1,
			"text": text,
			"metadata": {"range": f"{start_line}-{end_line}"}
		})
	return chunks


def parse_log_line(line: str) -> Dict[str, Any]:
	"""Parse a log line of the form 'timestamp | Level | SpanId | logline'.

	Returns a dict with keys: timestamp (str), level (str), span_id (str), body (str).
	If parsing fails, returns empty strings for fields.
	"""
	parts = [p.strip() for p in re.split(r"\s*\|\s*", line, maxsplit=3)]
	if len(parts) >= 4:
		return {"timestamp": parts[0], "level": parts[1], "span_id": parts[2], "body": parts[3]}
	# Fallback: not matching expected shape
	return {"timestamp": "", "level": "", "span_id": "", "body": line.rstrip("\n")}


def auto_chunk_by_span(lines: List[str], max_lines: int = None) -> List[Dict[str, Any]]:
	"""Automatically create contiguous chunks where SpanId stays the same.

	- Uses 1-based start/end line numbers
	- Resets chunk when SpanId changes or optional max_lines is reached
	"""
	chunks: List[Dict[str, Any]] = []
	current_span = None
	start_idx = None
	buf: List[str] = []
	start_ts = ""
	for idx0, line in enumerate(lines):
		line_no = idx0 + 1
		parsed = parse_log_line(line)
		span_id = parsed["span_id"]
		if start_idx is None:
			# start new chunk
			current_span = span_id
			start_idx = idx0
			start_ts = parsed["timestamp"]
			buf = [line]
			continue
		# Continue same span if equal (including empty string)
		should_split = False
		if span_id != current_span:
			should_split = True
		elif max_lines is not None and (idx0 - start_idx + 1) >= max_lines:
			should_split = True
		if should_split:
			start_line = start_idx + 1
			end_line = start_idx + len(buf)
			chunks.append({
				"id": len(chunks) + 1,
				"start_line": start_line,
				"end_line": end_line,
				"num_lines": end_line - start_line + 1,
				"text": "".join(buf),
				"metadata": {
					"strategy": "span",
					"span_id": current_span,
					"start_timestamp": start_ts,
					"end_timestamp": parse_log_line(buf[-1])["timestamp"],
					"range": f"{start_line}-{end_line}",
				},
			})
			# start a new chunk at current line
			current_span = span_id
			start_idx = idx0
			start_ts = parsed["timestamp"]
			buf = [line]
		else:
			buf.append(line)
	# tail
	if start_idx is not None and buf:
		start_line = start_idx + 1
		end_line = start_idx + len(buf)
		chunks.append({
			"id": len(chunks) + 1,
			"start_line": start_line,
			"end_line": end_line,
			"num_lines": end_line - start_line + 1,
			"text": "".join(buf),
			"metadata": {
				"strategy": "span",
				"span_id": current_span,
				"start_timestamp": start_ts,
				"end_timestamp": parse_log_line(buf[-1])["timestamp"],
				"range": f"{start_line}-{end_line}",
			},
		})
	return chunks


def auto_chunk_by_time(lines: List[str], gap_seconds: float = 60.0, max_lines: int = None) -> List[Dict[str, Any]]:
	"""Automatically create contiguous chunks separated by time gaps larger than threshold.

	Parses timestamps via pandas.to_datetime for robustness.
	"""
	chunks: List[Dict[str, Any]] = []
	start_idx = None
	buf: List[str] = []
	start_ts = ""
	prev_dt = None
	for idx0, line in enumerate(lines):
		parsed = parse_log_line(line)
		cur_ts = parsed["timestamp"]
		cur_dt = pd.to_datetime(cur_ts, errors="coerce")
		if start_idx is None:
			start_idx = idx0
			start_ts = cur_ts
			prev_dt = cur_dt
			buf = [line]
			continue
		should_split = False
		if prev_dt is not None and cur_dt is not None:
			delta = (cur_dt - prev_dt).total_seconds()
			if delta is not None and delta > gap_seconds:
				should_split = True
		if not should_split and max_lines is not None and (idx0 - start_idx + 1) >= max_lines:
			should_split = True
		if should_split:
			start_line = start_idx + 1
			end_line = start_idx + len(buf)
			chunks.append({
				"id": len(chunks) + 1,
				"start_line": start_line,
				"end_line": end_line,
				"num_lines": end_line - start_line + 1,
				"text": "".join(buf),
				"metadata": {
					"strategy": "time",
					"time_gap_seconds": gap_seconds,
					"start_timestamp": start_ts,
					"end_timestamp": parse_log_line(buf[-1])["timestamp"],
					"range": f"{start_line}-{end_line}",
				},
			})
			start_idx = idx0
			start_ts = cur_ts
			buf = [line]
		else:
			buf.append(line)
		prev_dt = cur_dt
	# tail
	if start_idx is not None and buf:
		start_line = start_idx + 1
		end_line = start_idx + len(buf)
		chunks.append({
			"id": len(chunks) + 1,
			"start_line": start_line,
			"end_line": end_line,
			"num_lines": end_line - start_line + 1,
			"text": "".join(buf),
			"metadata": {
				"strategy": "time",
				"time_gap_seconds": gap_seconds,
				"start_timestamp": start_ts,
				"end_timestamp": parse_log_line(buf[-1])["timestamp"],
				"range": f"{start_line}-{end_line}",
			},
		})
	return chunks


def chunk_by_size(lines: List[str], chunk_size: int) -> List[Dict[str, Any]]:
	"""Contiguous size-based chunking with 1-based start/end line numbers."""
	chunks: List[Dict[str, Any]] = []
	start_idx = 0
	while start_idx < len(lines):
		end_idx = min(start_idx + chunk_size - 1, len(lines) - 1)
		start_line = start_idx + 1
		end_line = end_idx + 1
		chunk_text = "".join(lines[start_idx : end_idx + 1])
		chunks.append({
			"id": len(chunks) + 1,
			"start_line": start_line,
			"end_line": end_line,
			"num_lines": end_line - start_line + 1,
			"text": chunk_text,
			"metadata": {"range": f"{start_line}-{end_line}"}
		})
		start_idx = end_idx + 1
	return chunks


def parse_ranges_arg(ranges: str) -> List[Dict[str, int]]:
	"""Parse comma-separated ranges into list of dicts with 1-based inclusive bounds."""
	if not ranges:
		return []
	range_specs: List[Dict[str, int]] = []
	for part in ranges.split(","):
		part = part.strip()
		if not part:
			continue
		m = re.match(r"^(\d+)-(\d+)$", part)
		if not m:
			raise SystemExit(f"Invalid range syntax: '{part}'. Expected 'start-end', e.g., '10-200'.")
		start, end = int(m.group(1)), int(m.group(2))
		if start < 1 or end < start:
			raise SystemExit(f"Invalid range bounds: '{part}'. Ensure 1 <= start <= end.")
		range_specs.append({"start": start, "end": end})
	return range_specs


def chunk_by_ranges(lines: List[str], range_specs: List[Dict[str, int]]) -> List[Dict[str, Any]]:
	"""Chunk by explicit 1-based inclusive line ranges."""
	chunks: List[Dict[str, Any]] = []
	max_line = len(lines)
	for spec in range_specs:
		start_line = max(1, min(max_line, spec["start"]))
		end_line = max(1, min(max_line, spec["end"]))
		if end_line < start_line:
			continue
		start_idx = start_line - 1
		end_idx = end_line - 1
		chunk_text = "".join(lines[start_idx : end_idx + 1])
		chunks.append({
			"id": len(chunks) + 1,
			"start_line": start_line,
			"end_line": end_line,
			"num_lines": end_line - start_line + 1,
			"text": chunk_text,
			"metadata": {"range": f"{start_line}-{end_line}"}
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
	parser = argparse.ArgumentParser(description="Chunk logs by contiguous line ranges and extract patterns.")
	parser.add_argument("--input", type=str, default=None, help="Path to input log file. Defaults to first data/*.log.txt")
	parser.add_argument("--chunk-size", type=int, default=None, help="Contiguous chunk size in lines.")
	parser.add_argument("--ranges", type=str, default=None, help="Comma-separated 1-based inclusive ranges, e.g., '10-200,250-400'.")
	parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR), help="Directory to write outputs.")
	parser.add_argument("--no-segments", action="store_true", help="Skip Drain-based segment extraction.")
	parser.add_argument("--auto", choices=["span", "time", "reduce"], default="span", help="Auto chunk strategy when ranges/chunk-size not provided.")
	parser.add_argument("--time-gap-seconds", type=float, default=60.0, help="Time gap threshold for --auto time.")
	parser.add_argument("--auto-max-lines", type=int, default=None, help="Optional max lines per auto chunk.")
	# Reduce-style options inspired by Vector Reduce transform: https://vector.dev/docs/reference/configuration/transforms/reduce/
	parser.add_argument("--reduce-group-by", type=str, default="span_id", help="Comma-separated fields to group by: span_id,level")
	parser.add_argument("--reduce-starts-when", type=str, default=None, help="Regex on body indicating start of a transaction.")
	parser.add_argument("--reduce-ends-when", type=str, default=None, help="Regex on body indicating end of a transaction.")
	parser.add_argument("--reduce-expire-after-ms", type=int, default=None, help="Flush group if idle longer than this many ms.")
	parser.add_argument("--reduce-end-every-ms", type=int, default=None, help="Force end for each group every this many ms from start.")
	parser.add_argument("--reduce-max-lines", type=int, default=None, help="Max lines per reduced chunk before forcing a flush.")
	args = parser.parse_args()

	CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	log_path = Path(args.input) if args.input else find_first_log()
	text = Path(log_path).read_text(errors="ignore")
	lines = text.splitlines(keepends=True)

	# Chunking
	if args.ranges:
		range_specs = parse_ranges_arg(args.ranges)
		norm_chunks = chunk_by_ranges(lines, range_specs)
	elif args.chunk_size:
		norm_chunks = chunk_by_size(lines, args.chunk_size)
	else:
		# Auto strategies take precedence; if not provided, fall back to optional logdetective or legacy split
		if args.auto == "span":
			norm_chunks = auto_chunk_by_span(lines, max_lines=args.auto_max_lines)
		elif args.auto == "time":
			norm_chunks = auto_chunk_by_time(lines, gap_seconds=args.time_gap_seconds, max_lines=args.auto_max_lines)
		elif args.auto == "reduce":
			group_by_fields = [f.strip() for f in (args.reduce_group_by or "").split(",") if f.strip()]
			if not group_by_fields:
				group_by_fields = ["span_id"]
			starts_pat = re.compile(args.reduce_starts_when) if args.reduce_starts_when else None
			ends_pat = re.compile(args.reduce_ends_when) if args.reduce_ends_when else None
			def _reduce_group_key(parsed_line: Dict[str, Any]):
				vals = []
				vals_map = {}
				for fld in group_by_fields:
					val = parsed_line.get(fld) if fld in parsed_line else (parsed_line.get("span_id") if fld == "span_id" else parsed_line.get("level") if fld == "level" else "")
					vals.append(val)
					vals_map[fld] = val
				return tuple(vals), vals_map
			# State per group
			state: Dict[Any, Dict[str, Any]] = {}
			norm_chunks = []
			def _flush_group(gkey, gstate):
				if not gstate or not gstate.get("buf"):
					return
				start_idx = gstate["start_idx"]
				start_line = start_idx + 1
				end_line = start_idx + len(gstate["buf"])
				metadata = {
					"strategy": "reduce",
					"group_by": group_by_fields,
					"group_values": gstate["group_values"],
					"starts_when": args.reduce_starts_when,
					"ends_when": args.reduce_ends_when,
					"expire_after_ms": args.reduce_expire_after_ms,
					"end_every_ms": args.reduce_end_every_ms,
					"range": f"{start_line}-{end_line}",
					"start_timestamp": gstate.get("start_ts", ""),
					"end_timestamp": parse_log_line(gstate["buf"][-1]).get("timestamp", ""),
				}
				norm_chunks.append({
					"id": len(norm_chunks) + 1,
					"start_line": start_line,
					"end_line": end_line,
					"num_lines": end_line - start_line + 1,
					"text": "".join(gstate["buf"]),
					"metadata": metadata,
				})
				# reset group
				state.pop(gkey, None)
			# Iterate lines
			for idx0, line in enumerate(lines):
				parsed = parse_log_line(line)
				gkey, gvals = _reduce_group_key(parsed)
				cur_ts = parsed.get("timestamp", "")
				cur_dt = pd.to_datetime(cur_ts, errors="coerce")
				g = state.get(gkey)
				# Determine if we should start a new group buffer
				if g is None:
					if starts_pat is not None and (not starts_pat.search(parsed.get("body", ""))):
						# Skip until a start condition
						continue
					g = {
						"buf": [line],
						"start_idx": idx0,
						"start_ts": cur_ts,
						"last_dt": cur_dt,
						"group_values": gvals,
					}
					state[gkey] = g
					continue
				# Existing buffer: check time-based flush conditions
				idle_ms = None
				if args.reduce_expire_after_ms and g.get("last_dt") is not None and cur_dt is not None:
					idle_ms = (cur_dt - g["last_dt"]).total_seconds() * 1000.0
					if idle_ms is not None and idle_ms > float(args.reduce_expire_after_ms):
						_flush_group(gkey, g)
						# Start new buffer at this line after flush
						g = {
							"buf": [line],
							"start_idx": idx0,
							"start_ts": cur_ts,
							"last_dt": cur_dt,
							"group_values": gvals,
						}
						state[gkey] = g
						continue
				# end_every_ms from start
				if args.reduce_end_every_ms and g.get("start_ts"):
					start_dt = pd.to_datetime(g["start_ts"], errors="coerce")
					if start_dt is not None and cur_dt is not None:
						elapsed_ms = (cur_dt - start_dt).total_seconds() * 1000.0
						if elapsed_ms is not None and elapsed_ms > float(args.reduce_end_every_ms):
							_flush_group(gkey, g)
							g = {
								"buf": [line],
								"start_idx": idx0,
								"start_ts": cur_ts,
								"last_dt": cur_dt,
								"group_values": gvals,
							}
							state[gkey] = g
							continue
				# Merge current line
				g["buf"].append(line)
				g["last_dt"] = cur_dt
				# ends_when condition
				if ends_pat is not None and ends_pat.search(parsed.get("body", "")):
					_flush_group(gkey, g)
					continue
				# max lines condition
				if args.reduce_max_lines and len(g["buf"]) >= int(args.reduce_max_lines):
					_flush_group(gkey, g)
					continue
			# flush remaining groups
			for k in list(state.keys()):
				_flush_group(k, state.get(k))
		elif has_logdetective and ld_get_chunks is not None:
			raw_chunks = ld_get_chunks(text)
			# Normalize output
			norm_chunks = []
			for ch in raw_chunks:
				if isinstance(ch, dict) and "text" in ch:
					start_line = ch.get("start_line")
					end_line = ch.get("end_line")
					norm_chunks.append({
						"id": len(norm_chunks) + 1,
						"start_line": start_line,
						"end_line": end_line,
						"num_lines": (end_line - start_line + 1) if isinstance(start_line, int) and isinstance(end_line, int) else None,
						"text": ch.get("text"),
						"metadata": {"range": f"{start_line}-{end_line}" if isinstance(start_line, int) and isinstance(end_line, int) else None},
					})
				else:
					# Fallback normalization for unexpected shapes
					norm_chunks.append({
						"id": len(norm_chunks) + 1,
						"start_line": None,
						"end_line": None,
						"num_lines": None,
						"text": str(ch),
						"metadata": {"range": None},
					})
		else:
			norm_chunks = simple_chunk_lines(lines)

	# Extract patterns/segments using Drain (optional)
	segments = []
	stats = {"parsed_rows": 0, "unique_patterns": 0, "top_patterns": {}}
	if not args.no_segments:
		df = read_log_as_df(log_path)
		pat = extract_patterns(df["logline"])
		parsed = pat["parsed"]
		stats = pat["stats"]
		# Derive representative segments: top templates with sample examples
		vc = parsed["parsed_logline"].value_counts()
		top_templates = list(vc.head(20).index)
		for tpl in top_templates:
			rows = parsed[parsed["parsed_logline"] == tpl].head(5)
			examples = rows["logline"].tolist()
			segments.append({
				"template": tpl,
				"count": int(vc[tpl]),
				"examples": examples,
			})

	# Write outputs
	raw_chunks_path = out_dir / "raw_chunks.json"
	seg_json_path = out_dir / "segments.json"
	seg_txt_path = out_dir / "segments.txt"
	summary_path = out_dir / "summary.json"
	raw_chunks_path.write_text(json.dumps(norm_chunks, indent=2))
	if segments:
		seg_json_path.write_text(json.dumps(segments, indent=2))
		with open(seg_txt_path, "w") as f:
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
	summary_path.write_text(json.dumps(summary, indent=2))
	print("Wrote:")
	print(" -", raw_chunks_path)
	if segments:
		print(" -", seg_json_path)
		print(" -", seg_txt_path)
	print(" -", summary_path)


if __name__ == "__main__":
	main()
