Rephrased task:

Process the first .log.txt file in the data directory using the repository’s native chunking and segmentation utilities.

Requirements:
- Create the chunks/ directory if it does not exist.
- Save this rephrased prompt to chunks/prompt.txt before any processing.
- Implement chunks/chunk_and_analyze.py that:
  - Loads the first .log.txt file from the data folder.
  - Uses logdetective.utils.get_chunks to produce raw chunks of the original log.
  - Uses logdetective.extractors.DrainExtractor to mine and select representative segments from the log.
  - Saves outputs under chunks/outputs/ as JSON (raw_chunks.json, segments.json) and a human-readable segments.txt.
- After implementing, run the script on the selected log and generate the outputs.

### Outputs (current status)
- Saved summary JSON: `chunks/results.json`
