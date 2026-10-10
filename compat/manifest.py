"""jsonl manifest I/O — drop-in replacement for the two NeMo helpers RULER uses.

Upstream RULER imports `read_manifest` / `write_manifest` from
`nemo.collections.asr.parts.utils.manifest_utils`, which pulls in the whole
NeMo toolkit for two functions that read and write jsonl. These keep the same
signatures and semantics (blank lines skipped on read, one JSON object per
line on write) so no call site needs to change beyond the import.
"""

import json
from pathlib import Path
from typing import List, Union


def read_manifest(manifest: Union[Path, str]) -> List[dict]:
    """Read a jsonl file into a list of dicts, skipping blank lines."""
    data = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_manifest(
    output_path: Union[Path, str],
    target_manifest: List[dict],
    ensure_ascii: bool = True,
) -> None:
    """Write a list of dicts as jsonl, one object per line."""
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write("\n")
