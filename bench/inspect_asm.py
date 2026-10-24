"""Inspect emitted assembly for the two SP kernels, without dumping LLVM blobs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("assembly", type=Path)
args = parser.parse_args()
lines = args.assembly.read_text().splitlines()
report = {}
for name in ["sweep", "cavities"]:
    start = next(i for i,line in enumerate(lines) if line.endswith(":") and f"{len(name)}{name}17h" in line and "Solver" in line)
    end = next(i for i in range(start+1,len(lines)) if ".cfi_endproc" in lines[i])
    body = lines[start:end]
    opcodes = Counter(match[1] for line in body if (match := re.match(r"\s+([a-z][a-z0-9.]*)\s",line)))
    report[name] = {"instruction_count":sum(opcodes.values()),"opcodes":dict(opcodes),
        "bounds_panic_paths":sum("panic_bounds_check" in line for line in body)}
print(json.dumps(report,indent=2))
