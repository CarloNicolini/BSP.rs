"""Archive campaign files deterministically with the original formulas and replay records."""
import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile

from net_analysis import verify_witness


def archive(output, directories):
    with gzip.GzipFile(output, "wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for mode, directory in directories.items():
                paths = list(directory.glob("case-*.json"))
                paths.extend(directory.glob("formula-*.cnf"))
                paths.extend(directory.glob("replay-*.json"))
                paths.extend(directory.glob("environment-*.txt"))
                for path in sorted(paths):
                    data = path.read_bytes()
                    if path.name.startswith("replay-"):
                        replay = json.loads(data)
                        verify_witness(
                            directory / f"formula-{replay['n']}-{replay['seed']}.cnf",
                            replay["n"], replay["result"],
                        )
                    info = tarfile.TarInfo(f"{mode}/{path.name}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive(args.output, {"development": args.development, "heldout": args.heldout})


if __name__ == "__main__":
    main()
