"""Record/verify a byte-exact migration. Git originals remain authoritative."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git_bytes(repo: Path, ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{path}"])


def record(root: Path, source: Path) -> dict:
    records, exclusions = [], []
    refs = {"portfolio-lstm": "180889d1dff70d84b5496125d191ec3d1d23fe0b",
            "PortfolioOptimizationICM": "3993334223923ffee2c585278a5bdedf82918ff7"}
    for name, repo in [("portfolio-lstm", root), ("PortfolioOptimizationICM", source)]:
        entries = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", refs[name]], text=True)
        for line in entries.splitlines():
            metadata, path = line.split("\t", 1)
            if path.startswith(".idea/") or metadata.startswith("160000"):
                exclusions.append({"repository": name, "path": path,
                                   "reason": "IDE metadata" if path.startswith(".idea/") else
                                   "gitlink without submodule URL; original Git bundle retains reference"})
                continue
            if name == "portfolio-lstm":
                destination = "legacy/code-repository/" + path
            elif path.startswith("images/"):
                destination = "results/historical/figures/" + Path(path).name
            elif path.endswith(".py") or path == "README.md":
                destination = "legacy/results-repository/" + path
            elif path.startswith("etf_"):
                destination = "data/snapshots/" + path
            else:
                destination = "results/historical/artifacts/" + path
            original = git_bytes(repo, refs[name], path)
            copied = (root / destination).read_bytes()
            # Windows checkout may use CRLF although the committed Git blob uses LF.
            if copied != original and copied.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
                raise ValueError(f"Copied file differs from committed source: {destination}")
            digest = hashlib.sha256(copied).hexdigest()
            records.append({"repository": name, "source_path": path,
                            "destination": destination, "sha256": digest,
                            "source_git_sha256": hashlib.sha256(original).hexdigest()})
    return {"source_commits": refs, "files": records, "excluded": exclusions}


def verify(root: Path, manifest: dict) -> None:
    for item in manifest["files"]:
        path = root / item["destination"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Migration hash mismatch: {item['destination']}")
    print(f"PASS: {len(manifest['files'])} original files preserved byte-for-byte; "
          f"{len(manifest['excluded'])} explicit metadata/gitlink exclusions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--record-from", type=Path, help="Initial migration only: source repository checkout")
    args = parser.parse_args()
    path = args.root / "docs/migration_manifest.json"
    if args.record_from:
        manifest = record(args.root, args.record_from)
        verify(args.root, manifest)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        verify(args.root, json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
