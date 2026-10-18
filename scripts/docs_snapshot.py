"""Read durable PR build records. This module never executes snapshot source."""

import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess

SNAPSHOT_ID = re.compile(r"[1-9][0-9]*-[1-9][0-9]*")
SHA = re.compile(r"[0-9a-f]{40}")
METADATA_KEYS = {
    "repository",
    "sha",
    "head_sha",
    "base_sha",
    "number",
    "version",
    "prefix",
}


def api(repo, path):
    return json.loads(
        subprocess.check_output(["gh", "api", f"repos/{repo}/{path}"], text=True)
    )


def validate_metadata(metadata, repo):
    if set(metadata) != METADATA_KEYS or metadata["repository"] != repo:
        raise ValueError("Invalid documentation snapshot metadata")
    if not all(
        isinstance(metadata[key], str) and SHA.fullmatch(metadata[key])
        for key in ("sha", "head_sha")
    ):
        raise ValueError("Invalid snapshot SHA")
    number = metadata["number"]
    if number is not None:
        if (
            type(number) is not int
            or number <= 0
            or metadata["version"] != f"pr-{number}"
            or metadata["prefix"] != "previews"
            or not isinstance(metadata["base_sha"], str)
            or not SHA.fullmatch(metadata["base_sha"])
        ):
            raise ValueError("Invalid PR snapshot")
    elif (
        metadata["prefix"] != ""
        or metadata["base_sha"] is not None
        or metadata["sha"] != metadata["head_sha"]
        or not re.fullmatch(r"latest|v[0-9]+\.[0-9]+\.[0-9]+", metadata["version"])
    ):
        raise ValueError("Invalid main/release snapshot")
    return metadata


def read_snapshot(repo, snapshot_id):
    if not SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("Snapshot ID must be RUN_ID-ATTEMPT (for example 123456-1)")
    response = api(
        repo, f"contents/.docs-state/snapshots/{snapshot_id}.json?ref=gh-pages"
    )
    record = json.loads(base64.b64decode(response["content"]))
    validate_metadata(record["metadata"], repo)
    if record["id"] != snapshot_id or record["metadata"]["number"] is None:
        raise ValueError("Expected a recorded PR snapshot")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ["GITHUB_REF"] != "refs/heads/main":
        raise ValueError("Run the rebuild workflow from main")
    repo = os.environ["GITHUB_REPOSITORY"]
    record = read_snapshot(repo, args.snapshot_id)
    sha = record["metadata"]["sha"]
    # Fail clearly if GitHub no longer retains the synthetic merge commit.
    if api(repo, f"commits/{sha}")["sha"] != sha:
        raise ValueError("Recorded source commit is unavailable; no fallback is used")
    args.output.write_text(json.dumps(record))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"sha={sha}\n")
        output.write("record=" + json.dumps(record) + "\n")
    print(f"Rebuilding snapshot {args.snapshot_id}: {sha}")


if __name__ == "__main__":
    main()
