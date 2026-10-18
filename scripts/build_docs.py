"""Build a validated, relocatable docs artifact without publishing credentials."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def release_version(value):
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", value):
        raise ValueError("Documentation releases require a vMAJOR.MINOR.PATCH tag")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, help="Recorded metadata for an explicit rebuild"
    )
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.snapshot:
        metadata = json.loads(args.snapshot.read_text())["metadata"]
        if metadata["repository"] != repo or metadata["sha"] != sha:
            raise ValueError("Checkout does not match the recorded build snapshot")
        version, prefix = metadata["version"], metadata["prefix"]
    else:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        event_name = os.environ["GITHUB_EVENT_NAME"]
        number, head_sha, base_sha = None, sha, None
        if event_name == "pull_request":
            number = int(event["number"])
            version, prefix = f"pr-{number}", "previews"
            head_sha = event["pull_request"]["head"]["sha"]
            parents = subprocess.check_output(
                ["git", "show", "-s", "--format=%P", "HEAD"], text=True
            ).split()
            if len(parents) != 2 or parents[1] != head_sha:
                raise ValueError("Expected GitHub's synthetic PR merge checkout")
            base_sha = parents[0]
        elif event_name == "release":
            version, prefix = release_version(event["release"]["tag_name"]), ""
        else:
            if os.environ["GITHUB_REF"] != "refs/heads/main":
                raise ValueError("Manual main builds must run on main")
            version, prefix = "latest", ""
        metadata = dict(
            repository=repo,
            sha=sha,
            head_sha=head_sha,
            base_sha=base_sha,
            number=number,
            version=version,
            prefix=prefix,
        )
    label = (
        f"PR #{metadata['number']} preview — not released"
        if prefix
        else "Latest main — unreleased"
        if version == "latest"
        else f"Release {version}"
    )
    owner, name = repo.split("/")
    base = f"https://{owner.lower()}.github.io/{name}/"
    config = Path("zensical.toml").read_text()
    config = config.replace(
        "[project]\n",
        f"[project]\nsite_url = {json.dumps(base + (prefix + '/' if prefix else ''))}\n",
        1,
    )
    config += f"\n[project.extra]\nbuild_label = {json.dumps(label)}\nbuild_sha = {json.dumps(sha)}\n"
    if not prefix:
        config += '\n[project.extra.version]\nprovider = "mike"\n'
    # Keep the config next to the original so all relative source paths still work.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", prefix=".docs-build-", dir=".", delete=False
    ) as f:
        f.write(config)
        config_path = Path(f.name)
    try:
        env = dict(os.environ, MIKE_DOCS_VERSION=version)
        subprocess.run(
            [
                "zensical",
                "build",
                "--clean",
                "--strict",
                "--config-file",
                str(config_path),
            ],
            check=True,
            env=env,
        )
    finally:
        config_path.unlink()
    if prefix:
        for page in Path("site").rglob("*.html"):
            page.write_text(
                page.read_text().replace(
                    "<head>",
                    '<head><meta name="robots" content="noindex, nofollow">',
                    1,
                )
            )
    target = Path(os.environ.get("DOCS_ARTIFACT_DIR", "docs-artifact"))
    target.mkdir(exist_ok=False)
    shutil.copytree("site", target / "site")
    (target / "build.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
