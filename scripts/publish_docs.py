"""Trusted publisher: validate run identity, consume inert files, update mike state.

Never import, build, or execute anything from a PR or its artifact here.
This script and its dependencies must come from the repository's default branch.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile
from urllib.parse import quote

from mike import commands
from build_docs import release_version


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def api(repo, path, method="GET", data=None):
    args = ["gh", "api", f"repos/{repo}/{path}", "--method", method]
    if data is not None:
        args += ["--input", "-"]
    result = subprocess.run(
        args,
        input=json.dumps(data) if data is not None else None,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def unpack(archive, target):
    """Reject traversal, links, duplicate members and oversized artifacts."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) > 20000 or sum(i.file_size for i in infos) > 200_000_000:
            raise ValueError("Documentation artifact exceeds extraction limits")
        seen = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or any(p.lower() == ".git" for p in path.parts)
                or not path.parts
                or path.parts[0] not in {"site", "build.json"}
                or info.filename in seen
                or stat.S_ISLNK(mode)
                or (
                    stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                )
            ):
                raise ValueError(f"Unsafe archive entry: {info.filename!r}")
            seen.add(info.filename)
        z.extractall(target)
    if not (target / "site/index.html").is_file():
        raise ValueError("Artifact has no site index")


def resolve(repo, event, manual_run, artifact_root):
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request_target":
        number = int(event["number"])
        if api(repo, f"pulls/{number}")["state"] != "closed":
            print("Skip cleanup: PR has reopened")
            return None
        return {"number": number, "kind": "cleanup"}
    run_id = int(manual_run or event["workflow_run"]["id"])
    build = api(repo, f"actions/runs/{run_id}")
    workflow = api(repo, "actions/workflows/docs-ci.yml")
    if (
        build["workflow_id"] != workflow["id"]
        or build["conclusion"] != "success"
        or build["repository"]["full_name"].lower() != repo.lower()
    ):
        raise ValueError(
            "Expected a successful Documentation workflow run from this repository"
        )
    sha = build["head_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid source commit")
    result = {"sha": sha, "run_id": run_id, "number": None, "prefix": ""}
    if build["event"] == "pull_request":
        candidates = api(repo, f"commits/{sha}/pulls?per_page=100")
        candidates = [
            p
            for p in candidates
            if p["head"]["sha"] == sha
            and p["state"] == "open"
            and p["base"]["ref"] == "main"
        ]
        if len(candidates) != 1:
            print("Skip closed, stale or ambiguous PR run")
            return None
        pr = candidates[0]
        fork = pr["head"]["repo"]["full_name"].lower() != repo.lower()
        if fork and not manual_run:
            print(
                f"Fork PR #{pr['number']}: maintainer must dispatch this workflow with run ID {run_id}"
            )
            return None
        result.update(
            kind="preview",
            version=f"pr-{pr['number']}",
            prefix="previews",
            number=pr["number"],
        )
    elif build["event"] in {"push", "workflow_dispatch"}:
        if build["head_branch"] != "main" or api(repo, "commits/main")["sha"] != sha:
            print("Skip obsolete or non-main build")
            return None
        result.update(kind="main", version="latest")
    elif build["event"] == "release":
        # workflow_run.head_branch is not a reliable release-tag identifier.
        # Treat the artifact tag only as a lookup key, then verify the published
        # release and its peeled commit against GitHub's authenticated run data.
        download_artifact(repo, run_id, artifact_root)
        hint = json.loads((artifact_root / "build.json").read_text())
        tag = release_version(hint["version"])
        release = api(repo, f"releases/tags/{quote(tag, safe='')}")
        if (
            release["draft"]
            or release["prerelease"]
            or api(repo, f"commits/{quote(tag, safe='')}")["sha"] != sha
        ):
            raise ValueError(
                "Only published, non-prerelease tags matching the build can publish docs"
            )
        result.update(kind="release", version=tag)
    else:
        raise ValueError("Unsupported build event")
    return result


def download_artifact(repo, run_id, target):
    artifacts = api(repo, f"actions/runs/{run_id}/artifacts?per_page=100")["artifacts"]
    artifacts = [a for a in artifacts if a["name"] == "docs-site" and not a["expired"]]
    if len(artifacts) != 1:
        raise ValueError("Expected exactly one unexpired docs-site artifact")
    if artifacts[0]["size_in_bytes"] > 100_000_000:
        raise ValueError("Compressed artifact exceeds download limit")
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "site.zip"
        with archive.open("wb") as f:
            subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{repo}/actions/artifacts/{int(artifacts[0]['id'])}/zip",
                ],
                check=True,
                stdout=f,
            )
        unpack(archive, target)


def existing_versions(prefix=""):
    return json.loads(
        run("git", "show", f"gh-pages:{prefix + '/' if prefix else ''}versions.json")
    )


def set_output(name, value):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def publish(repo, selected, pages_dir, artifact_root):
    branch_exists = bool(run("git", "ls-remote", "--heads", "origin", "gh-pages"))
    if branch_exists:
        run("git", "fetch", "origin", "gh-pages:refs/heads/gh-pages")
    run("git", "config", "user.name", "github-actions[bot]")
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    number = selected.get("number")
    if selected["kind"] == "cleanup":
        if not branch_exists:
            return
        files = run(
            "git", "ls-tree", "--name-only", "gh-pages", "previews/versions.json"
        )
        if files and any(
            v["version"] == f"pr-{number}" for v in existing_versions("previews")
        ):
            commands.delete(
                [f"pr-{number}"], branch="gh-pages", deploy_prefix="previews"
            )
        # Even if already absent in git, redeploy: an earlier Pages upload may
        # have failed after the deletion commit was pushed.
    else:
        if not artifact_root.exists():
            download_artifact(repo, selected["run_id"], artifact_root)
        metadata = json.loads((artifact_root / "build.json").read_text())
        expected = {
            "repository": repo,
            "sha": selected["sha"],
            "version": selected["version"],
            "prefix": selected["prefix"],
        }
        if metadata != expected:
            raise ValueError("Artifact metadata does not match verified GitHub run")
        site = artifact_root / "site"
        (site / "published-build.json").write_text(json.dumps(expected))
        version = selected["version"]
        cfg = {"site_dir": str(site), "use_directory_urls": True}
        exists = branch_exists and bool(
            run(
                "git",
                "ls-tree",
                "--name-only",
                "gh-pages",
                f"{version}/published-build.json",
            )
        )
        if selected["kind"] == "release" and exists:
            saved = json.loads(
                run("git", "show", f"gh-pages:{version}/published-build.json")
            )
            if saved != expected:
                raise ValueError("Refusing to overwrite an immutable release")
            print("Release already exists; preserve its original bytes")
        else:
            aliases = []
            if selected["kind"] == "release":
                versions = (
                    existing_versions()
                    if branch_exists
                    and run(
                        "git", "ls-tree", "--name-only", "gh-pages", "versions.json"
                    )
                    else []
                )
                released = [
                    tuple(map(int, v["version"][1:].split(".")))
                    for v in versions
                    if re.fullmatch(r"v\d+\.\d+\.\d+", v["version"])
                ]
                if not released or tuple(map(int, version[1:].split("."))) >= max(
                    released
                ):
                    aliases = ["stable"]
            # Pinned mike API manages versions, aliases and commits. No build here.
            with commands.deploy(
                cfg,
                version,
                title="Latest main" if version == "latest" else version,
                aliases=aliases,
                update_aliases=True,
                alias_type=commands.AliasType.redirect,
                branch="gh-pages",
                deploy_prefix=selected["prefix"],
                allow_empty=True,
            ):
                pass
        commands.set_default(
            "latest", branch="gh-pages", allow_empty=True, allow_undefined=True
        )
    run("git", "push", "origin", "gh-pages")
    # Materialize only generated output, never check out or execute its files.
    pages_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "pages.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={archive}", "gh-pages"],
            check=True,
        )
        with zipfile.ZipFile(archive) as z:
            z.extractall(pages_dir)
    owner, name = repo.split("/")
    base = f"https://{owner.lower()}.github.io/{name}/"
    path = f"previews/pr-{number}/" if number else selected["version"] + "/"
    set_output("publish", "true")
    set_output("url", base + path)
    set_output("number", number or "")
    set_output("removed", str(selected["kind"] == "cleanup").lower())
    set_output("sha", selected.get("sha", ""))


def comment(repo, number, url, sha, removed):
    marker = "<!-- docs-preview -->"
    body = marker + (
        "\nDocumentation preview removed because this PR is closed."
        if removed
        else f"\n[Open documentation preview]({url})\n\nBuilt for `{sha}`. This is an unreleased PR preview."
    )
    comments = json.loads(
        run(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/issues/{number}/comments?per_page=100",
        )
    )
    found = next(
        (
            c
            for page in comments
            for c in page
            if c["body"].startswith(marker)
            and c["user"]["login"] == "github-actions[bot]"
        ),
        None,
    )
    if found:
        api(repo, f"issues/comments/{found['id']}", "PATCH", {"body": body})
    elif not removed:
        api(repo, f"issues/{number}/comments", "POST", {"body": body})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pages-dir", type=Path)
    parser.add_argument("--comment", type=int)
    parser.add_argument("--url", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--removed", action="store_true")
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    if args.comment:
        comment(repo, args.comment, args.url, args.sha, args.removed)
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    with tempfile.TemporaryDirectory() as temp:
        artifact_root = Path(temp) / "artifact"
        selected = resolve(repo, event, args.run_id, artifact_root)
        if selected:
            publish(repo, selected, args.pages_dir, artifact_root)


if __name__ == "__main__":
    main()
