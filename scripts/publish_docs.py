"""Trusted publisher: validate run identity, consume inert files, update mike state.

Never import, build, or execute anything from a PR or its artifact here.
This script and its dependencies must come from the repository's default branch.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile
from urllib.parse import quote

from mike import commands, git_utils
from build_docs import release_version
from docs_snapshot import validate_metadata, read_snapshot


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


STATE = ".docs-state"


def now():
    return datetime.now(timezone.utc)


def read_state(path, default=None):
    if not run("git", "ls-tree", "--name-only", "gh-pages", f"{STATE}/{path}"):
        return default
    return json.loads(run("git", "show", f"gh-pages:{STATE}/{path}"))


def write_state(changes, message):
    with git_utils.Commit("gh-pages", message, allow_empty=True) as commit:
        for path, value in changes.items():
            commit.add_file(
                git_utils.FileInfo(
                    f"{STATE}/{path}", json.dumps(value, sort_keys=True) + "\n"
                )
            )


def prepare_branch():
    exists = bool(run("git", "ls-remote", "--heads", "origin", "gh-pages"))
    if exists:
        run("git", "fetch", "origin", "gh-pages:refs/heads/gh-pages")
    run("git", "config", "user.name", "github-actions[bot]")
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    if not exists:
        write_state(
            {"publications.json": {}, "pending.json": {}},
            "Initialize documentation state",
        )


def approval_path(metadata):
    return f"approvals/pr-{metadata['number']}/{metadata['sha']}.json"


def resolve(repo, event, manual_run, artifact_root, snapshot_id="", expire=False):
    if expire:
        return {"kind": "expire"}
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request_target":
        number = int(event["number"])
        if api(repo, f"pulls/{number}")["state"] != "closed":
            print("Skip cleanup: PR has reopened")
            return None
        return {"number": number, "kind": "cleanup"}
    if snapshot_id:
        # Only the trusted rebuild workflow can authorize this route. The build
        # job is a dependency of the reusable publisher and has already passed.
        expected_ref = f"{repo}/.github/workflows/docs-rebuild.yml@refs/heads/main"
        if (
            os.environ["GITHUB_EVENT_NAME"] != "workflow_dispatch"
            or os.environ["GITHUB_WORKFLOW_REF"] != expected_ref
            or event.get("inputs", {}).get("snapshot_id") != snapshot_id
        ):
            raise ValueError(
                "Explicit rebuild approval must originate from the main workflow"
            )
        record = read_snapshot(repo, snapshot_id)
        metadata = record["metadata"]
        run_id, attempt = (
            int(os.environ["GITHUB_RUN_ID"]),
            int(os.environ["GITHUB_RUN_ATTEMPT"]),
        )
        download_artifact(repo, run_id, artifact_root, attempt, reuse_previous=True)
        if json.loads((artifact_root / "build.json").read_text()) != metadata:
            raise ValueError("Rebuild artifact does not match the approved snapshot")
        write_state(
            {approval_path(metadata): metadata},
            "Approve exact PR documentation snapshot",
        )
        return dict(
            kind="preview",
            metadata=metadata,
            snapshot_id=snapshot_id,
            run_id=run_id,
            number=metadata["number"],
            **{k: metadata[k] for k in ("sha", "version", "prefix")},
        )
    if not manual_run and "workflow_run" not in event:
        raise ValueError(
            "Publishing an existing build requires run_id; choose cleanup for an expiry sweep"
        )
    run_id = int(manual_run or event["workflow_run"]["id"])
    attempt = (
        int(event["workflow_run"]["run_attempt"])
        if not manual_run
        else int(api(repo, f"actions/runs/{run_id}")["run_attempt"])
    )
    build = api(repo, f"actions/runs/{run_id}/attempts/{attempt}")
    workflow = api(repo, "actions/workflows/docs-ci.yml")
    if (
        build["workflow_id"] != workflow["id"]
        or build["conclusion"] != "success"
        or build["repository"]["full_name"].lower() != repo.lower()
    ):
        raise ValueError("Expected a successful Documentation run from this repository")
    download_artifact(repo, run_id, artifact_root, attempt)
    metadata = validate_metadata(
        json.loads((artifact_root / "build.json").read_text()), repo
    )
    if metadata["head_sha"] != build["head_sha"]:
        raise ValueError("Artifact head does not match the authenticated workflow run")
    result = dict(
        metadata=metadata,
        run_id=run_id,
        number=metadata["number"],
        **{k: metadata[k] for k in ("sha", "version", "prefix")},
    )
    if build["event"] == "pull_request":
        number = metadata["number"]
        if number is None:
            raise ValueError("PR build has no PR identity")
        # The artifact cannot choose an unrelated PR or invented merge snapshot.
        prs = api(repo, f"commits/{build['head_sha']}/pulls?per_page=100")
        if number not in {pr["number"] for pr in prs}:
            raise ValueError("Snapshot head is not associated with this PR")
        associated = {pr["number"] for pr in build.get("pull_requests", [])}
        if associated and number not in associated:
            raise ValueError("PR is not associated with the source run")
        # GitHub records the revision of locally called workflows independently
        # of artifact contents. Require the PR merge ref, not a caller-selected
        # tag/SHA, and reject absent/ambiguous provenance (including old builds).
        expected_ref = f"refs/pull/{number}/merge"
        expected_sha = metadata["sha"]
        expected_path = f"{repo}/.github/workflows/docs-build.yml@{expected_sha}"
        sources = [
            item
            for item in build.get("referenced_workflows", [])
            if item.get("path") == expected_path
            and item.get("ref") == expected_ref
            and item.get("sha") == expected_sha
        ]
        if len(sources) != 1:
            raise ValueError(
                "Artifact merge SHA lacks matching GitHub workflow provenance; "
                "start a new Documentation build"
            )
        commit = api(repo, f"commits/{metadata['sha']}")
        if [parent["sha"] for parent in commit["parents"]] != [
            metadata["base_sha"],
            metadata["head_sha"],
        ]:
            raise ValueError("Snapshot is not the recorded PR merge")
        pr = api(repo, f"pulls/{number}")
        if pr["base"]["repo"]["full_name"].lower() != repo.lower():
            raise ValueError("PR targets another repository")
        fork = (
            pr["head"]["repo"] is None
            or pr["head"]["repo"]["full_name"].lower() != repo.lower()
        )
        snapshot_id = f"{run_id}-{attempt}"
        record = {"id": snapshot_id, "metadata": metadata, "fork": fork}
        saved = read_state(f"snapshots/{snapshot_id}.json")
        if saved is not None and saved != record:
            raise ValueError("Refusing to overwrite a recorded snapshot")
        write_state(
            {f"snapshots/{snapshot_id}.json": record},
            "Record PR documentation snapshot",
        )
        print(f"Recorded snapshot {snapshot_id}: {metadata['sha']}")
        if manual_run:
            write_state(
                {approval_path(metadata): metadata},
                "Approve exact PR documentation snapshot",
            )
        approved = read_state(approval_path(metadata)) == metadata
        if fork and not approved:
            print(
                f"Fork preview requires approval: rebuild snapshot {snapshot_id} or approve run {run_id}"
            )
            return None
        if not manual_run and pr["head"]["sha"] != metadata["head_sha"]:
            print("Skip superseded PR head; its recorded snapshot remains rebuildable")
            return None
        # Closed PRs may be republished by a late build. Age-based cleanup is the
        # agreed fallback; retries do not have to infer human intent from actors.
        result.update(kind="preview", snapshot_id=snapshot_id)
    elif build["event"] in {"push", "workflow_dispatch"}:
        if metadata["number"] is not None or metadata["version"] != "latest":
            raise ValueError("Main build has incorrect destination")
        if (
            build["head_branch"] != "main"
            or api(repo, "commits/main")["sha"] != metadata["sha"]
        ):
            print("Skip obsolete or non-main build")
            return None
        result.update(kind="main")
    elif build["event"] == "release":
        tag = release_version(metadata["version"])
        release = api(repo, f"releases/tags/{quote(tag, safe='')}")
        if (
            metadata["number"] is not None
            or release["draft"]
            or release["prerelease"]
            or api(repo, f"commits/{quote(tag, safe='')}")["sha"] != metadata["sha"]
        ):
            raise ValueError("Expected a published release matching this build")
        result.update(kind="release")
    else:
        raise ValueError("Unsupported build event")
    return result


def download_artifact(repo, run_id, target, attempt, reuse_previous=False):
    artifacts = []
    page = 1
    while True:
        batch = api(repo, f"actions/runs/{run_id}/artifacts?per_page=100&page={page}")[
            "artifacts"
        ]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    if reuse_previous:
        # Re-running only a failed publish job reuses its successful build job.
        # That artifact retains the earlier attempt's name. Inputs/source are
        # immutable for this manual workflow run; metadata is still checked.
        attempts = [
            int(match[1])
            for artifact in artifacts
            if (match := re.fullmatch(r"docs-site-([1-9][0-9]*)", artifact["name"]))
            and int(match[1]) <= attempt
        ]
        if attempts:
            attempt = max(attempts)
    artifacts = [
        a for a in artifacts if a["name"] == f"docs-site-{attempt}" and not a["expired"]
    ]
    if len(artifacts) != 1:
        raise ValueError(
            "Expected exactly one unexpired docs-site artifact for this attempt"
        )
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
    branch_exists = True  # prepare_branch initializes durable state before resolution.
    number = selected.get("number")
    pending = read_state("pending.json", {})
    publications = read_state("publications.json", {})
    if selected["kind"] in {"cleanup", "expire"}:
        versions = (
            existing_versions("previews")
            if run(
                "git", "ls-tree", "--name-only", "gh-pages", "previews/versions.json"
            )
            else []
        )
        names = {version["version"] for version in versions}
        if selected["kind"] == "cleanup":
            expired = {f"pr-{number}"}
        else:
            days = int(os.environ.get("DOCS_PREVIEW_RETENTION_DAYS", "30"))
            if days < 1:
                raise ValueError("Preview retention must be a positive number of days")
            cutoff = now() - timedelta(days=days)
            # Pending deployments have unknown success until finalized. Redeploy
            # and finalize them before making an expiry decision on their contents.
            expired = {
                name
                for name, value in publications.items()
                if name in names
                and name not in pending
                and datetime.fromisoformat(value["published_at"]) < cutoff
            }
            # Legacy previews have no success timestamp. Preserve them for one
            # retention period from migration instead of guessing their age.
            for name in names - publications.keys() - pending.keys():
                pending[name] = {
                    "number": int(name.removeprefix("pr-")),
                    "sha": "",
                    "snapshot_id": None,
                }
        for name in sorted(expired):
            if name in names:
                commands.delete([name], branch="gh-pages", deploy_prefix="previews")
            pending[name] = None
    else:
        metadata = json.loads((artifact_root / "build.json").read_text())
        expected = selected["metadata"]
        if metadata != expected:
            raise ValueError("Artifact metadata does not match the verified build")
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
        if selected["kind"] == "preview":
            pending[selected["version"]] = {
                "number": number,
                "sha": selected["sha"],
                "snapshot_id": selected["snapshot_id"],
            }
    write_state({"pending.json": pending}, "Prepare documentation deployment")
    run("git", "push", "origin", "gh-pages")
    if selected["kind"] in {"cleanup", "expire"} and not pending:
        return
    if not run("git", "ls-tree", "--name-only", "gh-pages", "index.html"):
        # A registry containing only unapproved fork snapshots is not a site.
        return
    pages_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "pages.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={archive}", "gh-pages"],
            check=True,
        )
        with zipfile.ZipFile(archive) as z:
            # Durable records/approvals belong in Git, not on the public site.
            z.extractall(
                pages_dir, [n for n in z.namelist() if not n.startswith(STATE + "/")]
            )
    owner, name = repo.split("/")
    base = f"https://{owner.lower()}.github.io/{name}/"
    path = f"previews/pr-{number}/" if selected["kind"] == "preview" else "latest/"
    if selected["kind"] == "release":
        path = selected["version"] + "/"
    set_output("publish", "true")
    set_output("url", base + path)


def finish(repo):
    """Called only after Pages reports success, within the publisher's lock."""
    pending = read_state("pending.json", {})
    publications = read_state("publications.json", {})
    notifications = read_state("notifications.json", {})
    timestamp = now().isoformat()
    for name, value in pending.items():
        if value is None:
            publications.pop(name, None)
        else:
            publications[name] = dict(value, published_at=timestamp)
        notifications[name] = value
    write_state(
        {
            "publications.json": publications,
            "pending.json": {},
            "notifications.json": notifications,
        },
        "Record successful documentation publication",
    )
    run("git", "push", "origin", "gh-pages")
    notify(repo)


def notify(repo):
    """Retry comments independently of deployment and retain failed notifications."""
    notifications = read_state("notifications.json", {})
    owner, project = repo.split("/")
    for name, value in list(notifications.items()):
        number = int(name.removeprefix("pr-"))
        url = f"https://{owner.lower()}.github.io/{project}/previews/{name}/"
        try:
            comment(
                repo,
                number,
                url,
                value["sha"] if value else "",
                value is None,
                value.get("snapshot_id") if value else None,
            )
        except subprocess.CalledProcessError:
            # Comments are optional. Preserve this entry and try other PRs;
            # never let a PR-specific API failure block Pages deployment.
            print(
                f"::warning::Preview comment for PR {number} failed; queued for retry"
            )
            continue
        del notifications[name]
        write_state(
            {"notifications.json": notifications}, "Record preview notification"
        )
        run("git", "push", "origin", "gh-pages")


def comment(repo, number, url, sha, removed, snapshot_id=None):
    marker = "<!-- docs-preview -->"
    body = marker + (
        "\nDocumentation preview removed (PR closed or preview expired)."
        if removed
        else f"\n[Open documentation preview]({url})\n\nBuilt for `{sha}`. This is an unreleased PR preview."
    )
    if snapshot_id and not removed:
        body += f"\n\nRebuild snapshot: `{snapshot_id}` (Actions → Rebuild and publish preview)."
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--expire", action="store_true")
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--pages-dir", type=Path)
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    prepare_branch()
    if args.finish:
        finish(repo)
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    expire = args.expire or os.environ["GITHUB_EVENT_NAME"] == "schedule"
    with tempfile.TemporaryDirectory() as temp:
        artifact_root = Path(temp) / "artifact"
        selected = resolve(
            repo, event, args.run_id, artifact_root, args.snapshot_id, expire
        )
        if selected:
            publish(repo, selected, args.pages_dir, artifact_root)
        # Snapshot records must survive artifact expiry even when fork publication
        # has not been approved. Never build source in this privileged process.
        run("git", "push", "origin", "gh-pages")
        notify(repo)


if __name__ == "__main__":
    main()
