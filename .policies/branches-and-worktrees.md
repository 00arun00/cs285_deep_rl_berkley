## Branch and worktree conventions

### Owner prefix

Before creating a branch or named worktree, read `branchNamespace.owner` from the effective Git configuration:

```bash
git config --get branchNamespace.owner
```

- If configured, use that value as `<owner>`.
- If unset, tell the user and propose the authenticated GitHub login as the fallback, showing the actual login. **Pause and wait for explicit confirmation before proceeding.**
- If no authenticated GitHub login is available, ask the user to choose a prefix.
- Do not infer initials from `user.name` or silently select a prefix.
- Do not write a configuration value without permission.
- Once the user confirms a fallback, reuse it for the current task without repeatedly asking.

Developers can configure a repository-specific prefix:

```bash
git config --local branchNamespace.owner <owner>
```

Or a personal default across repositories:

```bash
git config --global branchNamespace.owner <owner>
```

### Branch names

Use:

```text
<owner>/<type>/<area>-<purpose>
```

| Type | Use | Example |
|---|---|---|
| `feat` | New functionality | `<owner>/feat/hw1-flow-matching` |
| `fix` | Bug fixes | `<owner>/fix/docs-artifact-collision` |
| `refactor` | Restructuring without intended behavior changes | `<owner>/refactor/hw1-checkpoint-storage` |
| `docs` | Documentation | `<owner>/docs/hw1-onboarding` |
| `ci` | CI workflows and checks | `<owner>/ci/docs-validation` |
| `chore` | Dependencies and maintenance | `<owner>/chore/update-dependencies` |
| `experiment` | Exploratory work | `<owner>/experiment/hw1-dropout-sweep` |

- Use lowercase letters, numbers, and hyphens within components.
- Include the area when useful; omit it for repository-wide work.
- Describe the concrete purpose. Avoid vague names such as `updates`, `final`, or unexplained `v2`.
- Select the type that describes the task’s primary purpose.
- Use identical names locally and remotely.
- Do not rename existing branches unless explicitly requested.
- Check for existing names and Git reference namespace conflicts before creating a branch.

### Tool and agent identifiers

Do not add tool or agent identifiers such as `codex`, `claude`, or `agent*` to branch or worktree names.

If such an identifier would help distinguish concurrent work or avoid naming clutter, explain why it would help in the current context and show the proposed name. **Pause and wait for explicit permission before creating or renaming anything using that identifier.**

Tool defaults do not override this convention.

### Worktree names

Derive implementation worktree names by replacing every `/` in the branch name with `-`:

```text
Branch:    <owner>/fix/docs-artifact-collision
Worktree:  <owner>-fix-docs-artifact-collision
```

- Prefer the current checkout or an existing suitable worktree.
- Create a new worktree when isolation is needed.
- Use one implementation worktree per branch, attached to that named branch.
- Reserve detached worktrees for temporary inspection. Name them `review-pr-<number>` or `review-<short-sha>`.
- Choose the parent storage directory separately; the worktree name follows this convention. An application-managed parent directory does not count as a tool identifier added to the worktree name.

### Backup and alternate-history branches

When you identify a need for a backup or alternate-history branch, explain that need in the context of the current work, then **pause and wait for explicit approval before creating it**.

Preserve the complete original branch name under the appropriate prefix:

```text
backup/<owner>/<type>/<area>-<purpose>
alter/<owner>/<type>/<area>-<purpose>
```

- A backup preserves a recovery point.
- An alternate-history branch holds a rewritten or competing commit sequence.

### Archived branches

With explicit approval, prefix the complete original branch name with `archive/`:

```text
<owner>/experiment/hw1-dropout-sweep
→ archive/<owner>/experiment/hw1-dropout-sweep
```

- Preserve the commit tip when archiving.
- Treat `archive/` as a naming convention, not a GitHub archival state.
- Branches under this prefix remain publicly visible when pushed to a public repository.

### PR and cleanup lifecycle

- Choose the branch name before opening a PR.
- Before renaming an open PR’s head branch, explain the consequences and obtain explicit approval.
- For stacked PRs, state each branch’s base explicitly.
- Before proposing cleanup, check live PR status, branch tips, worktree status, and uncommitted/untracked files.
- Present the exact branch names and full worktree locations, including any work that would be lost.
- Obtain explicit approval before deleting branches or worktrees.
- Distinguish local branches, remote branches, remote-tracking references, and worktrees when proposing changes.
- Verify the result afterward and report what changed and what remains.
