# Contributing to MultiMedAI

How we work on this repo. It's a two-person project, so this is deliberately light —
but the rules that *are* here are not optional, because breaking them costs us a
rewrite or a leaked API key.

**Maintainers:** [@UllasP0707](https://github.com/UllasP0707) · [@SaiCharan85](https://github.com/SaiCharan85)

---

## Table of contents

- [The five rules](#the-five-rules)
- [One-time setup](#one-time-setup)
- [The daily loop](#the-daily-loop)
- [Branch naming](#branch-naming)
- [Commit messages](#commit-messages)
- [Opening a pull request](#opening-a-pull-request)
- [Reviewing a pull request](#reviewing-a-pull-request)
- [Merging and cleanup](#merging-and-cleanup)
- [Handling conflicts](#handling-conflicts)
- [What must never be committed](#what-must-never-be-committed)
- [Before you push: manual checks](#before-you-push-manual-checks)
- [Splitting work between us](#splitting-work-between-us)

---

## The five rules

1. **Never commit directly to `main`.** Always branch.
2. **One issue = one branch = one PR.** Don't bundle unrelated fixes.
3. **Every PR gets reviewed by the other person** before merge. No self-merging.
4. **Never commit secrets, weights, or datasets.** See [what must never be committed](#what-must-never-be-committed).
5. **If you break `main`, fixing it is your top priority** — the other person is blocked.

---

## One-time setup

```bash
git clone https://github.com/SaiCharan85/MultiMedAI.git
cd MultiMedAI
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Windows**

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

> ⚠️ `requirements.txt` is currently incomplete — see issue #1. Until that's fixed
> you'll also need: `fastapi uvicorn python-multipart pypdf fpdf2 sentence-transformers google-generativeai`

Set your identity so commits are attributed correctly:

```bash
git config user.name  "Your Name"
git config user.email "your@email.com"
```

API keys go in a **gitignored** `.keys.json` at the repo root — never anywhere else:

```json
{ "gemini": "your-key-here", "serpapi": "your-key-here" }
```

Run the app:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

> 🔒 Keep it bound to `127.0.0.1`. There is **no authentication on any endpoint**
> (issue #14), so binding `0.0.0.0` exposes the key-setting endpoint to your network.

---

## The daily loop

```bash
# 1. Start from an up-to-date main — ALWAYS do this first
git checkout main
git pull origin main

# 2. Branch off it (see naming below)
git checkout -b fix/2-status-endpoint-guard

# 3. Work. Commit in small, logical chunks.
git add src/whatever.py
git commit -m "fix: guard /api/status when retrieval bank is absent"

# 4. Push and set the upstream (first push on a branch)
git push -u origin fix/2-status-endpoint-guard

# 5. Open the PR
gh pr create --web
#   ...or use the "Compare & pull request" button GitHub shows you.
#   Both load the PR template. Avoid `--fill` — it skips the template.

# 6. Ping the other person for review. After approval, merge on GitHub.

# 7. Clean up
git checkout main
git pull origin main
git branch -d fix/2-status-endpoint-guard
```

**Subsequent pushes on the same branch** are just `git push`.

---

## Branch naming

```
<type>/<issue-number>-<short-slug>
```

| Type | Use for | Example |
|---|---|---|
| `fix/` | Bug fixes | `fix/6-render-argument-collision` |
| `feat/` | New functionality | `feat/mps-device-support` |
| `sec/` | Security fixes | `sec/4-server-side-access-grant` |
| `docs/` | Documentation only | `docs/3-macos-setup-instructions` |
| `chore/` | Deps, config, cleanup | `chore/17-remove-dead-streamlit-app` |
| `refactor/` | Restructuring, no behaviour change | `refactor/split-chat-handler` |

Include the issue number when one exists — GitHub then cross-links the branch and
the issue automatically.

**Keep branches short-lived.** A branch open more than ~3 days will start
conflicting with the other person's work. If a task is big, split it into
several PRs.

---

## Commit messages

Format:

```
<type>: <what changed, imperative mood>

<optional body: why, and anything non-obvious>
```

Same types as branches (`fix`, `feat`, `sec`, `docs`, `chore`, `refactor`).

**Good:**

```
fix: read measured accuracy from chest_acc.txt instead of hardcoding 94%

findings.train_chest() already writes the real held-out accuracy but nothing
read it, so the UI claimed 94% regardless of what the head actually scored.

Closes #9
```

**Bad:**

```
update
fixed stuff
asdf
final version 2 REAL
```

### Closing issues automatically

Put `Closes #9` (or `Fixes #9`) in the **PR description**. GitHub closes the issue
when the PR merges. For the grouped checklist issues (#14–#18), use `Refs #15`
instead and tick the checkbox by hand — you don't want a partial fix closing five
other items.

---

## Opening a pull request

```bash
gh pr create --web      # opens the browser with the template pre-loaded
```

A PR description should answer three questions:

1. **What changed?** One or two sentences.
2. **Why?** Link the issue.
3. **How did you verify it?** The actual commands you ran and what you saw.

The [PR template](.github/pull_request_template.md) prompts for these. Fill it in —
"see title" is not a description.

**Keep PRs small.** A 200-line PR gets a real review. A 2,000-line PR gets a
rubber stamp, which is worse than no review at all.

**Draft PRs are encouraged.** Open one early with `gh pr create --draft` if you
want feedback on direction before you've finished. Mark it ready when it is.

---

## Reviewing a pull request

The other person's PR is blocked until you review it. **Aim to review within a day.**

```bash
gh pr list                      # see what's waiting
gh pr checkout 23               # check out PR #23 locally to actually run it
gh pr diff 23                   # or just read the diff
```

### What to look for, in priority order

1. **Does it do what the issue asked?** Read the issue first.
2. **Does it actually run?** Check it out and try the affected path. Reviews that
   only read the diff miss the things that matter in this codebase.
3. **Secrets and large files.** Scan the changed-files list for `.keys.json`,
   anything under `weights/` or `data/`, `*.pt`, `*.safetensors`.
4. **Does it break another branch of `chat()`?** `main.py` routes everything
   through one long function — changes to shared context or ordering have
   non-local effects.
5. **Honesty rules.** This project's stated core rule is *never fabricate metrics
   or findings*. Any new number shown to the user must come from a measurement,
   not a literal. Any new claim in the UI must be something the code verifies.
   (See issues #9, #10, #15.)
6. Style, naming, comments. Last, and least.

### How to leave feedback

- **Request changes** for anything in categories 1–5.
- **Comment** for suggestions you don't want to block on — prefix them `nit:` so
  it's clear they're optional.
- **Approve** when you'd be happy to own the code yourself.

Be direct about the code. It's two people and a shared repo — an unclear review is
worse than a blunt one. Explain *why* something is a problem, not just that it is.

---

## Merging and cleanup

- **Merge method: Squash and merge.** Keeps `main` history one commit per PR and
  readable. Edit the squash commit message to something meaningful before confirming.
- **The author merges**, after approval. Not the reviewer.
- **Delete the branch** on merge (GitHub offers a button; take it).
- Then locally:

```bash
git checkout main
git pull origin main
git branch -d <branch-name>
git remote prune origin        # occasionally, to clear deleted remote branches
```

---

## Handling conflicts

If GitHub says the branch has conflicts:

```bash
git checkout main
git pull origin main
git checkout your-branch
git merge main
# fix the conflicted files, then:
git add <resolved-files>
git commit
git push
```

We use **merge**, not rebase, for shared branches — rebasing a branch someone else
has pulled rewrites history and causes real pain. Rebase is fine on a branch only
you have touched.

**If a conflict looks scary, stop and ask.** A two-minute message beats an hour of
`git reflog`.

---

## What must never be committed

`.gitignore` covers these, but double-check before every push — a secret committed
once is in the history forever, even if you delete it in the next commit.

| Never commit | Why |
|---|---|
| `.keys.json`, `.env`, `*.key` | **API keys.** A leaked Gemini/SerpApi key gets abused within hours. |
| `weights/` | Model checkpoints and the retrieval bank — hundreds of MB to GB. |
| `data/` | Image bank thumbnails, ~80k files. |
| `venv/` | Environment. Rebuildable from `requirements.txt`. |
| `*.pt`, `*.safetensors`, `*.bin`, `*.onnx` | Model binaries. |
| `outputs/web/*.png`, `*.pdf`, `*.log` | Runtime artifacts from your local session. |
| `outputs/web/credentials/` | **Uploaded user credential files.** Real personal documents. |
| `__pycache__/`, `.DS_Store` | Noise. |

Check what you're about to commit:

```bash
git status
git diff --cached --stat        # after git add — review the file list
```

**If you commit a key by accident:** don't just delete it in a follow-up commit.
Revoke the key immediately at the provider, generate a new one, then tell the other
person. The old key is still in the git history.

---

## Before you push: manual checks

There's no CI on this repo yet, so these are on you.

```bash
# 1. Everything still compiles
python -m py_compile main.py app.py src/*.py

# 2. The app boots and the frontend loads
python -m uvicorn main:app --port 8000
#    -> open http://127.0.0.1:8000, send one message, check the browser console
#       for JS errors (especially if you touched static/)

# 3. You didn't stage anything you shouldn't have
git diff --cached --stat
```

If you touched a `src/*.py` module with a CLI, run it:

```bash
python -m src.retrieval "query:chest x-ray"
python -m src.research   # etc.
```

> 💡 Adding GitHub Actions to run step 1 automatically would be a good early PR.

---

## Splitting work between us

To avoid conflicts, try to keep parallel work in **different files**:

| Area | Files |
|---|---|
| Frontend | `static/app.js`, `static/index.html`, `static/style.css` |
| API / routing | `main.py` |
| Capabilities | `src/*.py` |
| Docs | `README.md`, `DEFENSE.md`, `CONTRIBUTING.md` |

`main.py` is the hotspot — it's one long `chat()` function that every feature
touches. If you're both working in it, say so in the issue thread first and go
sequentially rather than in parallel.

**Assign yourself on the issue** before starting, so the other person can see it's
taken:

```bash
gh issue edit 6 --add-assignee "@me"
```

**Current priorities** are labelled — start here:

```bash
gh issue list --label blocker      # can't install or boot
gh issue list --label security     # access control, XSS
```

---

## Quick reference

```bash
git checkout main && git pull origin main   # sync
git checkout -b fix/12-scholar-gate         # branch
git add -p                                  # stage interactively
git commit -m "fix: ..."                    # commit
git push -u origin fix/12-scholar-gate      # push (first time)
gh pr create --web                          # open PR (loads the template)
gh pr list                                  # what needs review
gh pr checkout 23                           # review PR #23 locally
gh issue list --label blocker               # what to work on
```
