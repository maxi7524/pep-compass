# PepCompass MCP And Scripts Plan

This file records the intended project automation without requiring custom MCP implementation yet.

## Current MCP Usage

Use the built-in Cursor MCP servers when useful:

- `cursor-app-control`: move the agent between the PepCompass and thesis roots, open files/resources, rename chats, and open automations.
- `cursor-ide-browser`: browser-based verification only when a task explicitly needs UI or web interaction.

Before calling any MCP tool, read its descriptor under the conversation MCP folder and follow its schema exactly.

## When To Move Workspace Roots

Use `cursor-app-control.move_agent_to_root` when the active work should happen in the thesis repository:

- From PepCompass to thesis: `C:/Users/Karol/Desktop/Magisterka/magisterka-karola-txt`.
- From thesis back to PepCompass: `C:/Users/Karol/Desktop/PepCompass/pep-compass`.

Prefer a multi-root workspace when a task needs to edit both analysis code and thesis LaTeX in one pass.

## Candidate Custom MCP

A future `pepcompass-thesis` MCP would be useful if these operations become frequent:

- `read_chapter(chapter)`: return one thesis chapter by canonical name.
- `list_figures()`: list expected and existing thesis figure PDFs.
- `list_labels()`: extract LaTeX labels and references.
- `check_todos_fixmes()`: report `TODO`, `FIXME`, and `\todo{}` entries.
- `branch_status(branch)`: summarize local and Bury branch status.
- `thesis_fact_sheet()`: return current research questions, core findings, datasets, and safe thesis wording.

Until then, the `.cursor/skills/pepcompass-thesis` skill is the source of truth for agent behavior.

## Helper Scripts To Implement Later

### `scripts/dev/check_bury_branches.sh`

Purpose: SSH to Bury and report clean branch state for `kjxpp/mutang_analysis_wip` and `rl_trials`.

Default command to mirror:

```bash
ssh kjurasz@bury.mimuw.edu.pl 'cd /home/kjurasz/pep-compass && git status --short --branch && git branch -a | egrep "kjxpp/mutang_analysis_wip|rl_trials|rl_trails"'
```

### `scripts/dev/worktree_branch.sh`

Purpose: create a clean sibling worktree for branch comparison without switching the dirty working tree.

### `scripts/thesis/run_figures.py`

Purpose: regenerate selected thesis figures with a configurable `THESIS_FIG_DIR`, instead of relying only on the hard-coded local path in `_common.py`.

### `scripts/thesis/check_thesis_consistency.py`

Purpose: scan LaTeX for unresolved `TODO`/`FIXME`, stale RQ numbering, missing figure PDFs, and broken label/reference pairs.

### `scripts/thesis/branch_diff_summary.py`

Purpose: compare `kjxpp/mutang_analysis_wip` and `rl_trials` for thesis-relevant changes in notebooks, figure scripts, generated results, and LaTeX-facing assumptions.
