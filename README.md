# checkpoint-plugin

**Turn-boundary checkpointing for Claude Code, Codex, OpenCode, and similar agent CLIs.**

Automatically saves your agent's state at each turn: **environment config, project files, and conversation trajectory**. Restore to any previous checkpoint with a diff preview and automatic backups, then continue the session in your agent right where it left off.

## Quick Start

```bash
# Install
pip install -e .

# Set up hooks (auto-configures Claude Code, Codex, and OpenCode)
checkpoint hooks install

# Restart your agent, then verify
checkpoint
```

## Features

- **Automatic checkpoints** at every turn, for Claude Code, Codex, and OpenCode
- **Diff-first resume** — preview environment and filesystem changes before restoring
- **Resume into your agent** — rebuilds a native provider session so you can keep going
- **Forks & subagents** — captures forked threads and subagent runs faithfully
- **Restore in place or into a copy** — leave your current workspace untouched

## Configuration

**Storage location**: `~/.checkpoint-plugin/` (override with `CHECKPOINT_PLUGIN_HOME`)

**Hook management**:

```bash
checkpoint hooks install            # All providers (Claude Code, Codex, and OpenCode)
checkpoint hooks install claude     # Claude Code only
checkpoint hooks install codex      # Codex only
checkpoint hooks install opencode   # OpenCode only
checkpoint hooks uninstall          # remove all hooks
checkpoint hooks uninstall claude   # remove Claude Code hooks only
checkpoint hooks uninstall codex    # remove Codex hooks only
checkpoint hooks uninstall opencode # remove OpenCode plugin
```

**Manual hook setup**: See [integrations/settings.example.json](integrations/settings.example.json) (Claude Code), [integrations/codex-settings.example.json](integrations/codex-settings.example.json) (Codex), or [integrations/opencode-plugin.example.ts](integrations/opencode-plugin.example.ts) (OpenCode)

## How It Works

Checkpoints capture three things at each turn:

- **Environment**: Provider settings (model, permission mode, collaboration mode), memory files, MCP config, skills
- **Filesystem**: All project files (respects `.gitignore` and excludes `.git`, `node_modules`, `.env`*, etc.)
- **Trajectory**: The conversation transcript through that turn

Restoring a checkpoint shows a summary diff and creates backups before modifying files. At the resume prompt, enter `d` to inspect detailed environment and filesystem diffs before choosing whether to proceed. On confirm, the plugin restores your files and writes a native provider session you can reopen with the printed `codex resume` / `claude --resume` command — continuing the conversation from that turn.

## Common Commands

```bash
# Open the interactive session browser
checkpoint

# List sessions (hides empty sessions by default)
checkpoint list
checkpoint list --all                    # show all sessions including empty ones
checkpoint list --session <session-id>   # list turns for a specific session

# Inspect a checkpoint
checkpoint show <session-id>             # show session overview with all turns
checkpoint show <session-id> <turn>      # show specific turn details
checkpoint show <session-id> --metadata-only  # quick metadata check
checkpoint diff <session-id> <turn>      # preview restore changes

# Restore (shows diff + confirmation prompt)
checkpoint resume <session-id> <turn>
checkpoint resume <session-id> <turn> --yes  # skip confirmation

# Cleanup
checkpoint clean --empty                 # remove empty/incomplete sessions
checkpoint clean --empty --dry-run       # preview what would be removed
checkpoint clean --keep-last 100         # keep only last N turns per session

# Manual checkpoint (automatic via hooks in normal use)
checkpoint save --session <session-id> --note "description"

# View or modify config
checkpoint config get .
checkpoint config set key value
```

### Interactive browser

`checkpoint` (no arguments) opens a terminal session browser grouped by provider. Navigate with arrow keys or vim bindings (`h`/`j`/`k`/`l`), `Enter` to expand, `/` to run a command (`/show`, `/diff`, `/resume`, `/quit`). `r` and `d` are shortcuts for resume and diff. When output is not a terminal, it prints a plain-text tree.

### Resume workflow

`checkpoint resume` shows a summary diff and prompts:

```text
y = restore checkpoint
n = cancel
d = view detailed environment and filesystem diffs
```

On confirm, choose to restore in place or into a new folder copy. Copy mode leaves your current workspace untouched.

## Storage Layout

Checkpoints live in `~/.checkpoint-plugin/sessions/<session-id>/` with:

- `metadata.json` — session metadata (provider, source, parent lineage, timestamps)
- `manifests/` — per-turn state (trajectory offsets, environment, filesystem)
- `env-snapshots/` — environment config at each turn
- `fs-snapshots/` — compressed project filesystem at each turn
- `blobs/` — content-addressed trajectory data, deduplicated across sessions

Fork/resume sessions store `fork_point_trajectory_ref` blobs to survive parent transcript rewrites.

## Troubleshooting

**Empty checkpoint list?**

1. Run `checkpoint hooks install` and restart your agent
2. Start a new session and send a prompt
3. Verify with `checkpoint list`

**Subagent capture incomplete?**

A subagent's closing record sometimes flushes just after the plugin reads its transcript. The plugin handles this two ways: it waits briefly at capture time (up to 1.0s), and — if the record still lands late — it recovers the trailing bytes automatically the next time the checkpoint is read (`show`, `diff`, or `resume`). So a late flush is no longer lost.

To disable the capture-time wait (the read-time recovery still backstops it):

```bash
export CHECKPOINT_SIDECHAIN_SETTLE_TIMEOUT=0
```

**Fork resume failures?**

Fork sessions preserve their fork-point trajectory at capture time via `fork_point_trajectory_ref` blobs. If the parent transcript is rewritten after forking (e.g., due to edit-send rollback), the plugin automatically recovers from the stored blob. If you encounter trajectory errors, verify the checkpoint was captured with the current version (checkpoints created before commit 56351f8 may lack this recovery feature).

## Development

```bash
# Run tests
pytest tests

# Run from source (no install)
PYTHONPATH=src python3 -m checkpoint_plugin.cli --help

# Uninstall
pip uninstall checkpoint-plugin
```

## Extending

- **New providers**: Add to `src/checkpoint_plugin/env/providers.py`
- **New integrations**: Create adapter in `src/checkpoint_plugin/integrations/` that calls `CheckpointCoordinator`

See [src/checkpoint_plugin/](src/checkpoint_plugin/) for architecture details.