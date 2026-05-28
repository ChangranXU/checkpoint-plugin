# checkpoint-plugin

**Turn-boundary checkpointing for Claude Code, Codex, and similar agent CLIs.**

Automatically saves your agent's state at each turn: environment config, project files, and conversation trajectory. Restore to any previous checkpoint with a diff preview and automatic backups.

## TODO

- [ ] Checkpoint modify-and-resume
- [ ] Other provider support
- [ ] Verify on Claude Code (very soon)

## Quick Start

```bash
# Install
pip install -e .

# Set up hooks (auto-configures Claude Code and Codex)
checkpoint hooks install

# Restart your agent, then verify
checkpoint list
```

## Configuration

**Storage location**: `~/.checkpoint-plugin/` (override with `CHECKPOINT_PLUGIN_HOME`)

**Hook management**:

```bash
checkpoint hooks install          # both Claude Code and Codex
checkpoint hooks install claude   # Claude Code only
checkpoint hooks install codex    # Codex only
checkpoint hooks uninstall        # remove all hooks
checkpoint hooks uninstall claude # remove Claude Code hooks only
checkpoint hooks uninstall codex  # remove Codex hooks only
```

**Manual hook setup**: See [integrations/settings.example.json](integrations/settings.example.json) (Claude Code) or [integrations/codex-settings.example.json](integrations/codex-settings.example.json) (Codex)

## How It Works

Checkpoints capture three things at each turn:

- **Environment**: Provider settings, memory files, MCP config, skills
- **Filesystem**: All project files (respects `.gitignore` and excludes `.git`, `node_modules`, `.env`*, etc.)
- **Trajectory**: Append-only turn history in JSONL format

Restoring a checkpoint shows a summary diff and creates backups before modifying files. At the resume prompt, enter `d` to inspect detailed environment and filesystem diffs before choosing whether to proceed.

## Common Commands

```bash
# Manual checkpoint (automatic via hooks in normal use)
checkpoint save --session <session-id> --note "description"

# List sessions, then list turns for one session
checkpoint list
checkpoint list --session <session-id>

# Inspect a checkpoint
checkpoint show <session-id> <turn>
checkpoint diff <session-id> <turn>

# Restore (shows diff + confirmation prompt)
checkpoint resume <session-id> <turn>
checkpoint resume <session-id> <turn> --yes  # skip confirmation

# Cleanup old checkpoints
checkpoint clean --keep-last 100

# View or modify config
checkpoint config get .
checkpoint config set . key value
```

`checkpoint list` shows one row per session:

```text
<session-id>  <session-title>  <source>
```

Use `checkpoint list --session <session-id>` to list that session's checkpoint turns.

During `checkpoint resume`, the confirmation prompt accepts:

```text
y = restore checkpoint
n = cancel
d = view detailed environment and filesystem diffs
```

The diff viewer opens a terminal UI grouped like the summary diff: `Environment` changes first, then `Filesystem` changes. Use `j`/`k` or arrow keys to move, `Enter` to open the selected diff in a full-window colored view, and `Esc`/`q` to go back.

After confirming with `y`, choose whether to restore in place or restore into a new folder copy. Copy mode duplicates the current workspace to a sibling folder, applies the checkpoint there, and leaves the original workspace untouched. The resumed provider session and checkpoint metadata point at the copied workspace; the command also prints the copied `Workspace:` path.

## Troubleshooting

**Empty checkpoint list?**

1. Run `checkpoint hooks install` and restart your agent
2. Start a new session and send a prompt
3. Verify with `checkpoint list`

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
