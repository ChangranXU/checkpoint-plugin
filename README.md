# checkpoint-plugin

**Turn-boundary checkpointing for Claude Code, Codex, and similar agent CLIs.**

Automatically saves your agent's state at each turn: environment config, project files, and conversation trajectory. Restore to any previous checkpoint with a diff preview and automatic backups.

## Quick Start

```bash
# Install
pip install -e .

# Set up hooks (auto-configures Claude Code and Codex)
checkpoint hooks install

# Restart your agent, then verify
checkpoint list
```

## How It Works

Checkpoints capture three things at each turn:

- **Environment**: Provider settings, memory files, MCP config, skills
- **Filesystem**: All project files (respects `.gitignore` and excludes `.git`, `node_modules`, `.env`*, etc.)
- **Trajectory**: Append-only turn history in JSONL format

Restoring a checkpoint shows a diff and creates backups before modifying files.

## Common Commands

```bash
# Manual checkpoint (automatic via hooks in normal use)
checkpoint save --session <session-id> --note "description"

# List all checkpoints
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

**Manual hook setup**: See `[integrations/settings.example.json](integrations/settings.example.json)` (Claude Code) or `[integrations/codex-settings.example.json](integrations/codex-settings.example.json)` (Codex)

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

See `[src/checkpoint_plugin/](src/checkpoint_plugin/)` for architecture details.