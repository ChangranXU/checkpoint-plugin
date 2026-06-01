# checkpoint-plugin

**Turn-boundary checkpointing for Claude Code, Codex, and similar agent CLIs.**

Automatically saves your agent's state at each turn: environment config, project files, and conversation trajectory. Restore to any previous checkpoint with a diff preview and automatic backups, then continue the session in your agent right where it left off.

## Features

- **Automatic checkpoints** at every turn, for both Claude Code and Codex
- **Diff-first resume** — preview environment and filesystem changes before restoring
- **Resume into your agent** — rebuilds a native provider session so you can keep going
- **Forks & subagents** — captures forked threads and subagent runs faithfully
- **Restore in place or into a copy** — leave your current workspace untouched

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
- **Trajectory**: The conversation transcript through that turn

Restoring a checkpoint shows a summary diff and creates backups before modifying files. At the resume prompt, enter `d` to inspect detailed environment and filesystem diffs before choosing whether to proceed. On confirm, the plugin restores your files and writes a native provider session you can reopen with the printed `codex resume` / `claude --resume` command — continuing the conversation from that turn.

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
<session-id>  <session-title>  <source>  [marker]
```

Sessions with `[no capture]` had no sidechain file at capture time (phantom subagent events) and contain metadata only.

Use `checkpoint list --session <session-id>` to list that session's checkpoint turns.

During `checkpoint resume`, the confirmation prompt accepts:

```text
y = restore checkpoint
n = cancel
d = view detailed environment and filesystem diffs
```

The diff viewer opens a terminal UI grouped like the summary diff: `Environment` changes first, then `Filesystem` changes. Use `j`/`k` or arrow keys to move, `Enter` to open the selected diff in a full-window colored view, and `Esc`/`q` to go back.

After confirming with `y`, choose whether to restore in place or restore into a new folder copy. Copy mode duplicates the current workspace to a sibling folder, applies the checkpoint there, and leaves the original workspace untouched. The resumed provider session and checkpoint metadata point at the copied workspace; the command also prints the copied `Workspace:` path.

## Checkpoint Metadata Structure

Each checkpoint stores:

- **metadata.json**: Session metadata including provider, source (startup/fork/resume), parent lineage, timestamps, and file references
- **manifests/**: Turn-by-turn manifests with trajectory offsets, environment snapshots, and filesystem state
- **env-snapshots/**: Environment configuration at each turn (provider settings, memory files, MCP config)
- **fs-snapshots/**: Compressed project filesystem at each turn
- **trajectory reference**: Points to the raw provider transcript file with byte offsets for each turn

**Fork and resume sessions**: Metadata includes `forked_from_transcript` path and `forked_at_offset` pointing to the parent session's fork point. Manifests reference the fork session's own file (which includes inherited content), while metadata offsets reference the parent file.

**Subagent sessions**: Metadata includes `parent_session_id` linking to the parent session. Subagents with `capture_status='no_sidechain_file'` indicate the sidechain file was not found at capture time.

## Known Limitations

### FORK-TRUNCATION

When forking a session, the fork offset is captured when the child session's `SessionStart` hook fires. However, the parent session may still be writing or flushing data at that moment. This race condition can cause the captured `forked_at_offset` to exceed the parent file's actual size by 36-52KB (observed range).

**Impact**: Fork sessions may reference byte ranges that don't exist in the parent file at capture time. This can cause resume failures or incorrect trajectory reconstruction for deep fork chains.

**Workaround**: None currently. This is a known limitation with no planned fix due to the structural timing of provider hooks.

**Detection**: The plugin does not currently validate that `forked_at_offset <= parent_file_size`. You may see this issue when resuming from forks of forks.

### MANIFEST-OFFSET Behavior

Checkpoint manifests store trajectory offsets relative to the fork session's own file (which includes inherited content from the parent). The metadata `forked_at_offset` field references the parent file's offset. This is expected behavior:

- **Manifest offsets**: Used for reading the fork session's trajectory file
- **Metadata offsets**: Used for tracking lineage and parent relationships

Both are needed for different purposes and the difference is not a bug.

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

If resuming from a fork fails with trajectory errors, you may be hitting the FORK-TRUNCATION limitation (see above). Try resuming from an earlier turn in the fork chain, or from the original parent session.

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
