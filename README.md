# checkpoint-plugin

**Turn-boundary checkpointing for Claude Code, Codex, OpenCode, and similar agent CLIs.**

Automatically saves your agent's state at each turn: **environment config, project files, and conversation trajectory**. Restore to any previous checkpoint with a diff preview and automatic backups, then continue the session in your agent right where it left off.

## Features

- **Automatic checkpoints** at every turn, for Claude Code, Codex, and OpenCode
- **Diff-first resume** — preview environment and filesystem changes before restoring
- **Resume into your agent** — rebuilds a native provider session so you can keep going
- **Forks & subagents** — captures forked threads and subagent runs faithfully
- **Restore in place or into a copy** — leave your current workspace untouched

## Quick Start

```bash
# Install
pip install -e .

# Set up hooks (auto-configures Claude Code, Codex, and OpenCode)
checkpoint hooks install

# Restart your agent, then verify
checkpoint
```

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

**OpenCode Note**: OpenCode uses a TypeScript plugin system instead of JSON hooks. The installer copies a plugin to `~/.config/opencode/plugins/checkpoint.ts`. **Restart OpenCode after installation**. See [OPENCODE_INTEGRATION.md](OPENCODE_INTEGRATION.md) for detailed setup and troubleshooting.

**Manual hook setup**: See [integrations/settings.example.json](integrations/settings.example.json) (Claude Code), [integrations/codex-settings.example.json](integrations/codex-settings.example.json) (Codex), or [integrations/opencode-settings.example.json](integrations/opencode-settings.example.json) (OpenCode)

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

`checkpoint` opens a terminal session browser grouped by provider. **Empty sessions are hidden by default** to keep the view clean. Use left/right or `h`/`l` to switch provider tabs, `j`/`k` or arrows to move, `PageUp`/`PageDown` to scroll, and `Enter` to expand a session or focus a turn. The tree shows session lineage, fork/resume branches under the parent turn where they split, subagents under the parent turn that spawned them, and each session's checkpoint timeline. Press `/` to run a command on the selected row:

```text
/show      show checkpoint metadata
/diff      preview restore changes
/resume    show the checkpoint resume command to run outside the browser
/quit      exit
```

`r` and `d` are shortcuts for resume and diff on the selected turn. Resume is
offered only for valid parent-session checkpoint turns. Commands render in an
inline output pane and keep the browser open; `/resume` prints a copyable
`checkpoint resume <session-id> <turn>` command instead of restoring inline. Once
command output is visible, `PageUp`/`PageDown` scrolls the output pane. When
output is not a terminal, `checkpoint` prints the same provider/session/turn tree
in a plain text form.

`checkpoint list` shows one row per session and **hides empty sessions by default** (those with no captured data). Use `--all` to see everything including empty sessions.

```text
<session-id>  <session-title>  <source>  [marker]
```

Sessions with `[no capture]` had no sidechain file at capture time and contain metadata only. Use `checkpoint clean --empty` to remove these automatically.

Use `checkpoint list --session <session-id>` to list that session's checkpoint turns. Turns marked with `[replaced by turn N]` were superseded by an edit-send operation. Both the replaced and replacement turns are valid resume points — the replaced turn restores the pre-edit state, while the replacement restores the post-edit state.

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
- **trajectory reference**: Points to the raw provider transcript file with byte offsets for each turn. Fork and resume sessions also store `fork_point_trajectory_ref` blobs containing the inherited trajectory to handle parent transcript rewrites.
- **blobs/**: Content-addressed storage for trajectory data, deduplicated across sessions

All JSON records use compact formatting (`separators=(",", ":")`) to match native provider output byte-for-byte.

**Fork and resume sessions**: Metadata includes `forked_from_transcript` path and `forked_at_offset` pointing to the parent session's fork point. The fork session writes its own transcript file containing both inherited and new content. Additionally, `fork_point_trajectory_ref` stores the inherited trajectory as a content-addressed blob to handle cases where the parent transcript is rewritten post-fork (e.g., due to edit-send rollback or compaction).

**Subagent sessions**: Metadata includes `parent_session_id` linking to the parent session. Subagents with `capture_status='no_sidechain_file'` indicate the sidechain file was not found at capture time.

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