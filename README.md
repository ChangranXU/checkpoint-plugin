# checkpoint-plugin

**Turn-boundary checkpointing for Claude Code, Codex, OpenCode, and similar agent CLIs.**

Automatically saves your agent's state at each turn: **provider environment, project files, and conversation trajectory**. Resume to any checkpoint with a diff preview and automatic backups, then open a native resumed session with an isolated provider home. Content-addressed storage with global deduplication across all sessions.

## Quick Start

```bash
# Install
pip install -e .

# Set up hooks (automatically detects and installs for all available providers)
checkpoint hooks install

# Restart your agent and use it normally
# Checkpoints are captured automatically at each turn

# List your checkpoints
checkpoint list

# Open the interactive browser
checkpoint
```

## Features

- **Automatic turn checkpoints** — Hooks for Claude Code, Codex, and OpenCode capture state on session start, turn end, and subagent completion.
- **Content-addressed storage** — SHA-256 blob storage with global deduplication across all sessions saves significant disk space.
- **Complete state snapshots** — Each checkpoint captures environment (provider config, MCP servers, memory files, model settings), filesystem (project files with .gitignore awareness), and conversation trajectory.
- **Diff-first workflow** — Preview environment and filesystem changes before restoring anything.
- **Safe restore with backups** — Changed files are automatically backed up before restoration; choose to restore in-place or to a new workspace copy.
- **Native resumed sessions** — Generates provider-native session artifacts and isolated config directories for seamless continuation.
- **Fork and lineage tracking** — Preserves fork/resume metadata and captures subagent runs as linked sessions.
- **Security-aware** — Skips credential files (.env, auth.json) and redacts secret-like values from config before storage.

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

Claude Code and Codex are installed by editing JSON hook files:

- Claude Code: `~/.claude/settings.json`
- Codex: `$CODEX_HOME/hooks.json` or `~/.codex/hooks.json`

OpenCode is installed by copying the TypeScript plugin to `$OPENCODE_HOME/plugins/checkpoint.ts` or `~/.config/opencode/plugins/checkpoint.ts`.

**Manual hook setup**: See [integrations/settings.example.json](integrations/settings.example.json) (Claude Code), [integrations/codex-settings.example.json](integrations/codex-settings.example.json) (Codex), or [integrations/opencode-plugin.example.ts](integrations/opencode-plugin.example.ts) (OpenCode).

## How It Works

Each turn, the plugin captures:

- **Environment** — provider name, model, permission/mode/effort hints, memory files, MCP config/status, skills, plugin metadata, settings, and project context.
- **Filesystem** — project files below the working directory, excluding `.git`, `node_modules`, virtualenv/build output, `.env*`, credential-like files, files over 10 MB, configured `exclude_patterns`, and ignored `.gitignore` entries.
- **Trajectory** — provider transcript byte ranges for Claude Code and Codex; OpenCode hook payloads and raw message data are stored in the turn metadata so resume can build an import file.

Manifests are written under a per-session file lock. Content is stored as SHA-256 blobs in a global store shared across all sessions, so identical file contents are deduplicated globally—saving significant disk space when the same files appear in multiple checkpoints.

On restore, the plugin shows a summary diff (enter `d` for detailed diffs), creates backups of any files that will be changed, restores the workspace and provider environment to the selected target, and writes a native provider session plus a `resume-open.json` launcher spec. Launch the restored session with `checkpoint resume-open <new-session-id>`.

## Common Commands

```bash
# Interactive browser - navigate sessions visually
checkpoint

# List all captured sessions
checkpoint list
checkpoint list --all                    # include empty/incomplete sessions
checkpoint list --session <session-id>   # show turns for a specific session

# Inspect checkpoints
checkpoint show <session-id>                  # overview with all turns
checkpoint show <session-id> <turn>           # specific turn details
checkpoint show <session-id> --metadata-only  # quick metadata check
checkpoint diff <session-id> <turn>           # preview restore changes

# Resume to a checkpoint
checkpoint resume <session-id> <turn>
checkpoint resume <session-id> <turn> --yes      # skip confirmation
checkpoint resume <session-id> <turn> --target /path/to/workspace

# Launch the restored session
checkpoint resume-open <new-session-id>

# Cleanup and maintenance
checkpoint clean --empty                 # remove empty/incomplete sessions
checkpoint clean --empty --dry-run       # preview removal without changes
checkpoint clean --keep-last 100         # keep only last N turns per session
checkpoint clean --blobs                 # migrate legacy per-session blobs to global store

# Manual checkpoint (normally automatic via hooks)
checkpoint save --session <session-id> --note "description"

# Configuration
checkpoint config get .                  # view all config
checkpoint config set key value          # update config value
```

### Interactive browser

Launch with `checkpoint` (no arguments) to open a terminal-based session browser. Sessions are grouped by provider with a visual tree structure.

**Navigation:**
- Arrow keys or vim bindings (`h`/`j`/`k`/`l`)
- `Enter` to expand/collapse
- `/` to run commands (`/show`, `/diff`, `/resume`, `/quit`)
- `r` resume shortcut, `d` diff shortcut

When stdout is not a terminal, it prints a plain-text tree instead.

### Resume workflow

`checkpoint resume` shows a summary diff of environment and filesystem changes, then prompts:

```text
y = proceed with restore
n = cancel
d = view detailed diffs (environment changes, file-by-file diffs)
```

After confirming, choose:
- **In-place restore** — apply changes to your current workspace (backs up changed files first)
- **Copy mode** — copy workspace to a new location, then apply checkpoint there (original workspace untouched)
- **Direct target** — use `--target /path` to restore directly to a specific location

Resume creates:
- Native provider session artifacts (`.jsonl` transcript, session metadata)
- Isolated provider config in `~/.checkpoint-plugin/env-state/<new-session-id>/`
- `resume-open.json` launcher spec

**Launch the restored session:**
```bash
checkpoint resume-open <new-session-id>
```

This command sets up the isolated environment and runs the appropriate provider command:
- **Codex:** `CODEX_HOME=<env-state>/codex codex resume <session-id>`
- **Claude Code:** `CLAUDE_CONFIG_DIR=<env-state>/claude claude --resume <session-id>`
- **OpenCode:** Imports session data and runs `opencode --session <session-id>`

Your live provider home remains untouched. Auth files (`.env`, `auth.json`, `credentials.json`) are copied from your live home to the isolated env-state when present, but never stored in checkpoint blobs.

## Storage Layout

All checkpoint data lives in `~/.checkpoint-plugin/` (override with `CHECKPOINT_PLUGIN_HOME`):

```
~/.checkpoint-plugin/
├── blobs/                    # Global content-addressed blob store (SHA-256)
│   ├── ab/                   # First 2 chars of SHA used for sharding
│   │   └── ab12cd...         # Blob files named by full SHA-256 hash
│   └── ...
├── sessions/                 # Per-session checkpoint data
│   └── <session-id>/
│       ├── metadata.json     # Session metadata (provider, timestamps, lineage)
│       ├── manifests/        # Per-turn checkpoint manifests
│       │   ├── index.json    # Turn index
│       │   └── turn_*.json   # Individual turn manifests
│       ├── env-snapshots/    # Human-readable environment snapshots
│       ├── trajectory.jsonl  # Legacy/manual trajectory storage
│       ├── resume-open.json  # Restored session launcher (if resumed)
│       └── .checkpoint.lock  # Per-session write lock
├── backups/                  # Automatic backups from resume operations
├── env-state/                # Isolated provider homes for resumed sessions
│   └── <session-id>/
│       ├── claude/           # Isolated Claude Code config
│       ├── codex/            # Isolated Codex config
│       └── opencode/         # Isolated OpenCode config
└── config.json               # Plugin configuration
```

**Key concepts:**
- **Global blob store:** All content (environment JSON, filesystem snapshots, file contents, fork-point trajectories) is stored in `blobs/` using SHA-256 hashes. Identical content across sessions is stored only once, achieving significant space savings.
- **Manifests:** Each turn checkpoint is a lightweight manifest pointing to blobs. Manifests contain `env_ref` (environment snapshot), `fs_ref` (filesystem snapshot), and `trajectory_ref` (conversation slice).
- **Trajectory references:** Store byte ranges into provider transcript files, allowing efficient slicing without copying conversation data.
- **Fork-point trajectories:** For forked/resumed sessions, the parent transcript at fork time is stored as a blob to survive parent transcript rewrites.

## Troubleshooting

**Empty checkpoint list?**

1. Install hooks: `checkpoint hooks install`
2. Restart your agent (Claude Code, Codex, or OpenCode)
3. Start a new conversation and send at least one prompt
4. Verify: `checkpoint list --all`

By default, `checkpoint list` hides empty or incomplete sessions. Use `--all` to show everything, including subagent shells marked `[no capture]` when no sidechain transcript was available.

**Cannot resume a subagent session?**

Subagent checkpoints are captured for audit and history tracking, but they're not standalone entry points—they depend on the parent session's context. When you try to resume a subagent, the plugin refuses and prints the parent session/turn to resume instead.

**Hooks not capturing?**

Check that hooks are properly installed:
```bash
# Claude Code
cat ~/.claude/settings.json | grep checkpoint

# Codex
cat ~/.codex/hooks.json | grep checkpoint

# OpenCode
ls ~/.config/opencode/plugins/checkpoint.ts
```

If hooks are present but not capturing, check permissions and restart your agent.

**Large blob storage?**

The global blob store deduplicates content across all sessions. To see storage breakdown:
```bash
du -sh ~/.checkpoint-plugin/blobs
checkpoint list --all  # See how many sessions you have
```

Clean up old sessions to reclaim space:
```bash
checkpoint clean --empty              # Remove incomplete sessions
checkpoint clean --keep-last 50       # Keep only last 50 turns per session
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests (256 tests covering all major functionality)
pytest tests
pytest tests -v              # verbose output
pytest tests -k test_name    # run specific test

# Run from source without installation
PYTHONPATH=src python3 -m checkpoint_plugin.cli --help

# Type checking and linting
mypy src                     # type checking (if mypy is installed)
ruff check src               # linting (if ruff is installed)

# Uninstall
pip uninstall checkpoint-plugin
```

## Extending

The plugin is designed to be extensible for new providers and integrations:

**Adding a new provider:**
1. Add provider detection logic to `src/checkpoint_plugin/env/providers.py`
2. Define `ProviderLayout` (config paths, home directory)
3. Define `ProviderResumePolicy` (how to launch resumed sessions)
4. Implement environment collection in `src/checkpoint_plugin/env/collector.py`

**Adding a new integration:**
1. Create an adapter in `src/checkpoint_plugin/integrations/`
2. Hook into provider lifecycle events (session start, turn end)
3. Call `CheckpointCoordinator.on_session_start()` and `CheckpointCoordinator.on_turn_end()`
4. Pass `TrajectoryReference` for conversation slicing

**Architecture overview:**
- `coordinator.py` — Main checkpoint orchestration and session lifecycle
- `store.py` — Content-addressed blob storage with SHA-256 deduplication
- `resume.py` — Diff computation and checkpoint restoration
- `env/` — Provider environment capture, diffing, and restoration
- `fs/` — Filesystem snapshot, diffing, and restoration
- `integrations/` — Provider-specific hooks (Claude Code, Codex, OpenCode)
- `ui/` — Terminal-based session browser and diff viewer

See source code in `src/checkpoint_plugin/` for implementation details.
