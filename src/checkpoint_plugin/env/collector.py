"""Collect provider environment state into checkpoint blobs."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Iterable

from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import EnvironmentState

from .providers import ProviderLayout

# Credential-shaped files are never copied into checkpoint blobs. Re-auth is the
# correct resume path; a checkpoint must not become a secrets store. Mirrors the
# fs-snapshot SECRET_PATTERNS but applied to the basename of every env file.
SECRET_BASENAME_PATTERNS = (
    "auth.json",
    ".env",
    ".env*",
    "*credential*",
    "*.pem",
    "*.key",
    "*.secret",
    "*token*",
)

# Config files (config.toml, settings.json, .mcp.json, ...) are kept verbatim for
# faithful restore, but they can embed secret material inline (e.g. Codex
# `experimental_bearer_token`). We redact the VALUE of any secret-shaped key
# before storing, gated to structured config so source/markdown is never altered.
_REDACTABLE_SUFFIXES = (".toml", ".json")
_SECRET_VALUE_KEY_PATTERNS = (
    "*token*",
    "*secret*",
    "*password*",
    "*passwd*",
    "*credential*",
    "*bearer*",
    "*api_key*",
    "*apikey*",
    "*access_key*",
    "*private_key*",
    "trusted_hash",
)
_REDACTED = '"***redacted***"'
# Matches `key = "..."` (TOML) and `"key": "..."` (JSON), preserving the key and
# separator so only the quoted value is replaced. Bare/numeric values are left
# alone — inline secrets are quoted strings in practice.
_SECRET_ASSIGNMENT = re.compile(
    r'(?P<prefix>(?P<q>["\']?)(?P<key>[\w.-]+)(?P=q)\s*[:=]\s*)'
    r'(?P<val>"(?:[^"\\]|\\.)*"|\'[^\']*\')'
)


def _is_secret_path(path: Path) -> bool:
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_BASENAME_PATTERNS)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in _SECRET_VALUE_KEY_PATTERNS)


def _redact_secret_values(data: bytes) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data

    def _replace(match: re.Match[str]) -> str:
        if _is_secret_key(match.group("key")):
            return match.group("prefix") + _REDACTED
        return match.group(0)

    return _SECRET_ASSIGNMENT.sub(_replace, text).encode("utf-8")


def _read_blob_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in _REDACTABLE_SUFFIXES:
        return _redact_secret_values(data)
    return data


def collect_environment(
    cwd: Path,
    provider: ProviderLayout,
    store: CheckpointStore,
) -> EnvironmentState:
    cwd = cwd.expanduser().resolve()
    skill_roots = _skill_roots(provider, cwd)
    skills = _collect_named_trees(skill_roots, store, follow_symlink_dirs=True)
    # Some provider fields (notably Claude's model) are only delivered to the
    # SessionStart hook, which runs in a separate process from Stop. on_session_start
    # persists them to metadata.json; we fall back to that when the live env var is
    # absent so the captured state still pins the model/agent/effort.
    session_env = _session_env_fallback(store)
    return EnvironmentState(
        provider=provider.name,
        model=_first_env("ANTHROPIC_MODEL", "CLAUDE_MODEL", "OPENAI_MODEL", "CODEX_MODEL", "OPENCODE_MODEL")
        or session_env.get("model"),
        permission_mode=_first_env("CLAUDE_PERMISSION_MODE", "CODEX_PERMISSION_MODE", "CODEX_SANDBOX_MODE", "OPENCODE_PERMISSION_MODE")
        or session_env.get("permission_mode"),
        mode=_first_env("CLAUDE_MODE", "CODEX_MODE", "OPENCODE_MODE") or session_env.get("mode"),
        effort=_first_env("CLAUDE_EFFORT", "OPENCODE_EFFORT") or session_env.get("effort") or _codex_effort(provider, cwd),
        agent_type=_first_env("CLAUDE_AGENT_TYPE", "CODEX_AGENT_TYPE", "OPENCODE_AGENT_TYPE") or session_env.get("agent_type"),
        memory_files=_collect_tree(provider.memory_dir, store),
        mcp_config=_store_file(provider.mcp_config, store),
        mcp_configs=_collect_named_files(_mcp_config_files(provider, cwd), store),
        mcp_servers=_collect_mcp_servers(provider, cwd),
        skills=skills,
        skill_status=_collect_skill_status(provider, cwd, skills),
        plugin_status=_collect_plugin_status(provider, cwd),
        settings=_collect_settings(provider.settings_files, store),
        project_context=_collect_project_context(cwd, provider.project_files, store),
        extra={
            "provider_home": str(provider.home),
            "cwd": str(cwd),
            "skill_symlinks": _collect_named_symlinks(skill_roots),
            **_codex_history_extra(provider, store),
        },
    )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _session_env_fallback(store: CheckpointStore) -> dict[str, str]:
    """Provider hints captured at SessionStart (e.g. Claude's model)."""
    metadata_path = store.session_dir / "metadata.json"
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    session_env = data.get("session_env") if isinstance(data, dict) else None
    if not isinstance(session_env, dict):
        return {}
    return {str(key): str(value) for key, value in session_env.items() if value}


def _collect_mcp_servers(provider: ProviderLayout, cwd: Path) -> dict[str, str]:
    if provider.name == "codex":
        servers: dict[str, str] = {}
        for config in _codex_configs(provider, cwd):
            servers.update(
                {
                    str(name): _status_from_config(server_config)
                    for name, server_config in (config.get("mcp_servers") or {}).items()
                }
            )
        for config in _mcp_json_configs(provider, cwd):
            servers.update({str(name): _status_from_config(value) for name, value in _json_mcp_servers(config).items()})
        return servers
    if provider.name == "claude":
        config = _load_json(provider.home.parent / ".claude.json")
        servers = {str(name): "active" for name in (config.get("mcpServers") or {})}
        project = _nearest_project_config(config, cwd)
        if isinstance(project, dict):
            for name in project.get("mcpServers") or {}:
                servers[str(name)] = "active"
            for name in project.get("enabledMcpjsonServers") or []:
                servers[str(name)] = "active"
            for name in project.get("disabledMcpjsonServers") or []:
                servers[str(name)] = "inactive"
        return servers
    return {}


def _collect_skill_status(provider: ProviderLayout, cwd: Path, skills: dict[str, str]) -> dict[str, str]:
    status = {name: "present" for name in _skill_names_from_files(skills)}
    if provider.name == "codex":
        for config in _codex_configs(provider, cwd):
            for item in (config.get("skills") or {}).get("config") or []:
                if not isinstance(item, dict):
                    continue
                name = _skill_name_from_config(item)
                enabled = item.get("enabled")
                if not name or not isinstance(enabled, bool):
                    continue
                status[name] = "active" if enabled else "inactive"
    elif provider.name == "claude":
        config = _load_json(provider.home.parent / ".claude.json")
        for name, value in _claude_skill_overrides(config).items():
            if isinstance(value, bool):
                status[str(name)] = "active" if value else "inactive"
    return dict(sorted(status.items()))


def _skill_names_from_files(skills: dict[str, str]) -> set[str]:
    names: set[str] = set()
    for rel_path in skills:
        path = PurePosixPath(rel_path)
        if path.name == "SKILL.md" and path.parent.name:
            names.add(path.parent.name)
    return names


def _skill_roots(provider: ProviderLayout, cwd: Path) -> dict[str, Path]:
    roots = dict(provider.skills_dirs)
    roots.update(_plugin_skill_roots(provider))
    if provider.name == "claude":
        for project_root in _ancestor_chain(_nearest_project_root(cwd), cwd):
            roots[f"project:{project_root}:.claude/skills"] = project_root / ".claude" / "skills"
    if provider.name == "codex":
        for project_root in _ancestor_chain(_nearest_project_root(cwd), cwd):
            roots[f"project:{project_root}:.codex/skills"] = project_root / ".codex" / "skills"
            roots[f"project:{project_root}:.agents/skills"] = project_root / ".agents" / "skills"
    return roots


def _plugin_skill_roots(provider: ProviderLayout) -> dict[str, Path]:
    if provider.name != "codex":
        return {}

    cache_root = provider.home / "plugins" / "cache"
    if not cache_root.exists() or not cache_root.is_dir():
        return {}

    roots: dict[str, Path] = {}
    for skills_dir in sorted(cache_root.glob("*/*/*/skills")):
        if not skills_dir.is_dir():
            continue
        try:
            rel = skills_dir.relative_to(cache_root)
        except ValueError:
            continue
        marketplace, plugin, version, _skills = rel.parts
        roots[f"plugin:{marketplace}:{plugin}:{version}"] = skills_dir
    return roots


def _mcp_config_files(provider: ProviderLayout, cwd: Path) -> dict[str, Path]:
    files = {path.name: path for path in provider.mcp_config_files}
    for project_root in _ancestor_chain(_nearest_project_root(cwd), cwd):
        path = project_root / ".mcp.json"
        files[f"project:{project_root}:.mcp.json"] = path
        if provider.name == "codex":
            files[f"project:{project_root}:.codex/config.toml"] = project_root / ".codex" / "config.toml"
    return files


def _codex_configs(provider: ProviderLayout, cwd: Path) -> list[dict[str, object]]:
    configs = [_load_toml(provider.home / "config.toml")]
    for project_root in _ancestor_chain(_nearest_project_root(cwd), cwd):
        configs.append(_load_toml(project_root / ".codex" / "config.toml"))
    return [config for config in configs if config]


def _codex_history_extra(provider: ProviderLayout, store: CheckpointStore) -> dict[str, str]:
    """Capture Codex prompt history (`history.jsonl`) by content hash (G3).

    Cross-session prompt recall lives here; like ~/.claude.json it's global state
    we record for drift visibility but do NOT restore wholesale (doing so would
    rewrite unrelated sessions' history). Stored as a deduped blob sha.
    """
    if provider.name != "codex":
        return {}
    path = provider.home / "history.jsonl"
    if not path.is_file() or _is_secret_path(path):
        return {}
    try:
        data = path.read_bytes()
    except OSError:
        return {}
    return {"codex_history_ref": store.store_blob(data)}


def _codex_effort(provider: ProviderLayout, cwd: Path) -> str | None:
    """Reasoning effort from Codex config (`model_reasoning_effort`).

    Codex delivers no effort field to hooks; it lives only in config.toml. Pin it
    on EnvironmentState so a resume can flag a drift, mirroring Claude's effort.
    Project-level config wins over home (later in the ancestor chain).
    """
    effort: str | None = None
    for config in _codex_configs(provider, cwd):
        value = config.get("model_reasoning_effort")
        if isinstance(value, str) and value:
            effort = value
    return effort


def _mcp_json_configs(provider: ProviderLayout, cwd: Path) -> list[dict[str, object]]:
    configs = [_load_json(path) for path in provider.mcp_config_files if path.name == ".mcp.json"]
    for project_root in _ancestor_chain(_nearest_project_root(cwd), cwd):
        configs.append(_load_json(project_root / ".mcp.json"))
    return [config for config in configs if config]


def _json_mcp_servers(config: dict[str, object]) -> dict[str, object]:
    servers = config.get("mcpServers")
    if isinstance(servers, dict):
        return servers
    return config


def _claude_skill_overrides(config: dict[str, object]) -> dict[str, object]:
    for key in ("skillOverrides", "skills"):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _enabled_claude_plugins(config: dict[str, object]) -> set[str]:
    enabled: set[str] = set()
    value = config.get("enabledPlugins")
    if isinstance(value, list):
        enabled.update(str(item) for item in value)
    elif isinstance(value, dict):
        enabled.update(str(name) for name, active in value.items() if active)
    return enabled


def _collect_plugin_status(provider: ProviderLayout, cwd: Path) -> dict[str, str]:
    if provider.name == "codex":
        status: dict[str, str] = {}
        for config in _codex_configs(provider, cwd):
            for name, plugin_config in (config.get("plugins") or {}).items():
                status[str(name)] = _status_from_config(plugin_config)
        return dict(sorted(status.items()))
    if provider.name == "claude":
        plugins_dir = provider.home / "plugins" / "marketplaces"
        status = {name: "present" for name in _installed_claude_plugins(plugins_dir)}
        config = _load_json(provider.home.parent / ".claude.json")
        for name in _enabled_claude_plugins(config):
            status[name] = "active"
        return dict(sorted(status.items()))
    return {}


def _status_from_config(config: object) -> str:
    if isinstance(config, dict):
        enabled = config.get("enabled")
        disabled = config.get("disabled")
        if isinstance(enabled, bool):
            return "active" if enabled else "inactive"
        if isinstance(disabled, bool):
            return "inactive" if disabled else "active"
    return "active"


def _skill_name_from_config(item: dict[str, object]) -> str:
    path = item.get("path")
    if isinstance(path, str) and path:
        skill_path = Path(path)
        if skill_path.name == "SKILL.md":
            return skill_path.parent.name
        return skill_path.stem or skill_path.name
    name = item.get("name")
    return str(name) if name else ""


def _installed_claude_plugins(plugins_dir: Path) -> list[str]:
    names: set[str] = set()
    for group in ("plugins", "external_plugins"):
        for path in plugins_dir.glob(f"*/{group}/*"):
            if path.is_dir():
                names.add(path.name)
    return sorted(names)


def _nearest_project_config(config: dict[str, object], cwd: Path) -> dict[str, object] | None:
    projects = config.get("projects")
    if not isinstance(projects, dict):
        return None
    for path in (cwd, *cwd.parents):
        value = projects.get(str(path))
        if isinstance(value, dict):
            return value
    return None


def _load_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_toml(path: Path) -> dict[str, object]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _store_file(path: Path | None, store: CheckpointStore) -> str | None:
    if path is None or not path.exists() or not path.is_file() or _is_secret_path(path):
        return None
    return store.store_blob(_read_blob_bytes(path))


def _collect_named_files(paths: dict[str, Path], store: CheckpointStore) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        sha = _store_file(path, store)
        if sha is not None:
            result[name] = sha
    return result


def _collect_named_trees(
    roots: dict[str, Path],
    store: CheckpointStore,
    *,
    follow_symlink_dirs: bool = False,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, root in sorted(roots.items()):
        for rel, sha in _collect_tree(root, store, follow_symlink_dirs=follow_symlink_dirs).items():
            result[f"{name}/{rel}"] = sha
    return result


def _collect_tree(
    root: Path | None,
    store: CheckpointStore,
    *,
    follow_symlink_dirs: bool = False,
) -> dict[str, str]:
    if root is None or not root.exists() or not root.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in _iter_files(root, follow_symlink_dirs=follow_symlink_dirs):
        if _is_secret_path(path):
            continue
        rel = path.relative_to(root).as_posix()
        result[rel] = store.store_blob(_read_blob_bytes(path))
    return result


def _collect_settings(paths: Iterable[Path], store: CheckpointStore) -> dict[str, str]:
    settings: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file() and not _is_secret_path(path):
            settings[path.name] = store.store_blob(_read_blob_bytes(path))
    return settings


def _collect_project_context(cwd: Path, project_files: list[str], store: CheckpointStore) -> dict[str, str]:
    context: dict[str, str] = {}
    for root in _ancestor_chain(_nearest_project_root(cwd), cwd):
        for rel_name in project_files:
            path = root / rel_name
            if path.exists() and path.is_file() and not _is_secret_path(path):
                key = path.relative_to(root).as_posix()
                context[str(root / key)] = store.store_blob(_read_blob_bytes(path))
            elif path.exists() and path.is_dir():
                for rel, sha in _collect_tree(path, store, follow_symlink_dirs=True).items():
                    key = path.relative_to(root).joinpath(rel).as_posix()
                    context[str(root / key)] = sha
    return context


def _collect_named_symlinks(roots: dict[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, root in sorted(roots.items()):
        for rel, target in _collect_symlinks(root).items():
            result[f"{name}/{rel}"] = target
    return result


def _collect_symlinks(root: Path | None) -> dict[str, str]:
    if root is None or not root.exists() or not root.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            result[path.relative_to(root).as_posix()] = str(path.resolve(strict=False))
    return result


def _iter_files(root: Path, *, follow_symlink_dirs: bool = False) -> Iterable[Path]:
    if not follow_symlink_dirs:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path
        return

    seen_dirs: set[Path] = set()
    yield from _iter_files_following_symlink_dirs(root, seen_dirs)


def _iter_files_following_symlink_dirs(root: Path, seen_dirs: set[Path]) -> Iterable[Path]:
    try:
        resolved = root.resolve(strict=True)
    except OSError:
        return
    if resolved in seen_dirs:
        return
    seen_dirs.add(resolved)

    try:
        children = sorted(root.iterdir())
    except OSError:
        return
    for child in children:
        if child.is_file():
            yield child
        elif child.is_dir():
            yield from _iter_files_following_symlink_dirs(child, seen_dirs)


def _nearest_project_root(cwd: Path) -> Path:
    for path in (cwd, *cwd.parents):
        if (path / ".git").exists():
            return path
    return cwd


def _ancestor_chain(root: Path, cwd: Path) -> list[Path]:
    try:
        relative = cwd.relative_to(root)
    except ValueError:
        return [cwd]

    paths = [root]
    current = root
    for part in relative.parts:
        current = current / part
        paths.append(current)
    return paths


def environment_to_blob(state: EnvironmentState, store: CheckpointStore) -> str:
    return store.store_json_blob(state.to_json())


def environment_from_blob(sha: str, store: CheckpointStore) -> EnvironmentState:
    data = store.load_json_blob(sha)
    if not isinstance(data, dict):
        raise ValueError(f"Environment blob {sha} is not a JSON object")
    return EnvironmentState.from_json(data)
