"""Human-readable environment diffs."""

from __future__ import annotations

from dataclasses import dataclass, field

from checkpoint_plugin.types import EnvironmentState


@dataclass(frozen=True)
class CategoryDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)

    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified)


@dataclass(frozen=True)
class EnvDiff:
    provider_changed: bool
    model_changed: bool
    permission_changed: bool
    memory: CategoryDiff
    mcp_changed: bool
    mcp_configs: CategoryDiff
    mcp_servers: CategoryDiff
    skills: CategoryDiff
    skill_status: CategoryDiff
    plugin_status: CategoryDiff
    settings: CategoryDiff
    project_context: CategoryDiff

    def has_changes(self) -> bool:
        return any(
            [
                self.provider_changed,
                self.model_changed,
                self.permission_changed,
                self.memory.has_changes(),
                self.mcp_changed,
                self.mcp_configs.has_changes(),
                self.mcp_servers.has_changes(),
                self.skills.has_changes(),
                self.skill_status.has_changes(),
                self.plugin_status.has_changes(),
                self.settings.has_changes(),
                self.project_context.has_changes(),
            ]
        )


def diff_environments(current: EnvironmentState, target: EnvironmentState) -> EnvDiff:
    return EnvDiff(
        provider_changed=current.provider != target.provider,
        model_changed=current.model != target.model,
        permission_changed=current.permission_mode != target.permission_mode,
        memory=_diff_maps(current.memory_files, target.memory_files),
        mcp_changed=current.mcp_config != target.mcp_config,
        mcp_configs=_diff_maps(current.mcp_configs, target.mcp_configs),
        mcp_servers=_diff_maps(current.mcp_servers, target.mcp_servers),
        skills=_diff_maps(current.skills, target.skills),
        skill_status=_diff_maps(current.skill_status, target.skill_status),
        plugin_status=_diff_maps(current.plugin_status, target.plugin_status),
        settings=_diff_maps(current.settings, target.settings),
        project_context=_diff_maps(current.project_context, target.project_context),
    )


def render_diff(diff: EnvDiff, current: EnvironmentState, target: EnvironmentState) -> str:
    if not diff.has_changes():
        return "Environment: no changes"

    lines = ["Environment:"]
    if diff.provider_changed:
        lines.append(f"  Provider: {current.provider or '-'} -> {target.provider or '-'}")
    if diff.model_changed:
        lines.append(f"  Model: {current.model or '-'} -> {target.model or '-'}")
    if diff.permission_changed:
        lines.append(f"  Permission: {current.permission_mode or '-'} -> {target.permission_mode or '-'}")
    if diff.mcp_changed:
        lines.append("  MCP config: modified")
    _append_category(lines, "MCP config files", diff.mcp_configs)
    _append_category(lines, "MCP servers", diff.mcp_servers)
    _append_category(lines, "Memory", diff.memory)
    _append_category(lines, "Skills", diff.skills)
    _append_category(lines, "Skill status", diff.skill_status)
    _append_category(lines, "Plugin status", diff.plugin_status)
    _append_category(lines, "Settings", diff.settings)
    _append_category(lines, "Project context", diff.project_context)
    return "\n".join(lines)


def _diff_maps(current: dict[str, str], target: dict[str, str]) -> CategoryDiff:
    current_keys = set(current)
    target_keys = set(target)
    common = current_keys & target_keys
    return CategoryDiff(
        added=sorted(target_keys - current_keys),
        removed=sorted(current_keys - target_keys),
        modified=sorted(key for key in common if current[key] != target[key]),
    )


def _append_category(lines: list[str], label: str, diff: CategoryDiff) -> None:
    if not diff.has_changes():
        return
    total = len(diff.added) + len(diff.removed) + len(diff.modified)
    lines.append(f"  {label} ({total} changes):")
    lines.extend(f"    + {item}" for item in diff.added)
    lines.extend(f"    - {item}" for item in diff.removed)
    lines.extend(f"    ~ {item}" for item in diff.modified)
