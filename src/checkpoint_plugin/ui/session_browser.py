"""Interactive checkpoint session browser."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style

from checkpoint_plugin.coordinator import reanchor_last_turn_to_eof
from checkpoint_plugin.paths import sessions_dir
from checkpoint_plugin.resume import ResumeOrchestrator, _parent_turn_for_subagent
from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import CheckpointManifest
from checkpoint_plugin.ui._helpers import format_timestamp, truncate_with_ellipsis
from checkpoint_plugin.ui._rendering import render_tree_row
from checkpoint_plugin.ui._help import render_help_text as _render_help_text


@dataclass(frozen=True)
class BrowserAction:
    command: str
    session_id: str | None = None
    turn_id: int | None = None


@dataclass
class SessionNode:
    session_id: str
    metadata: dict[str, Any]
    manifests: list[CheckpointManifest]
    marker: str = ""
    fork_parent: tuple[str, int | None] | None = None
    subagent_parent: tuple[str, int | None] | None = None
    fork_children: dict[int | None, list["SessionNode"]] = field(default_factory=dict)
    subagent_children: dict[int | None, list["SessionNode"]] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return _text(self.metadata.get("provider")) or _manifest_provider(self.manifests) or "generic"

    @property
    def created_ts(self) -> str:
        if self.metadata.get("start_ts"):
            return str(self.metadata["start_ts"])
        if self.manifests:
            return self.manifests[0].created_ts
        return ""

    @property
    def title(self) -> str:
        return _text(self.metadata.get("session_title")) or "-"

    @property
    def source(self) -> str:
        return _text(self.metadata.get("source")) or "startup"

    @property
    def lineage(self) -> dict[str, Any]:
        value = self.metadata.get("lineage")
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class TreeRow:
    kind: str
    node: SessionNode
    manifest: CheckpointManifest | None
    depth: int
    label: str
    style: str = ""
    expanded: bool = False
    has_children: bool = False


def show_session_browser(
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> BrowserAction | None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    providers = collect_provider_trees()
    if not providers:
        print("No checkpoint sessions found.", file=output_stream)
        return None
    if not input_stream.isatty() or not output_stream.isatty():
        print(render_session_tree(providers), file=output_stream)
        return None
    return _show_tui(providers)


def collect_provider_trees(root: Path | None = None, show_all: bool = False) -> dict[str, list[SessionNode]]:
    root = root or sessions_dir()
    nodes = _load_nodes(root, show_all=show_all)
    _link_nodes(nodes, root)
    attached = {
        child.session_id
        for node in nodes.values()
        for children in [*node.fork_children.values(), *node.subagent_children.values()]
        for child in children
    }
    providers: dict[str, list[SessionNode]] = {}
    for node in nodes.values():
        if node.session_id in attached:
            continue
        providers.setdefault(node.provider, []).append(node)
    for provider_nodes in providers.values():
        provider_nodes.sort(key=_session_sort_key)
    return dict(sorted(providers.items()))


def render_session_tree(providers: dict[str, list[SessionNode]]) -> str:
    lines: list[str] = []
    for provider, nodes in providers.items():
        total_turns = sum(_count_turns(node) for node in nodes)
        lines.append(f"{provider} ({len(nodes)} sessions, {total_turns} turns)")
        rows = _rows_for_nodes(nodes)
        for row in rows:
            # Simple text rendering for non-TTY
            prefix = "  " * row.depth
            marker = "▶" if row.kind == "session" else "├" if row.kind == "link" else "─"
            lines.append(f"  {prefix}{marker} {row.label}")
    return "\n".join(lines)


def _show_tui(providers: dict[str, list[SessionNode]]) -> BrowserAction | None:
    provider_names = list(providers)
    selected_by_provider = {name: 0 for name in provider_names}
    expanded = {node.session_id for nodes in providers.values() for node in _walk_sessions(nodes)}
    state: dict[str, Any] = {
        "provider": 0,
        "mode": "browse",
        "status": "↑↓:move  Enter:toggle  r:resume  d:diff  /:cmd  ?:help  q:quit",
        "action": None,
        "output_title": "Command Output",
        "output_text": "No command output yet.",
        "output_visible": False,
        "output_scroll": 0,
        "tree_scroll": 0,
        "output_height": 8,
        "show_help": False,
    }
    command_buffer = Buffer()

    def current_provider() -> str:
        return provider_names[int(state["provider"])]

    def rows() -> list[TreeRow]:
        return _rows_for_nodes(providers[current_provider()], expanded)

    def selected_row() -> TreeRow | None:
        current_rows = rows()
        if not current_rows:
            return None
        index = min(selected_by_provider[current_provider()], len(current_rows) - 1)
        selected_by_provider[current_provider()] = index
        return current_rows[index]

    header = Window(FormattedTextControl(lambda: _header_fragments(provider_names, providers, state)), height=1)
    body = Window(
        FormattedTextControl(lambda: _body_fragments(rows(), selected_by_provider[current_provider()], state)),
        height=lambda: _body_height(state),
    )
    detail = Window(FormattedTextControl(lambda: _detail_fragments(selected_row(), state)), height=8, wrap_lines=True)
    output = Window(
        FormattedTextControl(lambda: _output_fragments(state)),
        height=lambda: _output_height(state),
        wrap_lines=False,
    )
    help_panel = Window(
        FormattedTextControl(lambda: _help_fragments()),
        height=lambda: _help_height(state),
        wrap_lines=True,
    )
    status = Window(FormattedTextControl(lambda: _status_fragments(state)), height=1)
    command = VSplit(
        [
            Window(FormattedTextControl(lambda: [("class:command", "/" if state["mode"] == "command" else "")]), width=1),
            Window(BufferControl(buffer=command_buffer), height=1),
        ],
        height=1,
    )

    bindings = KeyBindings()
    browse_mode = Condition(lambda: state["mode"] == "browse")

    def invalidate(event) -> None:  # noqa: ANN001
        event.app.invalidate()

    def set_status(text: str, event=None) -> None:  # noqa: ANN001
        state["status"] = text
        if event is not None:
            invalidate(event)

    def set_output(title: str, text: str, event=None) -> None:  # noqa: ANN001
        state["output_title"] = title
        state["output_text"] = text
        state["output_visible"] = True
        state["output_scroll"] = 0
        if event is not None:
            invalidate(event)

    def move_provider(delta: int, event) -> None:  # noqa: ANN001
        state["provider"] = (int(state["provider"]) + delta) % len(provider_names)
        state["tree_scroll"] = 0
        node_count = len(providers[current_provider()])
        turn_count = sum(_count_turns(node) for node in providers[current_provider()])
        set_status(f"Provider: {current_provider()} ({node_count} sessions, {turn_count} turns)", event)

    def move_selection(delta: int, event) -> None:  # noqa: ANN001
        current_rows = rows()
        if not current_rows:
            return
        provider = current_provider()
        old_selection = selected_by_provider[provider]
        selected_by_provider[provider] = max(0, min(selected_by_provider[provider] + delta, len(current_rows) - 1))

        # Only invalidate if selection actually changed
        if old_selection != selected_by_provider[provider]:
            _sync_tree_scroll(state, selected_by_provider[provider], len(current_rows))
            invalidate(event)

    def scroll_output(delta: int, event) -> bool:  # noqa: ANN001
        if not state.get("output_visible") or not _output_can_scroll(state):
            return False
        old_scroll = int(state.get("output_scroll") or 0)
        new_scroll = _clamp_output_scroll(state, old_scroll + delta)
        state["output_scroll"] = new_scroll
        if new_scroll != old_scroll:
            set_status(_output_scroll_status(state), event)
        else:
            invalidate(event)
        return True

    def run_inline_action(action: BrowserAction, event) -> None:  # noqa: ANN001
        if action.session_id is None or action.turn_id is None:
            return
        if action.command == "show":
            title, text = _show_result(action.session_id, action.turn_id)
            set_output(title, text, event)
            set_status("Show result rendered inline.", event)
            return
        if action.command == "diff":
            title, text = _diff_result(action.session_id, action.turn_id)
            set_output(title, text, event)
            set_status("Diff result rendered inline.", event)
            return
        if action.command == "resume":
            set_output(*_resume_hint(action.session_id, action.turn_id), event)
            set_status("Resume command shown. Run it outside the browser to restore.", event)

    def open_resume_for_selection(event) -> None:  # noqa: ANN001
        action = selected_turn_action("resume")
        if action is None or action.session_id is None or action.turn_id is None:
            set_status("Resume is available only for valid parent-session checkpoint turns.", event)
            return
        set_output(*_resume_hint(action.session_id, action.turn_id), event)
        set_status("Resume command shown. Run it outside the browser to restore.", event)

    def selected_turn_action(command: str) -> BrowserAction | None:
        row = selected_row()
        if row is None:
            return None
        if command == "resume" and not _can_resume_row(row):
            return None
        if row.manifest is None:
            return None
        return BrowserAction(command, row.node.session_id, row.manifest.turn_id)

    @bindings.add("right", filter=browse_mode)
    @bindings.add("l", filter=browse_mode)
    def _next_provider(event) -> None:  # noqa: ANN001
        move_provider(1, event)

    @bindings.add("left", filter=browse_mode)
    @bindings.add("h", filter=browse_mode)
    def _previous_provider(event) -> None:  # noqa: ANN001
        move_provider(-1, event)

    @bindings.add("down", filter=browse_mode)
    @bindings.add("j", filter=browse_mode)
    def _down(event) -> None:  # noqa: ANN001
        move_selection(1, event)

    @bindings.add("up", filter=browse_mode)
    @bindings.add("k", filter=browse_mode)
    def _up(event) -> None:  # noqa: ANN001
        move_selection(-1, event)

    @bindings.add("enter")
    def _enter(event) -> None:  # noqa: ANN001
        if state["mode"] == "command":
            action = _command_action(command_buffer.text, selected_row())
            command_buffer.text = ""
            state["mode"] = "browse"
            if action is None:
                set_status("Command unavailable for this row. Select a valid checkpoint turn.", event)
                return
            if action.command == "help":
                set_output("Help", _render_help_text(), event)
                set_status("Help displayed. Use ? or F1 to toggle inline help panel.", event)
                return
            if action.command == "quit":
                event.app.exit(result=None)
                return
            run_inline_action(action, event)
            return
        row = selected_row()
        if row is None:
            return
        if row.kind == "session":
            if row.node.session_id in expanded:
                expanded.remove(row.node.session_id)
            else:
                expanded.add(row.node.session_id)
            invalidate(event)
            return
        if row.manifest is None:
            set_status("Select a checkpoint turn to inspect or resume.", event)
        elif _can_resume_row(row):
            set_status(f"✓ Turn {row.manifest.turn_id} | Resumable | Press: r=resume d=diff Enter=show", event)
        else:
            reason = ""
            if row.node.subagent_parent or row.node.source == "subagent":
                reason = "subagent"
            elif row.node.marker:
                reason = "no capture"
            set_status(f"Turn {row.manifest.turn_id} | Not resumable ({reason}) | Press: d=diff Enter=show", event)

    @bindings.add("/", filter=browse_mode)
    def _command_mode(event) -> None:  # noqa: ANN001
        state["mode"] = "command"
        command_buffer.text = ""
        event.app.layout.focus(command)
        set_status("Type: show | diff | resume | help | quit  (Esc cancels)", event)

    @bindings.add("?", filter=browse_mode)
    @bindings.add("f1", filter=browse_mode)
    def _toggle_help(event) -> None:  # noqa: ANN001
        state["show_help"] = not state.get("show_help", False)
        if state["show_help"]:
            set_status("Help panel shown. Press ? or F1 again to hide.", event)
        else:
            set_status("↑↓:move  Enter:toggle  r:resume  d:diff  /:cmd  ?:help  q:quit", event)
        invalidate(event)

    @bindings.add("escape")
    def _escape(event) -> None:  # noqa: ANN001
        if state["mode"] == "command":
            state["mode"] = "browse"
            command_buffer.text = ""
            event.app.layout.focus(body)
            set_status("Command cancelled.", event)
            return
        event.app.exit(result=None)

    @bindings.add("q", filter=browse_mode)
    def _quit(event) -> None:  # noqa: ANN001
        event.app.exit(result=None)

    @bindings.add("r", filter=browse_mode)
    def _resume(event) -> None:  # noqa: ANN001
        open_resume_for_selection(event)

    @bindings.add("d", filter=browse_mode)
    def _diff(event) -> None:  # noqa: ANN001
        action = selected_turn_action("diff")
        if action is None:
            set_status("Select a checkpoint turn before diffing.", event)
            return
        run_inline_action(action, event)

    @bindings.add("pagedown")
    def _page_down(event) -> None:  # noqa: ANN001
        if scroll_output(_output_page_size(state), event):
            return
        current_rows = rows()
        if current_rows:
            provider = current_provider()
            selected_by_provider[provider] = min(
                selected_by_provider[provider] + _tree_page_size(state),
                len(current_rows) - 1,
            )
            _sync_tree_scroll(state, selected_by_provider[provider], len(current_rows))
        invalidate(event)

    @bindings.add("pageup")
    def _page_up(event) -> None:  # noqa: ANN001
        if scroll_output(-_output_page_size(state), event):
            return
        current_rows = rows()
        if current_rows:
            provider = current_provider()
            selected_by_provider[provider] = max(0, selected_by_provider[provider] - _tree_page_size(state))
            _sync_tree_scroll(state, selected_by_provider[provider], len(current_rows))
        invalidate(event)

    @bindings.add("home", filter=browse_mode)
    def _home(event) -> None:  # noqa: ANN001
        if state.get("output_visible") and _output_can_scroll(state):
            state["output_scroll"] = 0
            set_status(_output_scroll_status(state), event)
            return
        current_rows = rows()
        if current_rows:
            selected_by_provider[current_provider()] = 0
            state["tree_scroll"] = 0
            invalidate(event)

    @bindings.add("end", filter=browse_mode)
    def _end(event) -> None:  # noqa: ANN001
        if state.get("output_visible") and _output_can_scroll(state):
            state["output_scroll"] = _max_output_scroll(state)
            set_status(_output_scroll_status(state), event)
            return
        current_rows = rows()
        if current_rows:
            selected_by_provider[current_provider()] = len(current_rows) - 1
            _sync_tree_scroll(state, len(current_rows) - 1, len(current_rows))
            invalidate(event)

    @bindings.add("tab", filter=browse_mode)
    def _expand_all(event) -> None:  # noqa: ANN001
        for nodes in providers.values():
            for node in _walk_sessions(nodes):
                expanded.add(node.session_id)
        set_status("Expanded all sessions", event)

    @bindings.add("s-tab", filter=browse_mode)
    def _collapse_all(event) -> None:  # noqa: ANN001
        expanded.clear()
        set_status("Collapsed all sessions", event)

    @bindings.add("c-up")
    def _resize_output_up(event) -> None:  # noqa: ANN001
        if state.get("output_visible"):
            state["output_height"] = min(20, int(state.get("output_height", 8)) + 2)
            state["output_scroll"] = _clamp_output_scroll(state, int(state.get("output_scroll") or 0))
            set_status(f"Output height: {state['output_height']} lines", event)

    @bindings.add("c-down")
    def _resize_output_down(event) -> None:  # noqa: ANN001
        if state.get("output_visible"):
            state["output_height"] = max(4, int(state.get("output_height", 8)) - 2)
            state["output_scroll"] = _clamp_output_scroll(state, int(state.get("output_scroll") or 0))
            set_status(f"Output height: {state['output_height']} lines", event)

    @bindings.add("c-f")
    def _output_page_down(event) -> None:  # noqa: ANN001
        scroll_output(_output_page_size(state), event)

    @bindings.add("c-b")
    def _output_page_up(event) -> None:  # noqa: ANN001
        scroll_output(-_output_page_size(state), event)

    @bindings.add("[", filter=browse_mode)
    def _prev_session(event) -> None:  # noqa: ANN001
        current_rows = rows()
        if not current_rows:
            return
        provider = current_provider()
        current = selected_by_provider[provider]
        for i in range(current - 1, -1, -1):
            if current_rows[i].kind == "session":
                selected_by_provider[provider] = i
                _sync_tree_scroll(state, i, len(current_rows))
                invalidate(event)
                return
        set_status("No previous session", event)

    @bindings.add("]", filter=browse_mode)
    def _next_session(event) -> None:  # noqa: ANN001
        current_rows = rows()
        if not current_rows:
            return
        provider = current_provider()
        current = selected_by_provider[provider]
        for i in range(current + 1, len(current_rows)):
            if current_rows[i].kind == "session":
                selected_by_provider[provider] = i
                _sync_tree_scroll(state, i, len(current_rows))
                invalidate(event)
                return
        set_status("No next session", event)

    root = HSplit([header, body, detail, help_panel, output, status, command])
    style = Style.from_dict(
        {
            # Tabs - more distinct active state
            "tab": "#888888",
            "tab.selected": "bg:#0087ff #ffffff bold",
            "tab.separator": "#666666",
            # Provider/Session - better hierarchy
            "provider": "#00d7ff bold",
            "session": "#ffffff bold",
            "session.startup": "#00ff87 bold",
            "session.fork": "#ffaf00 bold",
            "session.subagent": "#d787ff bold",
            # Turns - add recency indicator
            "turn": "#aaaaaa",
            "turn.recent": "#ffffff bold",
            # Relationships
            "fork": "#ff8700",
            "subagent": "#af87ff",
            "link": "#808080",
            # Tree structure
            "tree.branch": "#4e4e4e",
            # Text hierarchy
            "muted": "#767676",
            "dim": "#585858",
            # Status bar - contextual colors
            "status": "bg:#005f87 #ffffff",
            "status.command": "bg:#af8700 #ffffff",
            "status.confirm": "bg:#d70000 #ffffff bold",
            # Command mode
            "command": "#00ff00 bold",
            # Output pane
            "output.title": "#00d7ff bold",
            "output.meta": "#808080",
            "output.command": "bg:#005f00 #ffffff bold",
            # Help panel
            "help.header": "#00d7ff bold underline",
            "help.key": "#ffff00 bold",
            "help.text": "#bcbcbc",
            # Detail panel
            "detail.label": "#00afff bold",
            "detail.value": "#ffffff",
            "action.enabled": "#00ff87 bold",
            "action.disabled": "#767676",
            # Badges - more distinct
            "badge.resumable": "bg:#00af00 #ffffff bold",
            "badge.blocked": "bg:#af0000 #ffffff bold",
        }
    )
    return Application(layout=Layout(root, focused_element=body), key_bindings=bindings, style=style, full_screen=True).run()


def _load_nodes(root: Path, show_all: bool = False) -> dict[str, SessionNode]:
    if not root.exists():
        return {}
    nodes: dict[str, SessionNode] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        store = CheckpointStore(child)
        reanchor_last_turn_to_eof(store)
        metadata = _read_metadata(child)
        manifests = store.list_manifests()

        # Filter out empty sessions unless show_all is True
        if not show_all and _is_empty_node(manifests, metadata):
            continue

        marker = _session_marker(metadata)
        nodes[child.name] = SessionNode(child.name, metadata, manifests, marker=marker)
    return nodes


def _is_empty_node(manifests: list[CheckpointManifest], metadata: dict[str, Any]) -> bool:
    """Check if a session node is empty/dirty."""
    # No turns at all
    if not manifests:
        return True

    # All turns have empty trajectories (0 records)
    for manifest in manifests:
        if manifest.trajectory_ref and manifest.trajectory_ref.record_count > 0:
            return False

    return True


def _link_nodes(nodes: dict[str, SessionNode], root: Path) -> None:
    path_index = _transcript_path_index(nodes)
    for node in nodes.values():
        lineage = node.lineage
        parent_session = _text(lineage.get("parent_session_id"))
        if parent_session and parent_session in nodes:
            turn = _safe_parent_turn(root, parent_session, _text(lineage.get("agent_id")), node.metadata)
            node.subagent_parent = (parent_session, turn)
            nodes[parent_session].subagent_children.setdefault(turn, []).append(node)
            continue
        parent = _fork_parent(node, nodes, path_index)
        if parent is not None:
            parent_session, turn = parent
            node.fork_parent = parent
            nodes[parent_session].fork_children.setdefault(turn, []).append(node)
    for node in nodes.values():
        for children in [*node.fork_children.values(), *node.subagent_children.values()]:
            children.sort(key=_session_sort_key)


def _fork_parent(
    node: SessionNode,
    nodes: dict[str, SessionNode],
    path_index: dict[str, str],
) -> tuple[str, int | None] | None:
    parent_session = _text(node.metadata.get("forked_from_id"))
    transcript = _text(node.metadata.get("forked_from_transcript"))
    if (not parent_session or parent_session not in nodes) and transcript:
        parent_session = path_index.get(str(Path(transcript).expanduser()))
    if not parent_session or parent_session not in nodes or parent_session == node.session_id:
        return None
    turn = _fork_parent_turn(nodes[parent_session], node.metadata)
    return parent_session, turn


def _fork_parent_turn(parent: SessionNode, metadata: dict[str, Any]) -> int | None:
    offset = _int_or_none(metadata.get("forked_at_offset"))
    transcript = _text(metadata.get("forked_from_transcript"))
    if offset is not None and transcript:
        parent_path = str(Path(transcript).expanduser())
        candidates = []
        for manifest in parent.manifests:
            ref = manifest.trajectory_ref
            if ref is None:
                continue
            if str(Path(ref.transcript_path).expanduser()) != parent_path:
                continue
            end = ref.end_offset if ref.end_offset is not None else manifest.trajectory_end_offset
            if end is not None and end <= offset:
                candidates.append(manifest.turn_id)
        if candidates:
            return max(candidates)
    if parent.manifests:
        return parent.manifests[-1].turn_id
    return None


def _safe_parent_turn(root: Path, parent_session_id: str, agent_id: str | None, metadata: dict[str, Any]) -> int | None:
    try:
        return _parent_turn_for_subagent(root.parent, parent_session_id, agent_id, metadata)
    except Exception:
        return None


def _rows_for_nodes(nodes: list[SessionNode], expanded: set[str] | None = None) -> list[TreeRow]:
    expanded = expanded if expanded is not None else {node.session_id for node in _walk_sessions(nodes)}
    rows: list[TreeRow] = []
    for node in nodes:
        _append_session_rows(rows, node, 0, expanded)
    return rows


def _append_session_rows(rows: list[TreeRow], node: SessionNode, depth: int, expanded: set[str]) -> None:
    label = _session_label(node)
    is_expanded = node.session_id in expanded
    rows.append(
        TreeRow(
            "session",
            node,
            None,
            depth,
            label,
            _session_style(node),
            expanded=is_expanded,
            has_children=_has_session_children(node) or bool(node.manifests),
        )
    )
    if not is_expanded:
        return
    unknown_forks = node.fork_children.get(None, [])
    unknown_subagents = node.subagent_children.get(None, [])
    for child in [*unknown_forks, *unknown_subagents]:
        _append_session_rows(rows, child, depth + 1, expanded)
    for manifest in node.manifests:
        rows.append(TreeRow("turn", node, manifest, depth + 1, _turn_label(manifest), "class:turn"))
        for child in node.subagent_children.get(manifest.turn_id, []):
            rows.append(TreeRow("link", child, None, depth + 2, "subagent spawned here", "class:subagent"))
            _append_session_rows(rows, child, depth + 3, expanded)
        for child in node.fork_children.get(manifest.turn_id, []):
            rows.append(TreeRow("link", child, None, depth + 2, "forked/resumed here", "class:fork"))
            _append_session_rows(rows, child, depth + 3, expanded)


def _walk_sessions(nodes: list[SessionNode]) -> list[SessionNode]:
    result: list[SessionNode] = []
    for node in nodes:
        result.append(node)
        children = [child for group in node.fork_children.values() for child in group]
        children.extend(child for group in node.subagent_children.values() for child in group)
        result.extend(_walk_sessions(children))
    return result


def _has_session_children(node: SessionNode) -> bool:
    return any(node.fork_children.values()) or any(node.subagent_children.values())


def _header_fragments(
    provider_names: list[str],
    providers: dict[str, list[SessionNode]],
    state: dict[str, Any],
) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    selected = int(state["provider"])
    for index, provider in enumerate(provider_names):
        node_count = len(providers[provider])
        turn_count = sum(_count_turns(node) for node in providers[provider])
        style = "class:tab.selected" if index == selected else "class:tab"
        fragments.append((style, f" {provider} "))
        fragments.append(("class:dim", f"({node_count}/{turn_count}) "))
        if index < len(provider_names) - 1:
            fragments.append(("class:tab.separator", "│ "))
    return fragments


def _body_fragments(rows: list[TreeRow], selected: int, state: dict[str, Any]) -> list[tuple[str, str]]:
    if not rows:
        return [
            ("class:muted", "\n"),
            ("class:muted", "  📭 No sessions found for this provider.\n"),
            ("class:muted", "\n"),
            ("class:help.text", "  Try:\n"),
            ("class:help.text", "  • Switch provider tabs with ←/→ or h/l\n"),
            ("class:help.text", "  • Capture new sessions by using the CLI\n"),
            ("class:help.text", "  • Check that sessions directory exists\n"),
        ]
    selected = min(selected, len(rows) - 1)
    _sync_tree_scroll(state, selected, len(rows))
    start = int(state.get("tree_scroll") or 0)
    height = _tree_visible_row_count(state, start, len(rows))
    visible = rows[start : start + height]
    fragments: list[tuple[str, str]] = []
    if start > 0:
        percent = int((start / len(rows)) * 100)
        fragments.append(("class:muted", f"  ⬆ {start} rows above ({percent}% scrolled) ⬆\n"))
    selected_row = rows[selected]
    for row in visible:
        is_selected = row is selected_row
        row_frags = _row_fragments(row, is_selected, rows)
        fragments.extend(row_frags)
        fragments.append(("", "\n"))
    if start + height < len(rows):
        remaining = len(rows) - start - height
        percent = int(((start + height) / len(rows)) * 100)
        fragments.append(("class:muted", f"  ⬇ {remaining} rows below ({percent}% scrolled) ⬇\n"))
    return fragments


def _body_height(state: dict[str, Any]) -> int:
    return _tree_window_height(state)


def _tree_window_height(state: dict[str, Any]) -> int:
    terminal_lines = shutil.get_terminal_size().lines
    reserved = 1 + 8 + _help_height(state) + _output_height(state) + 1 + 1
    return max(8, terminal_lines - reserved)


def _tree_page_size(state: dict[str, Any]) -> int:
    return max(1, _tree_window_height(state) - 2)


def _tree_visible_row_count(state: dict[str, Any], start: int, row_count: int) -> int:
    window_height = _tree_window_height(state)
    if row_count <= window_height:
        return row_count
    top_hint = 1 if start > 0 else 0
    capacity = max(1, window_height - top_hint - 1)
    if start + capacity >= row_count:
        capacity = max(1, window_height - top_hint)
    return min(capacity, row_count - start)


def _help_height(state: dict[str, Any]) -> int:
    return 12 if state.get("show_help") else 0


def _output_height(state: dict[str, Any]) -> int:
    if not state.get("output_visible"):
        return 0
    return min(int(state.get("output_height", 8)), shutil.get_terminal_size().lines // 3)


def _output_page_size(state: dict[str, Any]) -> int:
    return max(1, _output_height(state) - 2)


def _sync_tree_scroll(state: dict[str, Any], selected: int, row_count: int) -> None:
    if row_count <= _tree_window_height(state):
        state["tree_scroll"] = 0
        return
    start = max(0, min(int(state.get("tree_scroll") or 0), row_count - 1))
    for _ in range(4):
        height = _tree_visible_row_count(state, start, row_count)
        if selected < start:
            start = selected
            continue
        if selected >= start + height:
            start = selected - height + 1
            continue
        break
    state["tree_scroll"] = max(0, min(start, row_count - 1))


def _detail_fragments(row: TreeRow | None, state: dict[str, Any]) -> list[tuple[str, str]]:
    if row is None:
        return [("class:muted", "\n" + "─" * 40 + "\nNo selection.\n")]
    node = row.node
    fragments: list[tuple[str, str]] = [("class:detail.label", "\n" + "━" * 40 + "\n")]

    # Session info with icons for better visual recognition
    fragments.append(("class:detail.label", "📋 Session: "))
    fragments.append(("class:detail.value", f"{node.session_id[:16]}…\n"))

    fragments.append(("class:detail.label", "🔌 Provider: "))
    fragments.append(("class:detail.value", node.provider))
    fragments.append(("class:dim", "  ┃  "))
    fragments.append(("class:detail.label", "Source: "))
    fragments.append(("class:detail.value", node.source))
    fragments.append(("class:dim", "  ┃  "))
    fragments.append(("class:detail.label", "⏰ Created: "))
    fragments.append(("class:detail.value", _format_timestamp(node.created_ts) + "\n"))

    fragments.append(("class:detail.label", "💬 Title: "))
    fragments.append(("class:detail.value", node.title + "\n"))
    fragments.extend(_selected_command_fragments(row))

    # Lineage info with better visual markers
    if node.fork_parent:
        fragments.append(("", "\n"))
        fragments.append(("class:fork", f"🔀 Fork from {node.fork_parent[0][:12]}… turn {_turn_text(node.fork_parent[1])}\n"))
    if node.subagent_parent:
        agent = _text(node.lineage.get("agent_id")) or "-"
        fragments.append(("", "\n"))
        fragments.append(("class:subagent", f"⚡ Subagent: {agent}, parent {node.subagent_parent[0][:12]}… turn {_turn_text(node.subagent_parent[1])}\n"))

    # Turn info with separator
    if row.manifest is not None:
        manifest = row.manifest
        fragments.append(("class:dim", "\n" + "─" * 40 + "\n"))
        fragments.append(("class:detail.label", f"🔄 Turn {manifest.turn_id}"))
        fragments.append(("class:dim", f"  ┃  {_format_timestamp(manifest.created_ts)}"))
        fragments.append(("class:dim", f"  ┃  prev: {_turn_text(manifest.parent_turn_id)}\n"))

        can_resume = _can_resume_row(row)
        if can_resume:
            fragments.append(("class:badge.resumable", " ✓ RESUMABLE "))
            fragments.append(("", "  Press r or /resume to show the restore command\n"))
        else:
            if row.node.subagent_parent or row.node.source == "subagent":
                fragments.append(("class:badge.blocked", " ✗ SUBAGENT "))
                fragments.append(("class:dim", "  Cannot resume subagent sessions\n"))
            elif row.node.marker:
                fragments.append(("class:badge.blocked", " ✗ NO CAPTURE "))
                fragments.append(("class:dim", "  Session was not fully captured\n"))
            else:
                fragments.append(("class:dim", "  Actions: show, diff\n"))

        msg_preview = manifest.user_message_preview or "-"
        if len(msg_preview) > 100:
            msg_preview = msg_preview[:97] + "…"
        fragments.append(("class:dim", "\n"))
        fragments.append(("class:muted", f"💭 {msg_preview}\n"))
    elif node.marker:
        fragments.append(("class:badge.blocked", f"\n{node.marker}\n"))

    return fragments


def _selected_command_fragments(row: TreeRow) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = [
        ("class:detail.label", "Commands: "),
    ]
    has_turn = row.manifest is not None
    commands = [
        ("show", has_turn),
        ("diff", has_turn),
        ("resume", _can_resume_row(row)),
    ]
    for index, (name, available) in enumerate(commands):
        if index:
            fragments.append(("class:dim", "  "))
        style = "class:action.enabled" if available else "class:action.disabled"
        value = "yes" if available else "no"
        fragments.append((style, f"{name}:{value}"))
    reason = _resume_unavailable_reason(row)
    if reason:
        fragments.append(("class:dim", f"  ({reason})"))
    fragments.append(("", "\n"))
    return fragments


def _row_fragments(row: TreeRow, is_selected: bool, all_rows: list[TreeRow]) -> list[tuple[str, str]]:
    """Generate formatted text fragments for a tree row using the rendering helper."""
    return render_tree_row(row, is_selected, all_rows)


def _format_timestamp(ts: str | None) -> str:
    """Format timestamp using helper function."""
    return format_timestamp(ts)


def _status_fragments(state: dict[str, Any]) -> list[tuple[str, str]]:
    """Generate status bar fragments with contextual styling."""
    mode = state.get("mode", "browse")
    status_text = str(state.get("status", ""))

    if mode == "command":
        return [("class:status.command", f" COMMAND: {status_text} ")]
    else:
        return [("class:status", f" {status_text} ")]


def _help_fragments() -> list[tuple[str, str]]:
    """Generate help panel fragments."""
    fragments: list[tuple[str, str]] = [
        ("class:help.header", "\n━━━ Keyboard Shortcuts ━━━\n"),
        ("class:help.key", "↑/↓ or j/k"),
        ("class:help.text", "      Move selection\n"),
        ("class:help.key", "←/→ or h/l"),
        ("class:help.text", "      Switch provider tab\n"),
        ("class:help.key", "Enter"),
        ("class:help.text", "            Toggle session expand/collapse\n"),
        ("class:help.key", "[ / ]"),
        ("class:help.text", "             Jump to previous/next session\n"),
        ("class:help.key", "Tab / Shift+Tab"),
        ("class:help.text", "   Expand all / Collapse all\n"),
        ("class:help.key", "r"),
        ("class:help.text", "                 Show resume command for selected checkpoint\n"),
        ("class:help.key", "d"),
        ("class:help.text", "                 Show diff for selected turn\n"),
        ("class:help.key", "/"),
        ("class:help.text", "                 Enter command mode\n"),
        ("class:help.key", "PageUp/PageDown"),
        ("class:help.text", "    Scroll tree or output\n"),
        ("class:help.key", "Ctrl+F / Ctrl+B"),
        ("class:help.text", "      Scroll output pane\n"),
        ("class:help.key", "Ctrl+↑/↓"),
        ("class:help.text", "          Resize output pane\n"),
        ("class:help.key", "Home / End"),
        ("class:help.text", "        Jump to first/last row\n"),
        ("class:help.key", "? or F1"),
        ("class:help.text", "          Toggle this help\n"),
        ("class:help.key", "q or Esc"),
        ("class:help.text", "          Quit browser\n"),
    ]
    return fragments


def _session_label(node: SessionNode) -> str:
    parts = [
        f"{node.session_id[:12]}…",
        f"[{node.source}]",
        f"{len(node.manifests)}T",
    ]
    ts = _format_timestamp(node.created_ts)
    if ts != "-":
        parts.append(ts)
    if node.title and node.title != "-":
        parts.append(truncate_with_ellipsis(node.title, 40))
    if node.marker:
        parts.append(f"⚠ {node.marker}")
    return " │ ".join(parts)


def _turn_label(manifest: CheckpointManifest) -> str:
    preview = manifest.user_message_preview.replace("\n", " ") or "-"
    preview = truncate_with_ellipsis(preview, 60)
    ts = _format_timestamp(manifest.created_ts)
    return f"T{manifest.turn_id:04d} │ {ts} │ {preview}"


def _session_style(node: SessionNode) -> str:
    if node.source in {"fork", "resume", "compact"} or node.fork_parent:
        return "class:fork"
    if node.source == "subagent" or node.subagent_parent:
        return "class:subagent"
    return "class:session"


def _command_action(command: str, row: TreeRow | None) -> BrowserAction | None:
    command = command.strip()
    if command.startswith("/"):
        command = command[1:]
    name = command.split(maxsplit=1)[0].lower() if command else "help"
    if name in {"help", "?"}:
        return BrowserAction("help")
    if name in {"quit", "q", "exit"}:
        return BrowserAction("quit")
    if name in {"show", "diff", "resume"}:
        if row is None or row.manifest is None:
            return None
        if name == "resume" and not _can_resume_row(row):
            return None
        return BrowserAction(name, row.node.session_id, row.manifest.turn_id)
    return None


def _can_resume_row(row: TreeRow) -> bool:
    if row.kind != "turn" or row.manifest is None:
        return False
    if row.node.subagent_parent is not None or row.node.source == "subagent":
        return False
    if row.node.marker:
        return False
    return True


def _resume_unavailable_reason(row: TreeRow) -> str | None:
    if _can_resume_row(row):
        return None
    if row.manifest is None:
        return "select a checkpoint turn"
    if row.node.subagent_parent is not None or row.node.source == "subagent":
        return "resume unavailable: subagent"
    if row.node.marker:
        return f"resume unavailable: {row.node.marker}"
    return "resume unavailable"


def _output_fragments(state: dict[str, Any]) -> list[tuple[str, str]]:
    if not state.get("output_visible"):
        return [("class:muted", "\nCommand Output\nNo command output yet.\n")]
    title = str(state.get("output_title") or "Command Output")
    text = str(state.get("output_text") or "")
    lines = text.splitlines() or [""]
    body_height = _output_content_height(state)
    scroll = _clamp_output_scroll(state, int(state.get("output_scroll") or 0))
    state["output_scroll"] = scroll
    visible = lines[scroll : scroll + body_height]
    end = scroll + len(visible)
    fragments: list[tuple[str, str]] = [
        ("class:output.title", title),
        ("class:output.meta", f"  {scroll + 1}-{max(scroll + 1, end)}/{len(lines)}"),
    ]
    if _output_can_scroll(state):
        fragments.append(("class:output.meta", "  PageUp/PageDown or Ctrl+F/Ctrl+B"))
    fragments.append(("", "\n"))
    for line in visible:
        fragments.append((_line_style(line), f"{line}\n"))
    if end < len(lines):
        fragments.append(("class:muted", f"... {len(lines) - end} lines below\n"))
    return fragments


def _output_content_height(state: dict[str, Any]) -> int:
    # One line is used for the title, and one line is reserved for the lower
    # scroll hint so it remains visible inside the pane.
    return max(1, _output_height(state) - 2)


def _max_output_scroll(state: dict[str, Any]) -> int:
    text = str(state.get("output_text") or "")
    line_count = len(text.splitlines() or [""])
    return max(0, line_count - _output_content_height(state))


def _clamp_output_scroll(state: dict[str, Any], scroll: int) -> int:
    return max(0, min(scroll, _max_output_scroll(state)))


def _output_can_scroll(state: dict[str, Any]) -> bool:
    return _max_output_scroll(state) > 0


def _output_scroll_status(state: dict[str, Any]) -> str:
    text = str(state.get("output_text") or "")
    line_count = len(text.splitlines() or [""])
    scroll = _clamp_output_scroll(state, int(state.get("output_scroll") or 0))
    content_height = _output_content_height(state)
    end = min(line_count, scroll + content_height)
    return f"Output lines {scroll + 1}-{end} of {line_count}"


def _line_style(line: str) -> str:
    if line.startswith("checkpoint resume "):
        return "class:output.command"
    if line.startswith("+"):
        return "ansigreen"
    if line.startswith("-"):
        return "ansired"
    if line.startswith("@@"):
        return "ansicyan"
    if line.startswith("Error") or line.startswith("Cannot"):
        return "ansired"
    return ""


def _show_result(session_id: str, turn_id: int) -> tuple[str, str]:
    session_path = sessions_dir() / session_id

    if not session_path.exists():
        return (
            f"Show {session_id} turn {turn_id}",
            f"Error: Session directory not found at {session_path}\n\nThe session may have been deleted or moved."
        )

    store = CheckpointStore(session_path)

    try:
        reanchor_last_turn_to_eof(store)
    except Exception:
        # Continue even if reanchor fails
        pass

    try:
        manifest = store.read_manifest(turn_id)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return (
            f"Show {session_id} turn {turn_id}",
            f"Error: {exc}\n\nAvailable turns: {[m.turn_id for m in store.list_manifests()]}"
        )

    return (
        f"Show {session_id} turn {turn_id}",
        json.dumps(manifest.to_json(), indent=2, sort_keys=True),
    )


def _diff_result(session_id: str, turn_id: int) -> tuple[str, str]:
    reanchor_last_turn_to_eof(CheckpointStore(sessions_dir() / session_id))
    orchestrator = ResumeOrchestrator()
    try:
        return f"Diff {session_id} turn {turn_id}", orchestrator.plan(session_id, turn_id).render()
    except RuntimeError as exc:
        return f"Diff {session_id} turn {turn_id}", f"Error: {exc}"


def _resume_hint(session_id: str, turn_id: int) -> tuple[str, str]:
    command = f"checkpoint resume {session_id} {turn_id}"
    return (
        f"Resume {session_id} turn {turn_id}",
        "\n".join(
            [
                "Run this command outside the browser to restore the checkpoint:",
                "",
                command,
                "",
                "The CLI will show the restore diff and ask for confirmation before applying changes.",
            ]
        ),
    )


def _read_metadata(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _session_marker(metadata: dict[str, Any]) -> str:
    lineage = metadata.get("lineage") or {}
    if isinstance(lineage, dict) and lineage.get("capture_status") == "no_sidechain_file":
        return "[no capture]"
    return ""


def _transcript_path_index(nodes: dict[str, SessionNode]) -> dict[str, str]:
    index: dict[str, str] = {}
    for node in nodes.values():
        for manifest in node.manifests:
            ref = manifest.trajectory_ref
            if ref is not None and ref.transcript_path:
                index[str(Path(ref.transcript_path).expanduser())] = node.session_id
    return index


def _manifest_provider(manifests: list[CheckpointManifest]) -> str | None:
    for manifest in manifests:
        ref = manifest.trajectory_ref
        if ref is not None and ref.provider:
            return ref.provider
    return None


def _count_turns(node: SessionNode) -> int:
    return len(node.manifests) + sum(_count_turns(child) for group in node.fork_children.values() for child in group) + sum(
        _count_turns(child) for group in node.subagent_children.values() for child in group
    )


def _session_sort_key(node: SessionNode) -> tuple[str, str]:
    return (node.created_ts, node.session_id)


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _turn_text(turn_id: int | None) -> str:
    return str(turn_id) if turn_id is not None else "?"
