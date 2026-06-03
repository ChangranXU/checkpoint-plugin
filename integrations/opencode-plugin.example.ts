/**
 * OpenCode Checkpoint Plugin
 *
 * Bridges OpenCode events to the Python checkpoint hook.
 *
 * Installation:
 *   checkpoint hooks install opencode
 * This file lands at ~/.config/opencode/plugins/checkpoint.ts
 * OpenCode auto-discovers and loads it.
 *
 * How it works:
 * - Bus events (session.idle, session.created, message.removed) are delivered
 *   via the single `event` callback — NOT as top-level hook keys.
 * - The `chat.message` trigger fires on every user prompt submission.
 * - We spawn the Python hook subprocess with event name + JSON payload on stdin.
 */

import { spawn } from "child_process"
import { appendFileSync } from "fs"

const DEBUG_LOG = "/tmp/checkpoint-plugin-debug.log"
function debug(msg: string) {
  try { appendFileSync(DEBUG_LOG, `${new Date().toISOString()} ${msg}\n`) } catch {}
}

interface CheckpointPayload {
  sessionID: string
  directory?: string
  worktree?: string
  source?: string
  agent_type?: "primary" | "subagent"
  parent_session_id?: string
  forked_from_session_id?: string
  messages?: any[]
  event_metadata?: Record<string, unknown>
}

function parseForkTitle(title: string | undefined): { baseTitle: string; forkNum: number } | null {
  if (!title) return null
  const match = title.match(/^(.+) \(fork #(\d+)\)$/)
  if (!match) return null
  return { baseTitle: match[1], forkNum: parseInt(match[2], 10) }
}

function invokeCheckpointHook(event: string, payload: CheckpointPayload): Promise<void> {
  debug(`INVOKE: event=${event} sessionID=${payload.sessionID}`)
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", ["-m", "checkpoint_plugin.integrations.opencode_hook", event], {
      stdio: ["pipe", "pipe", "pipe"],
    })

    let stderr = ""
    proc.stderr.on("data", (data: Buffer) => {
      stderr += data.toString()
    })

    proc.on("close", (code: number | null) => {
      if (code === 0) { debug(`HOOK OK: ${event}`); resolve() }
      else { debug(`HOOK FAIL: ${event} code=${code} err=${stderr}`); reject(new Error(`Checkpoint hook failed (exit ${code}): ${stderr}`)) }
    })

    proc.on("error", (err: Error) => {
      reject(new Error(`Failed to spawn checkpoint hook: ${err.message}`))
    })

    proc.stdin.write(JSON.stringify(payload))
    proc.stdin.end()
  })
}

export const CheckpointPlugin = async (ctx: {
  client: any
  project: any
  directory: string
  worktree: string
}) => {
  const { client, directory, worktree } = ctx
  const activeSessions = new Set<string>()
  // Track the last checkpointed message count per session to avoid duplicate
  // checkpoints when OpenCode fires both session.idle and session.status{idle}
  // for the same turn completion.
  const lastCheckpointedCount = new Map<string, number>()

  async function findForkOrigin(baseTitle: string, forkNum: number): Promise<string | undefined> {
    try {
      const resp = await client.session.list()
      const sessions: any[] = resp?.data ?? []
      // The fork's parent is the session with the base title (for fork #1)
      // or the previous fork number (for fork #N where N>1).
      const parentTitle = forkNum > 1 ? `${baseTitle} (fork #${forkNum - 1})` : baseTitle
      const match = sessions.find((s: any) => s.title === parentTitle)
      return match?.id
    } catch {
      return undefined
    }
  }

  return {
    /**
     * Bus events arrive here. OpenCode publishes session.created, session.idle,
     * message.removed etc through this single callback.
     */
    event: async ({ event }: { event: { id: string; type: string; properties: any } }) => {
      debug(`EVENT: type=${event.type} props=${JSON.stringify(event.properties ?? {}).slice(0, 200)}`)
      try {
        if (event.type === "session.created") {
          const sessionID: string = event.properties?.sessionID
          if (!sessionID) return
          activeSessions.add(sessionID)

          const info = event.properties?.info
          const parentID: string | undefined = info?.parentID
          const forkInfo = parseForkTitle(info?.title)

          let source: string | undefined
          let agentType: "primary" | "subagent" = "primary"
          let parentSessionId: string | undefined
          let forkedFromSessionId: string | undefined

          if (parentID) {
            source = "subagent"
            agentType = "subagent"
            parentSessionId = parentID
          } else if (forkInfo) {
            source = "fork"
            forkedFromSessionId = await findForkOrigin(forkInfo.baseTitle, forkInfo.forkNum)
          }

          const payload: CheckpointPayload = {
            sessionID,
            directory,
            worktree,
            source,
            agent_type: agentType,
            parent_session_id: parentSessionId,
            forked_from_session_id: forkedFromSessionId,
            event_metadata: {
              timestamp: new Date().toISOString(),
              hook_event_name: "SessionStart",
              model: info?.model?.id,
            },
          }

          await invokeCheckpointHook("session_start", payload)
        } else if (event.type === "session.idle" || event.type === "session.status") {
          // session.status with type=idle is the modern equivalent of session.idle
          if (event.type === "session.status") {
            const status = event.properties?.status
            if (status?.type !== "idle") return
          }

          const sessionID: string = event.properties?.sessionID
          if (!sessionID) return

          // Fetch messages to build the turn record
          let messages: any[] = []
          try {
            const resp = await client.session.messages({ path: { id: sessionID } })
            messages = resp?.data ?? []
          } catch {
            // SDK call may fail if session was deleted
          }

          debug(`IDLE: sessionID=${sessionID} msgCount=${messages.length} lastRole=${messages[messages.length - 1]?.info?.role ?? "none"}`)

          // Skip if no assistant turn completed
          // SDK returns {info: {role, ...}, parts: [...]} per message
          const lastMsg = messages[messages.length - 1]
          if (!lastMsg || lastMsg.info?.role !== "assistant") return

          // Deduplicate: OpenCode fires both session.idle AND session.status{idle}
          // for the same turn completion. Skip if we already checkpointed this state.
          if (lastCheckpointedCount.get(sessionID) === messages.length) return
          lastCheckpointedCount.set(sessionID, messages.length)

          // Flatten to simple {role, content} for the Python hook
          const flatMessages = messages.map((m: any) => ({
            role: m.info?.role,
            content: (m.parts ?? [])
              .filter((p: any) => p.type === "text")
              .map((p: any) => p.text ?? p.content ?? "")
              .join(""),
          }))

          // Detect subagent or fork
          let parentID: string | undefined
          let sessTitle: string | undefined
          try {
            const sess = await client.session.get({ path: { id: sessionID } })
            parentID = sess?.data?.parentID
            sessTitle = sess?.data?.title
          } catch {
            // best-effort
          }

          let source: string | undefined
          let agentType: "primary" | "subagent" = "primary"
          let parentSessionId: string | undefined
          let forkedFromSessionId: string | undefined

          if (parentID) {
            source = "subagent"
            agentType = "subagent"
            parentSessionId = parentID
          } else {
            const forkInfo = parseForkTitle(sessTitle)
            if (forkInfo) {
              source = "fork"
              forkedFromSessionId = await findForkOrigin(forkInfo.baseTitle, forkInfo.forkNum)
            }
          }

          const payload: CheckpointPayload = {
            sessionID,
            directory,
            worktree,
            source,
            agent_type: agentType,
            parent_session_id: parentSessionId,
            forked_from_session_id: forkedFromSessionId,
            messages: flatMessages,
            event_metadata: {
              timestamp: new Date().toISOString(),
              message_count: messages.length,
              hook_event_name: "Stop",
            },
          }

          await invokeCheckpointHook("turn_end", payload)
        } else if (event.type === "message.removed") {
          const sessionID: string = event.properties?.sessionID
          const messageID: string = event.properties?.messageID
          if (!sessionID) return

          // Reset dedup counter so the rollback checkpoint can proceed
          lastCheckpointedCount.delete(sessionID)

          let messages: any[] = []
          try {
            const resp = await client.session.messages({ path: { id: sessionID } })
            messages = resp?.data ?? []
          } catch {
            // best-effort
          }

          const flatMessages = messages.map((m: any) => ({
            role: m.info?.role,
            content: (m.parts ?? [])
              .filter((p: any) => p.type === "text")
              .map((p: any) => p.text ?? p.content ?? "")
              .join(""),
          }))

          const payload: CheckpointPayload = {
            sessionID,
            directory,
            worktree,
            messages: flatMessages,
            event_metadata: {
              timestamp: new Date().toISOString(),
              removed_message_id: messageID,
              rollback: true,
              hook_event_name: "Stop",
            },
          }

          await invokeCheckpointHook("turn_end", payload)
        }
      } catch (error) {
        console.error("[checkpoint]", event.type, error)
      }
    },

    /**
     * chat.message fires on every user prompt. We use it as a secondary
     * session-start signal in case session.created was missed (e.g. plugin
     * loaded after session already existed).
     */
    "chat.message": async (
      input: { sessionID: string; agent?: string; model?: any; messageID?: string },
      _output: any,
    ) => {
      debug(`CHAT.MESSAGE: sessionID=${input.sessionID} active=${activeSessions.has(input.sessionID)}`)
      try {
        const { sessionID } = input
        if (!sessionID || activeSessions.has(sessionID)) return
        activeSessions.add(sessionID)

        let parentID: string | undefined
        let sessTitle: string | undefined
        try {
          const sess = await client.session.get({ path: { id: sessionID } })
          parentID = sess?.data?.parentID
          sessTitle = sess?.data?.title
        } catch {
          // best-effort
        }

        let source: string | undefined
        let agentType: "primary" | "subagent" = "primary"
        let parentSessionId: string | undefined
        let forkedFromSessionId: string | undefined

        if (parentID) {
          source = "subagent"
          agentType = "subagent"
          parentSessionId = parentID
        } else {
          const forkInfo = parseForkTitle(sessTitle)
          if (forkInfo) {
            source = "fork"
            forkedFromSessionId = await findForkOrigin(forkInfo.baseTitle, forkInfo.forkNum)
          }
        }

        const payload: CheckpointPayload = {
          sessionID,
          directory,
          worktree,
          source,
          agent_type: agentType,
          parent_session_id: parentSessionId,
          forked_from_session_id: forkedFromSessionId,
          event_metadata: {
            timestamp: new Date().toISOString(),
            hook_event_name: "SessionStart",
            model: input.model?.modelID,
          },
        }

        await invokeCheckpointHook("session_start", payload)
      } catch (error) {
        console.error("[checkpoint] chat.message", error)
      }
    },
  }
}
