# Attaching another agent

This system is **one authoritative store that several agents share**. It is
installed once (see [install.md](install.md)); every agent after that
**attaches**. Attaching never creates a second store, and never re-installs the
plugin.

> This page uses **placeholder values** on purpose — see the note at the end.

## The usual entry point is a skill, not this file

Most agents on this machine do not find this repository first. They load a
**skill** — `hgm-shared-memory` — which carries the same contract in a form their
host knows how to discover. That is measured, not assumed: an agent attaching on
2026-09-18 read the skill, followed it, and never mentioned this repository.

Two consequences worth knowing:

- **There are two copies of the contract, and both are read.** If you change the
  contract, change both, or a reader of the copy you missed keeps the old rules.
  This exact gap is why a doc fix here went unexercised: the agent had already
  followed the skill before this page was consulted.
- **The skill is shared by every agent on the machine.** Its own text says so —
  *"Nothing here is specific to one agent"*. So **never put a per-agent value in
  it**: an agent name hardcoded there becomes every reader's name, and they will
  write under it. Per-agent values belong in that agent's own config.

## The three ways to attach

Pick by what the host supports. They are not alternatives to each other: a single
agent may use more than one.

| Surface | How it reaches the store | Good for |
|---|---|---|
| **CLI** | `python memory_cli.py <cmd>` | Any agent that can run a command. The lowest common denominator. |
| **MCP tools** | A stdio MCP server exposing five tools | Hosts with MCP support. **~10× faster per call** than the CLI (see below). |
| **Host hooks** | `SessionStart` / `UserPromptSubmit` / `PreToolUse` | Automating *when* memory is consulted and injected, so the agent does not have to remember to. |

### Why MCP is worth preferring

The CLI pays process start-up plus index load on **every** call — measured at
**~2.8s**. The MCP server is a resident process, so it pays that once and then
answers in **~0.30s**. A host that calls memory a few times per turn will feel
the difference.

### CLI

```bash
python memory_cli.py health                 # is the store reachable?
python memory_cli.py recall "<query>"       # L1+L2+L3+L4 + KB
python memory_cli.py remember "<fact>"      # write an L2 fact (goes through a gate)
python memory_cli.py kb-search "<query>"    # notes only
python memory_cli.py agents                 # who wrote what
```

### MCP tools

Run the server as a stdio MCP server and register it with the host. The five
tools are `hgm_recall`, `hgm_remember`, `hgm_kb_search`, `hgm_kb_add`,
`hgm_agents`; `memory_cli.py` is their only implementation, so the gate and the
provenance rules are identical to the CLI's.

Declare the agent's identity in the server's environment — that is the second
step of the identity chain below:

```json
{
  "mcpServers": {
    "hgm-memory": {
      "type": "stdio",
      "command": "python",
      "args": ["<repo>/scripts/hgm_mcp.py"],
      "env": { "HGM_AGENT": "<this-agent-name>" }
    }
  }
}
```

Any script that reads stdin must declare its encoding
(`sys.stdin.reconfigure(encoding="utf-8")`). The one shipped here does; a host
that hands over a non-UTF-8 stdin would otherwise get **mojibake, or a crash —
both silent**.

### Host hooks

Three events, each with one job:

| Event | Job |
|---|---|
| `SessionStart` | Inject L1 rules + which layers exist + **how to reach them** |
| `UserPromptSubmit` | Inject the L2 facts that match this turn |
| `PreToolUse` | Refuse writes to the hand-written layers (L1 / L4) |

If the host has hooks, the injected text decides what the agent does: an agent
that is told only about the CLI will use only the CLI, however good the other
surfaces are. Say which one to prefer.

## Identity

The store attributes every fact to an agent, and the chain is:

```
--agent  >  $HGM_AGENT  >  host-specific marker  >  "external"
```

Framework markers are probed **only** by their own dedicated variables. **An
unrecognised agent is recorded as `external` — never guessed.** A wrong name is
worse than a vague one: it puts one agent's facts under another's name, and
provenance is the only way to tell them apart later.

### Where to declare your name — and where not to

Prefer, in this order:

1. **Your own client config** — an MCP server's `env`, a launch profile, a tool
   config. This is the right home: the name travels with the client it belongs to.
2. `--agent <you>` on each call. Correct, just repetitive.

**Not** the shared shell profile (`~/.bashrc`, `~/.zshrc`, a system-wide
environment variable). Measured hesitation about this is a real signal — an
agent asked on 2026-09-18 whether to add `export HGM_AGENT=<itself>` to
`~/.bashrc`. The answer is no, for two reasons:

- **It poisons provenance.** Every process launched from that shell inherits the
  name — including other agents' processes. The one property the store relies on
  is that the attribution is *true*.
- **It often does not apply anyway.** A shell profile is read by interactive
  shells; a host that spawns a tool without one gets nothing, and the failure is
  silent — writes land as `external`.

## What an attaching agent may and may not write

- **L2 facts and KB notes: yes.** `remember` passes a gate that looks at how many
  independent pieces of evidence a statement carries; a refusal says why.
- **L1 (`MEMORY.md`, `USER.md`) and L4 (`persona.md`): no.** Those are injected as
  rules into *every* agent, so one agent editing them edits everyone's operating
  instructions. They are human-authored. The `PreToolUse` hook enforces this.
- A gate refusal is **structured JSON** (`{"ok": false, "error": "refused: …"}`),
  not a traceback — the caller is an agent, and a stack trace tells it nothing
  actionable. Do not retry a refusal unchanged; read the reason.

## Verify the attachment

Steps 1–3 prove you can reach the store. **They do not prove the thing you
actually attached for.** Do 4 as well.

1. `python memory_cli.py health` → `"ok": true`
2. `python memory_cli.py agents` → your agent name appears (or `external`, if the
   host has no marker)
3. Write something and read it back:
   `remember "<a fact with a concrete identifier>"` then `recall "<part of it>"`
4. **Read something *another agent* wrote, and check the signature comes back.**
   Pick a topic only the shared store knows — something you have no way to answer
   from your own context — then look at which agent the hit is attributed to:

   ```bash
   python memory_cli.py recall "<a topic from another agent's work>" --top-k 5
   ```

   ⚠️ **A hit with `agent=None` proves nothing about sharing.** Those are legacy
   rows nobody claimed; finding one only shows you can see the store, which step
   3 already showed. You are looking for a hit **signed with another agent's
   name**. Measured: an agent reported "proves cross-agent visibility" from two
   `agent=None` rows — the check looked passed and had tested nothing.
5. If hooks are wired: start a fresh session and confirm the injection arrived
6. If MCP is wired: confirm the host lists the five tools, then confirm a call
   actually happens — a tool that is *listed* and never *called* is the common
   failure, and it is invisible without a log

If step 4 cannot find an attributed fact from anyone else, that is worth saying
plainly: either nobody else has written anything you would find, or the two of
you are not actually sharing a store.

## Note on placeholders

This repository is public. Documentation, tests and examples must use
**placeholder values** — `192.0.2.x` (RFC 5737) for addresses, `C:/Users/<user>/`
for paths, `示例NAS`-style names for equipment. Real hostnames, addresses,
credentials and device names belong only in a local, untracked log.
