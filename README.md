# smol — a tiny coding agent

A minimal ReAct coding agent built with LangGraph. It loops between thinking (LLM) and acting (tools) to complete programming tasks, with a human-in-the-loop approval gate for any destructive operations.

## Architecture

```
main.py  (REPL)
  └── agent/graph.py  (LangGraph StateGraph)
        ├── think      →  LLM with bound tools
        ├── act        →  tool execution + approval gate
        ├── summarize  →  wrap-up when step limit is hit
        └── should_continue  →  "continue" | "done" | "summarize"
```

State is persisted across turns using LangGraph's `MemorySaver` checkpointer (keyed by `thread_id`). Branches each get their own `thread_id`, so history is fully isolated between them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here
```

## Usage

```bash
python3 main.py
```

The prompt shows the active branch: `smol[main]>` or `smol[auth-refactor]>`.

Optional flags:

```
--provider        anthropic | openai | ollama  (default: anthropic)
--model           model name                   (default: claude-sonnet-4-6)
--max-iterations  safety loop limit            (default: 15)
--cwd             workspace root               (default: current directory)
--config          path to config YAML          (default: config.yaml)
```

## Commands

### General

| Command    | Description                              |
|------------|------------------------------------------|
| `/help`    | List all commands                        |
| `/model`   | Show current provider and model          |
| `/memory`  | Show agent memory for the current thread |
| `/cost`    | Show token usage and estimated cost      |
| `/compact` | Summarize old history to save context    |
| `/clear`   | Start a fresh conversation thread        |
| `/exit`    | Quit                                     |

### Branching

Branches let you explore a risky or experimental path without polluting your main conversation. When done, `/fold` summarizes the branch and returns you to the parent with a one-paragraph recap.

| Command           | Description                                          |
|-------------------|------------------------------------------------------|
| `/branch <name>`  | Create a new branch from the current thread          |
| `/switch <name>`  | Switch to an existing branch                         |
| `/branches`       | List all branches (active one marked with `*`)       |
| `/fold`           | Summarize current branch and merge back into parent  |

**Example workflow:**

```
smol[main]> /branch auth-refactor
Created and switched to branch 'auth-refactor'.

smol[auth-refactor]> rewrite the auth middleware to use JWTs
  ⚡ read_file(...)
  ⚡ write_file(...)
  ...

smol[auth-refactor]> /fold
Folded 'auth-refactor' → 'main'. Summarised 14 messages.

Summary:
Rewrote auth/middleware.py to use JWT tokens via PyJWT. Added
token expiry (24h) and a refresh endpoint at /auth/refresh.
Tests updated in tests/test_auth.py — all passing.

smol[main]>
```

## Robustness

- **Iteration cap** — each user message gets up to `max_iterations` tool calls (default 15); if the cap is hit, the agent summarizes progress before stopping
- **Loop detection** — if the same tool is called twice with identical args, the agent stops and summarizes rather than burning through all iterations
- **Tool timeout** — every tool call has a 30-second deadline; hung shell commands are killed and return an error
- **Approval gate** — risky tools require explicit `y/N` before executing; `run_shell` shows the exact command

## Tools

**Safe** (no approval needed):

- `list_files` — explore directories
- `read_file` — read file contents
- `search` — grep for patterns in code

**Risky** (require `y/N` approval):

- `write_file` — create or overwrite a file
- `patch_file` — apply a targeted edit
- `run_shell` — run shell commands, tests, git

All MCP tools also require approval.

## MCP Servers

Configured in `config.yaml` under `mcp_servers`. The GitHub MCP server requires Docker:

```yaml
mcp_servers:
  github:
    command: docker
    args: ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
```

## Skills

Drop `.md` files into `skills/` to give the agent reusable playbooks. The agent lists available skills in its system prompt and can load one on demand via `load_skill`.

## Configuration

`config.yaml` supports `${VAR}` references that are expanded from `.env` at startup.

```yaml
agent:
  provider: anthropic
  model: claude-sonnet-4-6
  max_iterations: 15
  temperature: 0
```

## Providers

| Provider  | Env var              | Notes                        |
|-----------|----------------------|------------------------------|
| anthropic | `ANTHROPIC_API_KEY`  | Default                      |
| openai    | `OPENAI_API_KEY`     |                              |
| ollama    | —                    | Needs Ollama running locally |
