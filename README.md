# smol — a tiny coding agent

A minimal ReAct coding agent built with LangGraph. It loops between thinking (LLM) and acting (tools) to complete programming tasks, with a human-in-the-loop approval gate for any destructive operations.

## Architecture

```
main.py  (REPL)
  └── agent/graph.py  (LangGraph StateGraph)
        ├── think  →  LLM with bound tools
        ├── act    →  tool execution + approval gate
        └── should_continue  →  loop or stop
```

State is persisted across turns using LangGraph's `MemorySaver` checkpointer (keyed by `thread_id`), so the agent remembers context within a session.

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

Optional flags:

```
--provider   anthropic | openai | ollama  (default: anthropic)
--model      model name                   (default: claude-sonnet-4-6)
--max-iterations  safety loop limit       (default: 15)
--cwd        workspace root               (default: current directory)
--config     path to config YAML          (default: config.yaml)
```

## Commands

| Command    | Description                              |
|------------|------------------------------------------|
| `/help`    | List commands                            |
| `/model`   | Show current provider and model          |
| `/memory`  | Show agent memory (task, files, notes)   |
| `/cost`    | Show token usage and estimated cost      |
| `/compact` | Summarize old history to save context    |
| `/clear`   | Start a new conversation thread          |
| `/exit`    | Quit                                     |

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
