# AI Agency OS

A config-driven AI agent operating system powered by Claude. Inspired by OpenClaw, this system lets you define unlimited AI agents entirely in YAML — no Python required to add a new agent.

## Architecture

```
Telegram / Cron / CLI
        │
        ▼
   Orchestrator          ← regex-based routing from agents/orchestrator.yaml
        │
        ▼
 Domain Agent            ← config from agents/{name}.yaml
   (ReAct Loop)          ← native Anthropic tool_use blocks
        │
    ┌───┴───┐
  Tools   Memory
  (YAML-defined,   (JSONL sessions +
  subprocess-sandboxed)  markdown summaries)
```

**Key design principles:**
- YAML-first: add an agent by creating a YAML file in `agents/`
- Raw Anthropic SDK: no LangChain, no agent frameworks — full control over the ReAct loop
- Two-layer memory: JSONL conversation history + markdown flat-file summaries (silently flushed in background)
- Subprocess tool sandboxing: user-authored tools in `tools/` run in isolated child processes

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd ai-agency-os
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (and TELEGRAM_BOT_TOKEN if using Telegram)
```

### 3. Run locally (CLI mode — no Telegram needed)

```bash
python main.py --trigger=cli
```

### 4. Run with Telegram

```bash
# Set TELEGRAM_BOT_TOKEN in .env first
python main.py --trigger=telegram
```

## Adding a New Agent

Create `agents/my_agent.yaml`:

```yaml
name: my_agent
description: "What this agent does"
model: claude-sonnet-4-6
max_tokens: 4096
max_react_iterations: 8

system_prompt: |
  You are a specialist in X. Your job is to Y.

skills:
  - name: reply_formatting   # inject skills/reply_formatting.md

tools:
  - name: get_datetime
    description: "Get the current date and time"
    module: src.tools.builtins.datetime_tool
    function: get_datetime
    sandboxed: false
    parameters:
      type: object
      properties:
        timezone: {type: string, default: UTC}
      required: []

memory:
  enabled: true
  flush_interval_seconds: 60
```

Then add a routing rule in `agents/orchestrator.yaml`:

```yaml
routing_rules:
  - pattern: "\\b(my|keyword)\\b"
    agent: my_agent
    priority: 10
```

Restart the bot — no code changes needed.

## Adding a Custom Tool

Create `tools/my_tool.py`:

```python
def my_tool(param: str) -> dict:
    """Do something and return a result dict."""
    return {"result": f"Processed: {param}"}
```

Reference it in an agent YAML:

```yaml
tools:
  - name: my_tool
    description: "Does something useful"
    module: tools.my_tool
    function: my_tool
    sandboxed: true        # runs in a subprocess for safety
    parameters:
      type: object
      properties:
        param: {type: string, description: "Input parameter"}
      required: [param]
```

## Adding Skills

Create `skills/my_skill.md` with instructions to inject into the agent's system prompt:

```markdown
## My Skill

When doing X, always follow these steps:
1. Step one
2. Step two
```

Reference it in any agent YAML under `skills:`.

## Scheduling (Cron Jobs)

Add cron jobs to `agents/orchestrator.yaml`:

```yaml
cron_jobs:
  - id: daily_report
    cron: "0 9 * * 1-5"      # 9am Mon-Fri UTC
    agent: research_agent
    message: "Generate a daily briefing on AI news"
    reply_chat_id: "${TELEGRAM_ADMIN_CHAT_ID}"
```

## Project Structure

```
├── main.py                   # Entry point
├── agents/                   # YAML agent configs
├── skills/                   # Markdown skill files
├── tools/                    # User-authored custom tools
├── memory/                   # Runtime data (git-ignored)
│   ├── sessions/             # JSONL conversation files
│   └── flat/                 # Markdown memory summaries
└── src/
    ├── config/               # Pydantic models + YAML loader
    ├── core/                 # Orchestrator, BaseAgent, AgentRegistry
    ├── tools/                # ToolRegistry, ToolExecutor, builtins
    ├── memory/               # SessionManager, MemoryFlusher
    ├── skills/               # SkillLoader
    ├── triggers/             # Telegram, Cron, CLI triggers
    └── gateway/              # Messaging gateways
```

## Running Tests

```bash
pytest
```

## Deploying to Render

1. Fork this repo
2. Create a new Render **Worker** service (not Web — bots are workers)
3. Connect your repo, Render will use `render.yaml` automatically
4. Set environment variables in Render dashboard:
   - `ANTHROPIC_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_ALLOWED_CHAT_IDS` (optional, comma-separated)
5. Deploy

The worker uses a persistent disk for the `memory/` directory.

## Running with Docker

```bash
# Copy and edit .env first
docker-compose up --build
```
