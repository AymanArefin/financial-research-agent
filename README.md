# Financial Research Agent

An AI-powered financial research assistant inspired by https://github.com/virattt/dexter that gives you real-time market data, financial statements, and news — all in a single conversational interface. Built on Cloudflare Workers with the Agents SDK, it orchestrates multi-step tool calls across Yahoo Finance, Financial Modeling Prep, and Cloudflare Browser Rendering to answer complex equity, crypto, and macro questions.

![Financial Research Agent](./npm-agents-banner.svg)

---

## What It Does

Ask the agent anything about a stock, crypto asset, or ETF and it will autonomously:

1. **Resolve tickers** — normalizes "Apple" → `AAPL`, "Bitcoin" → `BTC-USD`
2. **Fetch live prices** — current price, day change %, volume, market cap
3. **Pull historical data** — OHLCV bars with natural-language date parsing ("last 3 months" → concrete dates)
4. **Retrieve company fundamentals** — P/E ratio, EPS, sector, 52-week range, dividend yield
5. **Read financial statements** — income statement, balance sheet, cash flow (quarterly or annual)
6. **Surface recent news** — latest headlines and sentiment
7. **Fall back to a real browser** — when APIs rate-limit or fail, Cloudflare Browser Rendering opens the page like a human would
8. **Synthesize a report** — structures findings under `## Price`, `## Company Overview`, `## Financials`, `## Recent News`, `## Key Takeaways`

---

## Agentic Logic

### Architecture Overview

```
Browser (React 19)
  └─ useAgent / useAgentChat (WebSocket)
       └─ Cloudflare Workers (HTTP router)
            └─ FinancialResearchAgent (Durable Object per session)
                 ├─ SQLite state (scratchpad, MCP connections)
                 ├─ Workers AI  →  GLM 4.7 Flash
                 ├─ Tool loop  (up to 10 iterations)
                 │    ├─ resolveTickerSymbol
                 │    ├─ inferDateRange
                 │    ├─ getSnapshot        ← Yahoo Finance /v8/finance/chart
                 │    ├─ getTickerDetails   ← Yahoo Finance /v7/finance/quote
                 │    ├─ getAggregates      ← Yahoo Finance OHLCV bars
                 │    ├─ getTickerNews      ← Yahoo Finance /v1/finance/search
                 │    ├─ getFinancials      ← Financial Modeling Prep /stable/*
                 │    └─ browseUrl          ← Cloudflare Browser Rendering (Puppeteer)
                 └─ MCP servers (user-connected, OAuth-aware)
```

### One Durable Object Per Session

`FinancialResearchAgent` extends `AIChatAgent` from `@cloudflare/ai-chat`, which is itself built on top of `Agent` from the Agents SDK. Each session gets its own Durable Object instance backed by **SQLite** — conversation history, scratchpad state, and MCP server configurations all survive hibernation and redeploys automatically.

```ts
export class FinancialResearchAgent extends AIChatAgent<Env, AgentState> {
  initialState = INITIAL_STATE;   // typed scratchpad, zero iterations
  waitForMcpConnections = true;   // restore user-added MCP servers after wake
  // ...
}
```

### Scratchpad State Pattern

Every tool invocation appends a structured `ToolResult` to `this.state.scratchpad`. This lets the frontend visualize exactly what the agent is doing in real-time, and lets the model "look back" at what it has already fetched before deciding its next action:

```ts
this.setState({
  ...this.state,
  scratchpad: {
    ...this.state.scratchpad,
    iterations: this.state.scratchpad.iterations + 1,
    toolResults: [...this.state.scratchpad.toolResults, ...newResults],
    totalTokensUsed: this.state.scratchpad.totalTokensUsed + tokensThisStep,
  },
});
```

### Multi-Step Tool Loop with `streamText`

The agent uses the Vercel AI SDK's `streamText` with `toolChoice: "auto"` and a `stopWhen: stepCountIs(MAX_ITERATIONS)` guard (10 steps). On each step:

1. The model picks the best tool for the current sub-question
2. The tool executes (Yahoo Finance, FMP, or browser)
3. Results stream back to the client over WebSocket
4. The scratchpad is updated with the result and token usage
5. The model decides whether to call another tool or write its final answer

### Graceful Fallback Chain

The agent never gives up on data. It follows a strict fallback order:

```
1. Yahoo Finance REST API  (free, no auth)
2. Financial Modeling Prep /stable/* API  (requires FMP_API_KEY)
3. Cloudflare Browser Rendering (Puppeteer)  — scrapes finance.yahoo.com or stockanalysis.com
```

Both `getTickerDetails` and `getFinancials` automatically fall through to the browser if their primary API returns an error, making the agent resilient to rate limits and credential issues.

### MCP Server Support

Users can connect their own [Model Context Protocol](https://modelcontextprotocol.io/) servers at runtime. The agent exposes three callable RPC methods from the frontend:

| Method | Description |
|---|---|
| `addServer(name, url, host)` | Register an MCP server (HTTPS only) |
| `removeServer(serverId)` | Disconnect an MCP server |
| `resetScratchpad()` | Clear research state |
| `getScratchpad()` | Read current research state |

MCP tools are merged into the tool set at inference time via `this.mcp.getAITools()`, so the model treats user-added tools the same as built-in ones.

### System Prompt Injection

The system prompt is rebuilt on every request to include today's date and the current list of available tools (built-in + MCP). This keeps the model grounded in real time and prevents it from inventing tool names:

```ts
system: buildSystemPrompt(mcpToolNames),
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Cloudflare Workers |
| Agent framework | Agents SDK (`agents` package) |
| Chat layer | `@cloudflare/ai-chat` (`AIChatAgent`) |
| AI model | Workers AI — `@cf/zai-org/glm-4.7-flash` |
| AI streaming | Vercel AI SDK (`streamText`, `pruneMessages`) |
| Durable state | Durable Objects + SQLite |
| Browser scraping | Cloudflare Browser Rendering (`@cloudflare/puppeteer`) |
| Frontend | React 19 + Vite |
| UI components | `@cloudflare/kumo` |
| Icons | `@phosphor-icons/react` |
| Markdown rendering | `streamdown` |
| Schema validation | Zod v4 |
| Styling | Tailwind CSS v4 |
| Financial APIs | Yahoo Finance (free), Financial Modeling Prep |
| Protocol | Model Context Protocol (MCP) |

---

## Project Structure

```
financial-research-agent/
├── src/
│   ├── server.ts        # Durable Object agent + all tools + Worker entrypoint
│   ├── app.tsx          # React 19 frontend (chat UI, tool rendering, MCP panel)
│   ├── client.tsx       # React DOM entry point
│   ├── prompts.ts       # System prompt builder + context summary prompts
│   ├── types.ts         # AgentState, Scratchpad, ToolResult interfaces
│   └── styles.css       # Tailwind + custom CSS variables
├── public/
│   └── favicon.ico
├── env.d.ts             # Wrangler-generated Cloudflare env types
├── wrangler.jsonc       # Cloudflare Workers config (bindings, migrations)
├── vite.config.ts       # Vite + Cloudflare plugin + Tailwind
├── tsconfig.json
├── package.json
└── index.html
```

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/install-and-update/) (`npm install -g wrangler`)
- A Cloudflare account (free tier works)
- A Financial Modeling Prep API key (free tier: 250 req/day — [sign up here](https://financialmodelingprep.com/developer/docs))

### 1. Clone and install

```bash
git clone https://github.com/aymanarefin/financial-research-agent.git
cd financial-research-agent
npm install
```

### 2. Configure secrets

Create a `.dev.vars` file for local development (this file is gitignored — never commit it):

```bash
# .dev.vars
FMP_API_KEY=your_fmp_api_key_here
```

For production, set the secret via Wrangler:

```bash
wrangler secret put FMP_API_KEY
```

### 3. Run locally

```bash
npm run dev
```

The agent will be available at `http://localhost:5173`.

### 4. Deploy to Cloudflare

```bash
npm run deploy
```

This runs `vite build && wrangler deploy`. Wrangler will:
- Create the Durable Object namespace and SQLite migration
- Bind Workers AI and Browser Rendering
- Deploy the static frontend as Workers Static Assets

---

## Cloudflare Bindings

Defined in `wrangler.jsonc`:

| Binding | Type | Purpose |
|---|---|---|
| `AI` | Workers AI | Run the GLM 4.7 Flash model for inference |
| `BROWSER` | Browser Rendering | Headless Puppeteer for financial page scraping |
| `FinancialResearchAgent` | Durable Object (SQLite) | Per-session agent state and conversation history |
| `FMP_API_KEY` | Secret / env var | Financial Modeling Prep API authentication |

---

## Example Queries

```
What's the current price and market cap of NVIDIA?
Compare Apple and Microsoft's P/E ratios
Show me Tesla's revenue trend over the last 4 quarters
What's Bitcoin doing today?
Get me Amazon's latest income statement
Any recent news on Palantir?
What's the YTD performance of SPY vs QQQ?
```

---

## Customization

### Add a new built-in tool

In `src/server.ts`, add a new entry inside the `tools: { ... }` object passed to `streamText`:

```ts
myNewTool: tool({
  description: "What this tool does",
  inputSchema: z.object({ param: z.string() }),
  execute: async ({ param }) => {
    // fetch data, return result
  },
}),
```

Then add the tool name to `builtInToolNames` so the system prompt lists it.

### Connect an MCP server

Click the **MCP** icon in the chat UI, enter the server name and HTTPS URL, and click Connect. The agent will automatically include all tools from the MCP server in subsequent requests.

### Change the AI model

In `src/server.ts`, replace the model string:

```ts
model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
```

Note: the model must support structured tool calls through the Workers AI binding.

---

## License

MIT — see [LICENSE](./LICENSE).
