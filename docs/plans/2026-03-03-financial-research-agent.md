# Financial Research Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the existing `agent-starter` ChatAgent into Dexter's financial research agent architecture — an iterative, scratchpad-driven agent that connects to Polygon.io via MCP for real-time financial data.

**Architecture:** The agent extends `AIChatAgent` (already installed) and adds: (1) a financial-research system prompt that injects available MCP tool descriptions, (2) a scratchpad state object tracking iterations, tool results, and token usage, and (3) an `onStart()` hook that pre-connects to the Polygon.io MCP server. The frontend gets a financial-themed UI with suggested research prompts and a collapsible scratchpad panel.

**Tech Stack:** TypeScript, `agents` SDK v0.7+, `@cloudflare/ai-chat`, Vercel AI SDK `streamText`, Workers AI (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`), Polygon.io MCP (HTTP/SSE), `@cloudflare/kumo` UI, Tailwind CSS, `zod`

---

## Existing File Inventory

```
src/server.ts     ← Main agent (transform in place)
src/app.tsx       ← Frontend (update UI copy + prompts)
src/client.tsx    ← Entry point (no changes needed)
src/styles.css    ← Styles (no changes needed)
wrangler.jsonc    ← Config (add POLYGON_API_KEY var)
env.d.ts          ← Types (add POLYGON_API_KEY)
```

New files to create:
```
src/types.ts      ← Shared TypeScript types
src/prompts.ts    ← Prompt management system
```

---

## Key Design Decisions

### Why keep AIChatAgent?
`AIChatAgent` from `@cloudflare/ai-chat` already handles:
- Message history persistence (`this.messages`)
- Streaming via `toUIMessageStreamResponse()`
- Tool approval flow
- Resumable streaming on reconnect

We layer Dexter's patterns **on top** rather than replacing it.

### Dexter's Iterative Loop → `stopWhen: stepCountIs(N)`
The Vercel AI SDK's `stepCountIs(N)` already implements the max-iterations pattern. We set it to `10` (matching Dexter's default) and track each step in scratchpad state via `onStepFinish`.

### MCP for Financial Data
Polygon.io's MCP server (`https://mcp.polygon.io/sse`) exposes tools like:
- `get_ticker_details` — company info, market cap
- `get_stock_price` — latest trade/quote
- `get_aggregates` — OHLCV bars (day/hour/minute)
- `get_news` — ticker-specific news articles
- `get_financials` — income statement, balance sheet, cash flow

Connection is established in `onStart()` with the API key passed as a header. Tools are injected into the AI SDK via `this.mcp.getAITools()`.

### Scratchpad State
```typescript
interface Scratchpad {
  iterations: number;
  toolResults: Array<{ tool: string; result: unknown; timestamp: string }>;
  totalTokensUsed: number;
  researchQuery: string | null;
  contextThreshold: number; // max tokens before summarization
}
```

State is stored via `this.setState()` so it persists across hibernation and syncs to the frontend in real-time.

---

## Task 1: Create Shared Type Definitions

**Files:**
- Create: `src/types.ts`

### Step 1: Write the types file

```typescript
// src/types.ts

export interface ToolResult {
  tool: string;
  input: Record<string, unknown>;
  result: unknown;
  timestamp: string;
  durationMs: number;
}

export interface Scratchpad {
  iterations: number;
  toolResults: ToolResult[];
  totalTokensUsed: number;
  researchQuery: string | null;
  /** Token count at which context summarization kicks in */
  contextThreshold: number;
}

export interface AgentState {
  scratchpad: Scratchpad;
}

export const INITIAL_STATE: AgentState = {
  scratchpad: {
    iterations: 0,
    toolResults: [],
    totalTokensUsed: 0,
    researchQuery: null,
    contextThreshold: 50_000,
  },
};

export const MAX_ITERATIONS = 10;
export const POLYGON_MCP_URL = "https://mcp.polygon.io/sse";
```

### Step 2: Verify no lint errors after creation

Run: `cd financial-research-agent && npx tsc --noEmit 2>&1 | head -20`
Expected: no errors referencing `types.ts`

### Step 3: Commit

```bash
git add src/types.ts
git commit -m "feat: add shared type definitions for financial agent state"
```

---

## Task 2: Create Prompt Management System

**Files:**
- Create: `src/prompts.ts`

### Step 1: Write the prompts file

```typescript
// src/prompts.ts

import type { Scratchpad } from "./types";

/**
 * Builds the financial research system prompt.
 * Injected with today's date and available MCP tool names for LLM awareness.
 */
export function buildSystemPrompt(mcpToolNames: string[]): string {
  const today = new Date().toISOString().split("T")[0];
  const toolList =
    mcpToolNames.length > 0
      ? mcpToolNames.map((t) => `  - ${t}`).join("\n")
      : "  (no MCP tools connected yet)";

  return `You are a financial research analyst with access to real-time and historical market data.

Today's date: ${today}

## Your Capabilities
You can research stocks, ETFs, cryptocurrencies, and financial news by using the tools available to you.
When a user asks about a company or ticker, use multiple tools in sequence to build a comprehensive picture.

## Available Financial Data Tools
${toolList}

## Research Methodology
1. Identify the ticker symbol(s) relevant to the query
2. Gather current price and recent performance data
3. Retrieve company fundamentals if available
4. Fetch recent news for sentiment context
5. Synthesize findings into a clear, structured response

## Response Format
- Use markdown headers to organize sections (## Price, ## Fundamentals, ## News)
- Include specific numbers with units (e.g., "$182.34", "+2.3%", "P/E: 28.5x")
- Cite the data source (Polygon.io) and timestamp when possible
- If data is unavailable for a field, say so explicitly rather than guessing
- End with a brief "Key Takeaways" section with 2-3 bullet points

## Limitations
- You have real-time market data but cannot predict future prices
- Historical data older than 2 years may require specific date parameters
- Crypto data is available 24/7; equity data reflects last market session`;
}

/**
 * Builds the context summary prompt when approaching the token threshold.
 * Used to compress earlier tool results before continuing.
 */
export function buildContextSummaryPrompt(scratchpad: Scratchpad): string {
  const toolSummaries = scratchpad.toolResults
    .slice(0, -2) // keep last 2 raw results
    .map((r) => `[${r.tool} @ ${r.timestamp}]: ${JSON.stringify(r.result).slice(0, 200)}...`)
    .join("\n");

  return `Summarize the following tool results into a concise context paragraph (max 300 words) that preserves all key financial data points:

${toolSummaries}`;
}

/**
 * Builds the final answer prompt when max iterations are reached.
 * Instructs the model to synthesize everything in the scratchpad.
 */
export function buildFinalAnswerPrompt(scratchpad: Scratchpad): string {
  const allResults = scratchpad.toolResults
    .map((r) => `### ${r.tool}\nInput: ${JSON.stringify(r.input)}\nResult: ${JSON.stringify(r.result, null, 2)}`)
    .join("\n\n");

  return `You have completed ${scratchpad.iterations} research iterations. 
Based on all the data collected below, provide a comprehensive final answer.

## Collected Research Data
${allResults}

## Instructions
Synthesize all data into a well-structured financial research report.
Include: current price, performance metrics, key fundamentals, recent news sentiment, and risks.
End with 3 actionable takeaways.`;
}
```

### Step 2: Verify TypeScript compiles

Run: `npx tsc --noEmit 2>&1 | head -20`
Expected: no errors

### Step 3: Commit

```bash
git add src/prompts.ts
git commit -m "feat: add financial research prompt management system"
```

---

## Task 3: Transform the Main Agent

**Files:**
- Modify: `src/server.ts` (full replacement)

This is the core transformation. We keep `AIChatAgent` as the base class but replace the system prompt, tools, and state management with financial-research-specific logic.

### Step 1: Write the new server.ts

```typescript
// src/server.ts
import { createWorkersAI } from "workers-ai-provider";
import { routeAgentRequest, callable } from "agents";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs,
} from "ai";
import { z } from "zod";
import { buildSystemPrompt, buildFinalAnswerPrompt } from "./prompts";
import {
  type AgentState,
  type ToolResult,
  INITIAL_STATE,
  MAX_ITERATIONS,
  POLYGON_MCP_URL,
} from "./types";

export class FinancialResearchAgent extends AIChatAgent<Env, AgentState> {
  initialState = INITIAL_STATE;

  // Wait for MCP connections (Polygon.io) to restore after hibernation
  waitForMcpConnections = true;

  async onStart() {
    // Configure OAuth popup behavior for MCP servers
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200,
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      },
    });

    // Pre-connect to Polygon.io MCP if API key is configured
    if (this.env.POLYGON_API_KEY) {
      try {
        await this.addMcpServer("polygon", POLYGON_MCP_URL, {
          transport: {
            type: "sse",
            headers: {
              Authorization: `Bearer ${this.env.POLYGON_API_KEY}`,
            },
          },
        });
        console.log("Connected to Polygon.io MCP server");
      } catch (err) {
        // Non-fatal: agent works without MCP, user can add manually
        console.warn("Failed to connect to Polygon.io MCP:", err);
      }
    }
  }

  // ── Callable RPC methods (invoked from frontend via agent.call()) ──

  @callable()
  async addServer(name: string, url: string, host: string) {
    return await this.addMcpServer(name, url, { callbackHost: host });
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  @callable()
  resetScratchpad() {
    this.setState(INITIAL_STATE);
    return { success: true };
  }

  @callable()
  getScratchpad() {
    return this.state.scratchpad;
  }

  // ── Core chat handler ──

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const mcpToolNames = Object.keys(mcpTools);
    const workersai = createWorkersAI({ binding: this.env.AI });

    // Track current query for scratchpad context
    const latestUserMessage = this.messages
      .filter((m) => m.role === "user")
      .at(-1);
    if (latestUserMessage) {
      const text =
        latestUserMessage.parts?.find((p) => p.type === "text")?.text ??
        String(latestUserMessage.content ?? "");
      if (text) {
        this.setState({
          ...this.state,
          scratchpad: {
            ...this.state.scratchpad,
            researchQuery: text,
            iterations: 0,
          },
        });
      }
    }

    const result = streamText({
      model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
      system: buildSystemPrompt(mcpToolNames),
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
      }),
      tools: {
        // All financial data tools from Polygon.io MCP
        ...mcpTools,

        // Built-in: ticker normalization helper
        resolveTickerSymbol: tool({
          description:
            "Normalize a company name or partial ticker to its canonical ticker symbol. Use this FIRST when the user provides a company name instead of a ticker.",
          inputSchema: z.object({
            query: z
              .string()
              .describe(
                "Company name or partial ticker, e.g. 'Apple' or 'APPL'"
              ),
          }),
          execute: async ({ query }) => {
            // Simple normalization map for common names — MCP handles the rest
            const COMMON: Record<string, string> = {
              apple: "AAPL",
              microsoft: "MSFT",
              google: "GOOGL",
              alphabet: "GOOGL",
              amazon: "AMZN",
              meta: "META",
              facebook: "META",
              nvidia: "NVDA",
              tesla: "TSLA",
              netflix: "NFLX",
              "berkshire hathaway": "BRK.B",
              bitcoin: "X:BTCUSD",
              ethereum: "X:ETHUSD",
            };
            const normalized = COMMON[query.toLowerCase().trim()];
            return {
              original: query,
              ticker: normalized ?? query.toUpperCase(),
              note: normalized
                ? "Matched from common names list"
                : "Passed through as-is — verify with get_ticker_details",
            };
          },
        }),

        // Built-in: date inference helper
        inferDateRange: tool({
          description:
            "Convert natural language time expressions to concrete date strings (YYYY-MM-DD). Use before any historical data queries.",
          inputSchema: z.object({
            expression: z
              .string()
              .describe(
                "Natural language like 'last month', 'past 3 months', 'YTD', 'last year'"
              ),
          }),
          execute: async ({ expression }) => {
            const now = new Date();
            const expr = expression.toLowerCase();
            let fromDate: Date;

            if (expr.includes("week")) {
              fromDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
            } else if (expr.includes("month") && expr.includes("3")) {
              fromDate = new Date(now.getFullYear(), now.getMonth() - 3, 1);
            } else if (expr.includes("month") && expr.includes("6")) {
              fromDate = new Date(now.getFullYear(), now.getMonth() - 6, 1);
            } else if (expr.includes("month")) {
              fromDate = new Date(now.getFullYear(), now.getMonth() - 1, 1);
            } else if (expr.includes("ytd") || expr.includes("year to date")) {
              fromDate = new Date(now.getFullYear(), 0, 1);
            } else if (expr.includes("year")) {
              fromDate = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
            } else {
              fromDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
            }

            return {
              from: fromDate.toISOString().split("T")[0],
              to: now.toISOString().split("T")[0],
              expression,
            };
          },
        }),
      },

      // Dexter's max-iterations pattern
      stopWhen: stepCountIs(MAX_ITERATIONS),

      // Track each tool step in the scratchpad
      onStepFinish: async (step) => {
        const toolCalls = step.toolCalls ?? [];
        const toolResults = step.toolResults ?? [];

        if (toolCalls.length === 0) return;

        const newResults: ToolResult[] = toolCalls.map((call, i) => ({
          tool: call.toolName,
          input: call.input as Record<string, unknown>,
          result: toolResults[i]?.output ?? null,
          timestamp: new Date().toISOString(),
          durationMs: 0,
        }));

        const usage = step.usage;
        const tokensThisStep =
          (usage?.promptTokens ?? 0) + (usage?.completionTokens ?? 0);

        this.setState({
          ...this.state,
          scratchpad: {
            ...this.state.scratchpad,
            iterations: this.state.scratchpad.iterations + 1,
            toolResults: [
              ...this.state.scratchpad.toolResults,
              ...newResults,
            ],
            totalTokensUsed:
              this.state.scratchpad.totalTokensUsed + tokensThisStep,
          },
        });
      },

      abortSignal: options?.abortSignal,
    });

    return result.toUIMessageStreamResponse();
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  },
} satisfies ExportedHandler<Env>;
```

### Step 2: Run TypeScript check

Run: `npx tsc --noEmit 2>&1`
Expected: zero errors

### Step 3: Run dev server briefly to verify it starts

Run: `npm run dev -- --port 8788 &` (background), wait 3s, then `kill %1`
Expected: "Ready on http://localhost:8788" in output

### Step 4: Commit

```bash
git add src/server.ts
git commit -m "feat: transform ChatAgent into FinancialResearchAgent with MCP + scratchpad"
```

---

## Task 4: Update Environment Type Definitions

**Files:**
- Modify: `env.d.ts`
- Modify: `wrangler.jsonc`

### Step 1: Update env.d.ts to add POLYGON_API_KEY

Change the `Env` interface (or `interface Env {}` block) to add:
```typescript
POLYGON_API_KEY: string;
```

### Step 2: Update wrangler.jsonc

Add under the `"name"` key:
```jsonc
"vars": {
  "POLYGON_API_KEY": ""
}
```

And update the `"durable_objects"` binding class_name from `"ChatAgent"` to `"FinancialResearchAgent"`:
```jsonc
"durable_objects": {
  "bindings": [
    {
      "class_name": "FinancialResearchAgent",
      "name": "FinancialResearchAgent"
    }
  ]
},
"migrations": [
  {
    "new_sqlite_classes": ["FinancialResearchAgent"],
    "tag": "v1"
  }
]
```

### Step 3: Verify wrangler config parses

Run: `npx wrangler types env.d.ts --include-runtime false 2>&1 | head -20`
Expected: regenerates env.d.ts without errors

### Step 4: Commit

```bash
git add wrangler.jsonc env.d.ts
git commit -m "feat: update wrangler config for FinancialResearchAgent + Polygon API key"
```

---

## Task 5: Update the React Frontend

**Files:**
- Modify: `src/app.tsx` (targeted changes only — not a full rewrite)

### Step 1: Update app branding and agent name

In `app.tsx`, make these targeted changes:

**Change 1** — Header title (around line where `"⛅"` emoji and `"Agent Starter"` appear):
```tsx
// FROM:
<h1 className="text-lg font-semibold text-kumo-default">
  <span className="mr-2">⛅</span>Agent Starter
</h1>

// TO:
<h1 className="text-lg font-semibold text-kumo-default">
  <span className="mr-2">📈</span>Financial Research Agent
</h1>
```

**Change 2** — `useAgent` agent name (from `"ChatAgent"` to `"FinancialResearchAgent"`):
```tsx
// FROM:
const agent = useAgent({
  agent: "ChatAgent",

// TO:
const agent = useAgent({
  agent: "FinancialResearchAgent",
```

**Change 3** — Suggested prompts in the Empty state (replace the 4 weather/timezone prompts):
```tsx
// FROM:
{[
  "What's the weather in Paris?",
  "What timezone am I in?",
  "Calculate 5000 * 3",
  "Remind me in 5 minutes to take a break"
].map((prompt) => (

// TO:
{[
  "Research Apple stock — price, fundamentals, and recent news",
  "How has NVDA performed over the past 3 months?",
  "Compare MSFT vs GOOGL revenue growth",
  "What are Bitcoin and Ethereum doing today?",
  "Show me Tesla's latest quarterly financials",
  "What's the P/E ratio for the S&P 500 ETF (SPY)?"
].map((prompt) => (
```

**Change 4** — Add scratchpad panel state and display. Insert after the `mcpState`/`showMcpPanel` state declarations:
```tsx
const [showScratchpad, setShowScratchpad] = useState(false);
const [scratchpad, setScratchpad] = useState<{
  iterations: number;
  toolResults: Array<{ tool: string; timestamp: string }>;
  totalTokensUsed: number;
  researchQuery: string | null;
} | null>(null);
```

Update `onStateUpdate` in `useAgent` to capture scratchpad:
```tsx
onStateUpdate: useCallback((state: unknown) => {
  const s = state as { scratchpad?: typeof scratchpad };
  if (s?.scratchpad) setScratchpad(s.scratchpad);
}, []),
```

**Change 5** — Add scratchpad button to the header toolbar (next to the MCP button):
```tsx
<Button
  variant="secondary"
  size="sm"
  onClick={() => setShowScratchpad((v) => !v)}
  icon={<BrainIcon size={14} />}
>
  Scratchpad
  {scratchpad && scratchpad.iterations > 0 && (
    <Badge variant="secondary" className="ml-1">
      {scratchpad.iterations}
    </Badge>
  )}
</Button>
```

**Change 6** — Add scratchpad panel below the MCP panel (conditional render):
```tsx
{showScratchpad && scratchpad && (
  <Surface className="absolute top-full right-0 mt-2 w-80 p-4 shadow-xl z-50 space-y-3">
    <Text size="sm" bold>Research Scratchpad</Text>
    {scratchpad.researchQuery && (
      <div>
        <Text size="xs" variant="secondary">Query</Text>
        <Text size="xs">{scratchpad.researchQuery}</Text>
      </div>
    )}
    <div className="flex gap-4">
      <div>
        <Text size="xs" variant="secondary">Iterations</Text>
        <Text size="sm" bold>{scratchpad.iterations} / 10</Text>
      </div>
      <div>
        <Text size="xs" variant="secondary">Tokens Used</Text>
        <Text size="sm" bold>{scratchpad.totalTokensUsed.toLocaleString()}</Text>
      </div>
    </div>
    {scratchpad.toolResults.length > 0 && (
      <div>
        <Text size="xs" variant="secondary" className="mb-1">Tool Calls</Text>
        <div className="space-y-1 max-h-40 overflow-y-auto">
          {scratchpad.toolResults.map((r, i) => (
            <div key={i} className="flex items-center gap-2">
              <WrenchIcon size={10} className="text-kumo-subtle" />
              <Text size="xs">{r.tool}</Text>
              <Text size="xs" variant="secondary" className="ml-auto">
                {new Date(r.timestamp).toLocaleTimeString()}
              </Text>
            </div>
          ))}
        </div>
      </div>
    )}
    <Button
      variant="ghost"
      size="sm"
      onClick={() => agent.call("resetScratchpad", [])}
    >
      Reset Scratchpad
    </Button>
  </Surface>
)}
```

### Step 2: Run TypeScript check

Run: `npx tsc --noEmit 2>&1`
Expected: zero errors

### Step 3: Run dev server and verify it loads

Run: `npm run dev`
Expected: app loads at `localhost:5173` with "Financial Research Agent" title

### Step 4: Commit

```bash
git add src/app.tsx
git commit -m "feat: update frontend for financial research — branding, prompts, scratchpad panel"
```

---

## Task 6: Configure Polygon.io MCP Connection

**This task is a setup step, not a code change.**

### Step 1: Get a Polygon.io API key

1. Sign up at https://polygon.io (free tier available)
2. Copy your API key from the dashboard

### Step 2: Set the secret for local dev

```bash
# For local development, add to .dev.vars (gitignored)
echo "POLYGON_API_KEY=your_api_key_here" >> .dev.vars
```

### Step 3: Verify .dev.vars is gitignored

Run: `cat .gitignore | grep dev.vars`
Expected: `.dev.vars` appears in output

If not:
```bash
echo ".dev.vars" >> .gitignore
git add .gitignore && git commit -m "chore: gitignore .dev.vars"
```

### Step 4: For production deployment

```bash
npx wrangler secret put POLYGON_API_KEY
# Paste your key when prompted
```

### Step 5: Verify MCP connection in dev

Run: `npm run dev`
In browser console or wrangler tail output, look for:
`"Connected to Polygon.io MCP server"`

If you see `"Failed to connect to Polygon.io MCP"`, the agent still works — you can connect manually via the MCP panel in the UI using:
- Name: `polygon`
- URL: `https://mcp.polygon.io/sse`

---

## Task 7: End-to-End Verification

### Step 1: Start dev server

```bash
npm run dev
```

Open `http://localhost:5173`

### Step 2: Verify branding

- Header shows "📈 Financial Research Agent"
- Empty state shows 6 financial research prompts
- Scratchpad button appears in header

### Step 3: Test a research query

Click: "Research Apple stock — price, fundamentals, and recent news"

Expected behavior:
1. Agent calls `resolveTickerSymbol` → returns `AAPL`
2. Agent calls Polygon.io MCP tools in sequence
3. Scratchpad panel shows iterations counting up (1→2→3...)
4. Response is structured with ## Price, ## Fundamentals, ## News sections

### Step 4: Test scratchpad state sync

- Open Scratchpad panel
- Verify iteration count matches number of tool calls shown
- Verify token usage increases with each message
- Click "Reset Scratchpad" → counters return to zero

### Step 5: Test MCP panel still works

- Open MCP panel
- Verify Polygon.io server shows as "ready"
- Tool count shows available Polygon.io tools

### Step 6: Final commit

```bash
git add -A
git commit -m "chore: verified financial research agent end-to-end"
```

---

## Deployment Checklist

```bash
# 1. Set production secrets
npx wrangler secret put POLYGON_API_KEY

# 2. Build and deploy
npm run deploy

# 3. Verify deployment
npx wrangler tail

# 4. Test production endpoint
curl https://financial-research-agent.<your-subdomain>.workers.dev/agents/FinancialResearchAgent/test
```

---

## Alternative: Yahoo Finance MCP (Free Tier)

If you prefer not to use Polygon.io, use this URL in `POLYGON_MCP_URL`:

```
https://mcp-yahoo-finance.vercel.app/mcp
```

No API key required. Update `onStart()` in `server.ts`:
```typescript
await this.addMcpServer("yahoo-finance", YAHOO_FINANCE_MCP_URL);
```

Update `POLYGON_MCP_URL` constant in `types.ts` or add:
```typescript
export const YAHOO_FINANCE_MCP_URL = "https://mcp-yahoo-finance.vercel.app/mcp";
```

Note: Yahoo Finance MCP has rate limits and less reliable uptime than Polygon.io.

---

## Architecture Diagram

```
Browser (React)
  │
  ├── useAgent("FinancialResearchAgent") ─── WebSocket ──► Agent (Durable Object)
  │       │                                                    │
  │       ├── onStateUpdate(scratchpad)                        ├── onStart() → addMcpServer(polygon)
  │       └── agent.call("resetScratchpad")                    │
  │                                                            ├── onChatMessage()
  │                                                            │     ├── buildSystemPrompt(mcpToolNames)
  │                                                            │     ├── streamText(model, tools, stopWhen: stepCountIs(10))
  │                                                            │     │     ├── resolveTickerSymbol (built-in)
  │                                                            │     │     ├── inferDateRange (built-in)
  │                                                            │     │     └── [Polygon.io MCP tools]
  │                                                            │     └── onStepFinish → setState(scratchpad++)
  │                                                            │
  │                                                            └── SQLite (state, messages, MCP tokens)
  │
  └── MCP Panel (manual server management)
```
