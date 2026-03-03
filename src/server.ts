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
import { buildSystemPrompt, buildFinalAnswerPrompt as _buildFinalAnswerPrompt } from "./prompts";
import {
  type AgentState,
  type ToolResult,
  INITIAL_STATE,
  MAX_ITERATIONS,
} from "./types";

const POLYGON_BASE = "https://api.polygon.io";

export class FinancialResearchAgent extends AIChatAgent<Env, AgentState> {
  initialState = INITIAL_STATE;

  // Wait for any user-added MCP connections to restore after hibernation
  waitForMcpConnections = true;

  async onStart() {
    // Remove any MCP servers that are in a failed state from previous runs.
    // This prevents stale persisted connections (e.g. from a wrong URL in a
    // prior deploy) from generating "failed" warnings on every agent wake.
    const mcpState = this.getMcpServers();
    for (const [id, server] of Object.entries(mcpState.servers)) {
      if (server.state === "failed") {
        try {
          await this.removeMcpServer(id);
        } catch {
          // Non-fatal — best effort cleanup
        }
      }
    }

    // Configure OAuth popup behavior for user-added MCP servers
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
  }

  /** Helper: fetch from Polygon.io REST API with API key auth */
  private async polygonFetch(path: string, params: Record<string, string> = {}): Promise<unknown> {
    const url = new URL(`${POLYGON_BASE}${path}`);
    url.searchParams.set("apiKey", this.env.POLYGON_API_KEY);
    for (const [k, v] of Object.entries(params)) {
      url.searchParams.set(k, v);
    }
    const res = await fetch(url.toString());
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Polygon.io API error ${res.status}: ${text.slice(0, 200)}`);
    }
    return res.json();
  }

  // ── Callable RPC methods (invoked from frontend via agent.call()) ──

  @callable()
  async addServer(name: string, url: string, host: string) {
    // Enforce HTTPS-only MCP server URLs to prevent protocol downgrade attacks
    if (!url.startsWith("https://")) {
      throw new Error("MCP server URL must use HTTPS");
    }
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
    const workersai = createWorkersAI({ binding: this.env.AI });

    // Built-in Polygon.io tool names always available (+ any user-added MCP tools)
    const builtInToolNames = [
      "getSnapshot", "getAggregates", "getTickerDetails", "getTickerNews",
      "getFinancials", "resolveTickerSymbol", "inferDateRange",
    ];
    const mcpToolNames = [...builtInToolNames, ...Object.keys(mcpTools)];

    // Track current query for scratchpad context
    const userMessages = this.messages.filter((m) => m.role === "user");
    const latestUserMessage = userMessages[userMessages.length - 1];
    if (latestUserMessage) {
      const text =
        (latestUserMessage.parts as any[])?.find((p: { type: string }) => p.type === "text")?.text ??
        String((latestUserMessage as any).content ?? "");
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
      // Hermes 2 Pro is specifically trained for function/tool calling on Workers AI.
      // The 70B fp8-fast model outputs tool calls as raw text instead of API-level
      // tool_calls, making the Vercel AI SDK unable to intercept and execute them.
      model: workersai("@hf/nousresearch/hermes-2-pro-mistral-7b"),
      // Explicitly set toolChoice so workers-ai-provider sends tool_choice:"auto"
      // rather than undefined, which some Workers AI models require.
      toolChoice: "auto",
      system: buildSystemPrompt(mcpToolNames),
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
      }),
      tools: {
        // All financial data tools from Polygon.io MCP
        ...mcpTools,

        // ── Polygon.io REST API tools ──────────────────────────────

        // Get latest price snapshot for a stock/crypto/forex ticker
        getSnapshot: tool({
          description:
            "Get the current price, day change, and trading volume for a ticker. Use this for 'what is X trading at' or 'current price of X' queries.",
          inputSchema: z.object({
            ticker: z.string().describe("Ticker symbol e.g. AAPL, X:BTCUSD, C:EURUSD"),
            market: z.enum(["stocks", "crypto", "forex"]).default("stocks").describe("Market type"),
          }),
          execute: async ({ ticker, market }) => {
            const path =
              market === "stocks"
                ? `/v2/snapshot/locale/us/markets/stocks/tickers/${ticker}`
                : market === "crypto"
                  ? `/v2/snapshot/locale/global/markets/crypto/tickers/${ticker}`
                  : `/v2/snapshot/locale/global/markets/forex/tickers/${ticker}`;
            return this.polygonFetch(path);
          },
        }),

        // Get OHLCV aggregate bars for historical price data
        getAggregates: tool({
          description:
            "Get historical OHLCV (open/high/low/close/volume) price bars for a ticker over a date range. Use for performance charts and trend analysis.",
          inputSchema: z.object({
            ticker: z.string().describe("Ticker symbol e.g. AAPL"),
            from: z.string().describe("Start date YYYY-MM-DD"),
            to: z.string().describe("End date YYYY-MM-DD"),
            timespan: z.enum(["day", "week", "month"]).default("day").describe("Bar size"),
            multiplier: z.number().int().default(1).describe("Timespan multiplier"),
          }),
          execute: async ({ ticker, from, to, timespan, multiplier }) => {
            return this.polygonFetch(
              `/v2/aggs/ticker/${ticker}/range/${multiplier}/${timespan}/${from}/${to}`,
              { adjusted: "true", sort: "asc", limit: "120" }
            );
          },
        }),

        // Get company details: description, market cap, employees, SIC
        getTickerDetails: tool({
          description:
            "Get company profile information: description, market cap, employee count, sector, exchange. Use this to understand what a company does.",
          inputSchema: z.object({
            ticker: z.string().describe("Ticker symbol e.g. AAPL"),
          }),
          execute: async ({ ticker }) => {
            return this.polygonFetch(`/v3/reference/tickers/${ticker}`);
          },
        }),

        // Get recent news articles for a ticker
        getTickerNews: tool({
          description:
            "Get the latest news articles and sentiment for a ticker. Use this for recent news context, earnings coverage, and market sentiment.",
          inputSchema: z.object({
            ticker: z.string().describe("Ticker symbol e.g. AAPL"),
            limit: z.number().int().min(1).max(10).default(5).describe("Number of articles"),
          }),
          execute: async ({ ticker, limit }) => {
            return this.polygonFetch("/v2/reference/news", {
              ticker,
              limit: String(limit),
              order: "desc",
              sort: "published_utc",
            });
          },
        }),

        // Get financials: income statement, balance sheet, cash flow
        getFinancials: tool({
          description:
            "Get quarterly or annual financial statements (revenue, net income, EPS, assets, liabilities, cash flow) for a company. Use for fundamental analysis.",
          inputSchema: z.object({
            ticker: z.string().describe("Ticker symbol e.g. AAPL"),
            timeframe: z.enum(["quarterly", "annual"]).default("quarterly"),
            limit: z.number().int().min(1).max(8).default(4).describe("Number of periods"),
          }),
          execute: async ({ ticker, timeframe, limit }) => {
            return this.polygonFetch("/vX/reference/financials", {
              ticker,
              timeframe,
              limit: String(limit),
              order: "desc",
              sort: "period_of_report_date",
            });
          },
        }),

        // ── Built-in utility tools ──────────────────────────────────

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
        const toolCalls = (step as any).toolCalls ?? [];
        const toolResults = (step as any).toolResults ?? [];

        if (toolCalls.length === 0) return;

        const newResults: ToolResult[] = toolCalls.map((call: any) => ({
          tool: call.toolName,
          input: call.input as Record<string, unknown>,
          result: toolResults.find((r: any) => r.toolCallId === call.toolCallId)?.output ?? null,
          timestamp: new Date().toISOString(),
          durationMs: 0,
        }));

        const usage = step.usage;
        const tokensThisStep =
          (usage?.inputTokens ?? 0) + (usage?.outputTokens ?? 0);

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
