// src/server.ts
import puppeteer from "@cloudflare/puppeteer";
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

const YAHOO_Q1 = "https://query1.finance.yahoo.com";
const YAHOO_Q2 = "https://query2.finance.yahoo.com";
// FMP migrated from /api/v3/ (legacy, deprecated Aug 2025) to /stable/ for new API keys
const FMP_BASE = "https://financialmodelingprep.com";

export class FinancialResearchAgent extends AIChatAgent<Env, AgentState> {
  initialState = INITIAL_STATE;

  // Wait for any user-added MCP connections to restore after hibernation
  waitForMcpConnections = true;

  async onStart() {
    // Remove any MCP servers that are in a failed state from previous runs.
    // This prevents stale persisted connections from generating warnings on every agent wake.
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

  /** Helper: fetch from Yahoo Finance APIs (no auth required) */
  private async yahooFetch(base: string, path: string, params: Record<string, string> = {}): Promise<unknown> {
    const url = new URL(`${base}${path}`);
    for (const [k, v] of Object.entries(params)) {
      url.searchParams.set(k, v);
    }
    const res = await fetch(url.toString(), {
      headers: {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
      },
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Yahoo Finance API error ${res.status}: ${text.slice(0, 200)}`);
    }
    return res.json();
  }

  /** Helper: fetch from Financial Modeling Prep (new /stable/ endpoints, requires FMP_API_KEY) */
  private async fmpFetch(path: string, params: Record<string, string> = {}): Promise<unknown> {
    const url = new URL(`${FMP_BASE}${path}`);
    url.searchParams.set("apikey", this.env.FMP_API_KEY);
    for (const [k, v] of Object.entries(params)) {
      url.searchParams.set(k, v);
    }
    const res = await fetch(url.toString());
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`FMP API error ${res.status}: ${text.slice(0, 300)}`);
    }
    const data = await res.json() as unknown;
    // FMP returns an error object even on 200 for invalid/legacy endpoint access
    if (typeof data === "object" && data !== null && "Error Message" in data) {
      throw new Error(`FMP error: ${(data as Record<string, string>)["Error Message"]}`);
    }
    return data;
  }

  /** Helper: open a real browser via Cloudflare Browser Rendering and return page text */
  private async browserFetch(url: string): Promise<{ url: string; content: string }> {
    const browser = await puppeteer.launch(this.env.BROWSER);
    try {
      const page = await browser.newPage();
      await page.setUserAgent(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
      );
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
      // Allow JS-rendered content to settle
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const text = await page.$eval("body", (el) => (el as HTMLElement).innerText ?? el.textContent ?? "");
      return {
        url,
        content: text.replace(/\s+/g, " ").trim().slice(0, 8000),
      };
    } finally {
      await browser.close();
    }
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

    // Built-in tool names (Yahoo Finance + FMP + browser fallback + any user-added MCP tools)
    const builtInToolNames = [
      "getSnapshot", "getAggregates", "getTickerDetails", "getTickerNews",
      "getFinancials", "resolveTickerSymbol", "inferDateRange", "browseUrl",
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
      // GLM 4.7 Flash returns proper structured tool_calls through the Workers AI binding.
      // Other models (llama-3.3-70b-fp8, hermes-2-pro) output function calls as raw text
      // which the workers-ai-provider cannot intercept as actual tool invocations.
      model: workersai("@cf/zai-org/glm-4.7-flash"),
      toolChoice: "auto",
      system: buildSystemPrompt(mcpToolNames),
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
      }),
      tools: {
        // All tools from user-added MCP servers
        ...mcpTools,

        // ── Yahoo Finance tools (no API key required) ──────────────

        // Get current price, day change, volume for any ticker
        getSnapshot: tool({
          description:
            "Get the current price, day change %, volume, and market cap for a ticker. Works for stocks (AAPL), crypto (BTC-USD, ETH-USD), and forex (EURUSD=X).",
          inputSchema: z.object({
            ticker: z.string().describe("Yahoo Finance ticker: AAPL for stocks, BTC-USD for crypto, EURUSD=X for forex"),
          }),
          execute: async ({ ticker }) => {
            return this.yahooFetch(YAHOO_Q1, `/v8/finance/chart/${encodeURIComponent(ticker)}`, {
              interval: "1d",
              range: "1d",
            });
          },
        }),

        // Get OHLCV aggregate bars for historical price data
        getAggregates: tool({
          description:
            "Get historical OHLCV (open/high/low/close/volume) price bars for a ticker over a date range. Use for trend analysis and performance charts.",
          inputSchema: z.object({
            ticker: z.string().describe("Yahoo Finance ticker e.g. AAPL, BTC-USD"),
            from: z.string().describe("Start date YYYY-MM-DD"),
            to: z.string().describe("End date YYYY-MM-DD"),
            interval: z.enum(["1d", "1wk", "1mo"]).default("1d").describe("Bar size"),
          }),
          execute: async ({ ticker, from, to, interval }) => {
            const period1 = Math.floor(new Date(from).getTime() / 1000).toString();
            const period2 = Math.floor(new Date(to).getTime() / 1000).toString();
            return this.yahooFetch(YAHOO_Q1, `/v8/finance/chart/${encodeURIComponent(ticker)}`, {
              interval,
              period1,
              period2,
            });
          },
        }),

        // Get company profile using Yahoo Finance v7/quote (works without crumb), with browser fallback
        getTickerDetails: tool({
          description:
            "Get company profile: full name, sector, industry, market cap, P/E ratio (trailing and forward), 52-week high/low, dividend yield, EPS. Automatically falls back to browser if Yahoo Finance fails.",
          inputSchema: z.object({
            ticker: z.string().describe("Yahoo Finance ticker e.g. AAPL"),
          }),
          execute: async ({ ticker }) => {
            try {
              return await this.yahooFetch(YAHOO_Q2, "/v7/finance/quote", {
                symbols: ticker,
              });
            } catch (yahooErr) {
              console.warn(`[getTickerDetails] Yahoo v7/quote failed for ${ticker}, falling back to browser: ${yahooErr}`);
              return this.browserFetch(`https://finance.yahoo.com/quote/${encodeURIComponent(ticker)}`);
            }
          },
        }),

        // Get recent news articles for a ticker
        getTickerNews: tool({
          description:
            "Get the latest news headlines for a ticker. Use for recent news context, earnings coverage, and market sentiment.",
          inputSchema: z.object({
            ticker: z.string().describe("Yahoo Finance ticker e.g. AAPL"),
            count: z.number().int().min(1).max(10).default(5).describe("Number of articles"),
          }),
          execute: async ({ ticker, count }) => {
            return this.yahooFetch(YAHOO_Q2, "/v1/finance/search", {
              q: ticker,
              newsCount: String(count),
              quotesCount: "0",
            });
          },
        }),

        // ── Financial Modeling Prep (requires FMP_API_KEY) ──────────

        // Get financial statements via FMP stable API, with automatic browser fallback
        getFinancials: tool({
          description:
            "Get financial statements (income statement, balance sheet, cash flow) via Financial Modeling Prep. Use for revenue, net income, EPS, assets, liabilities, and cash flow data. Automatically falls back to browser scraping if the API fails.",
          inputSchema: z.object({
            ticker: z.string().describe("Stock ticker e.g. AAPL"),
            period: z.enum(["quarterly", "annual"]).default("quarterly"),
            statement: z.enum(["income", "balance", "cashflow", "all"]).default("income").describe("Which statement(s) to fetch"),
          }),
          execute: async ({ ticker, period, statement }) => {
            const p = period === "quarterly" ? "quarter" : "annual";
            // Try FMP stable endpoints first (new API, not deprecated v3)
            try {
              const results: Record<string, unknown> = {};
              if (statement === "income" || statement === "all") {
                results.incomeStatement = await this.fmpFetch("/stable/income-statement", {
                  symbol: ticker, period: p, limit: "4",
                });
              }
              if (statement === "balance" || statement === "all") {
                results.balanceSheet = await this.fmpFetch("/stable/balance-sheet-statement", {
                  symbol: ticker, period: p, limit: "4",
                });
              }
              if (statement === "cashflow" || statement === "all") {
                results.cashFlow = await this.fmpFetch("/stable/cash-flow-statement", {
                  symbol: ticker, period: p, limit: "4",
                });
              }
              return results;
            } catch (fmpErr) {
              // FMP failed — fall back to browser scraping stockanalysis.com
              console.warn(`[getFinancials] FMP failed for ${ticker}, falling back to browser: ${fmpErr}`);
              const t = ticker.toUpperCase();
              const browserUrl =
                statement === "balance"
                  ? `https://stockanalysis.com/stocks/${t.toLowerCase()}/financials/balance-sheet/`
                  : statement === "cashflow"
                    ? `https://stockanalysis.com/stocks/${t.toLowerCase()}/financials/cash-flow-statement/`
                    : `https://stockanalysis.com/stocks/${t.toLowerCase()}/financials/`;
              return this.browserFetch(browserUrl);
            }
          },
        }),

        // ── Built-in utility tools ──────────────────────────────────

        // Normalize company name to Yahoo Finance ticker symbol
        resolveTickerSymbol: tool({
          description:
            "Normalize a company name or partial ticker to its canonical Yahoo Finance ticker symbol. Call this FIRST when the user provides a company name instead of a ticker.",
          inputSchema: z.object({
            query: z.string().describe("Company name or partial ticker, e.g. 'Apple', 'bitcoin', 'APPL'"),
          }),
          execute: async ({ query }) => {
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
              "berkshire hathaway": "BRK-B",
              bitcoin: "BTC-USD",
              ethereum: "ETH-USD",
              solana: "SOL-USD",
              dogecoin: "DOGE-USD",
              ripple: "XRP-USD",
              cardano: "ADA-USD",
            };
            const normalized = COMMON[query.toLowerCase().trim()];
            return {
              original: query,
              ticker: normalized ?? query.toUpperCase(),
              note: normalized
                ? "Matched from common names list"
                : "Passed through as-is — verify with getTickerDetails",
            };
          },
        }),

        // Convert natural language date expressions to YYYY-MM-DD
        inferDateRange: tool({
          description:
            "Convert natural language time expressions to concrete date strings (YYYY-MM-DD). Call before any historical data queries.",
          inputSchema: z.object({
            expression: z
              .string()
              .describe("Natural language like 'last month', 'past 3 months', 'YTD', 'last year'"),
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

        // ── Browser fallback (Cloudflare Browser Rendering) ─────────

        // Last-resort: navigate to any financial URL and extract text content
        browseUrl: tool({
          description:
            "Navigate to a financial website using a real browser and extract the page text. Use as a LAST RESORT when all other tools fail or return errors. Good URLs: https://finance.yahoo.com/quote/AAPL, https://finance.yahoo.com/quote/AAPL/financials, https://stockanalysis.com/stocks/AAPL/financials/",
          inputSchema: z.object({
            url: z.string().describe("Full URL to visit, e.g. https://finance.yahoo.com/quote/AAPL/financials"),
          }),
          execute: async ({ url }) => {
            return this.browserFetch(url);
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
