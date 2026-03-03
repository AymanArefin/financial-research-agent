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
    // @ts-ignore
    if (this.env.POLYGON_API_KEY) {
      try {
        // @ts-ignore
        await this.addMcpServer("polygon", POLYGON_MCP_URL, {
          transport: {
            type: "sse",
            headers: {
              // @ts-ignore
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
        const toolCalls = (step as any).toolCalls ?? [];
        const toolResults = (step as any).toolResults ?? [];

        if (toolCalls.length === 0) return;

        const newResults: ToolResult[] = toolCalls.map((call: any, i: number) => ({
          tool: call.toolName,
          input: call.input as Record<string, unknown>,
          result: toolResults[i]?.output ?? null,
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
