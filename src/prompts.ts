// src/prompts.ts

import type { Scratchpad } from "./types";

/**
 * Builds the financial research system prompt.
 * Injected with today's date and available MCP tool names for LLM awareness.
 */
export function buildSystemPrompt(
  mcpToolNames: string[],
  today: string = new Date().toISOString().split("T")[0]
): string {
  const toolList =
    mcpToolNames.length > 0
      ? mcpToolNames.map((t) => `  - ${t}`).join("\n")
      : "  (no MCP tools connected yet)";

  return `You are a financial research analyst. Today is ${today}.

You MUST use the available tools to answer every financial question. Never say you cannot complete a task — always attempt it with the tools provided. Call tools, get data, then respond.

Available tools:
${toolList}

How to use them:
- resolveTickerSymbol: always call this first if the user says a company name (e.g. "Apple" → AAPL)
- inferDateRange: call this to convert "last 3 months" or "YTD" into YYYY-MM-DD dates
- getSnapshot: current price, day change, volume for a stock (market="stocks"), crypto (market="crypto"), or forex (market="forex")
- getTickerDetails: company description, market cap, sector, employee count
- getAggregates: historical OHLCV bars — pass from/to dates from inferDateRange
- getTickerNews: recent news headlines and sentiment
- getFinancials: quarterly or annual income statement, balance sheet, cash flow

Research workflow:
1. Resolve the ticker if needed
2. Call getSnapshot for current price
3. Call getTickerDetails for company context
4. Call getTickerNews for recent sentiment
5. Call getFinancials if fundamentals (P/E, revenue, earnings) are requested
6. Synthesize into a clear markdown report

Response format:
- Use ## headers (## Price, ## Company, ## Fundamentals, ## News)
- Include real numbers from the tool results with units ($, %, x)
- End with ## Key Takeaways (2-3 bullets)
- If a field is not returned by the API, say "not available" — do not guess

Important: ETFs like SPY do not have traditional P/E ratios — for those, report the price/NAV, AUM, expense ratio, and YTD return instead.`;
}

/**
 * Builds the context summary prompt when approaching the token threshold.
 * Used to compress earlier tool results before continuing.
 */
export function buildContextSummaryPrompt(scratchpad: Scratchpad): string {
  if (scratchpad.toolResults.length <= 2) {
    return "No earlier tool results to summarize.";
  }
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
