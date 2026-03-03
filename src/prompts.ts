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

  return `You are a financial research analyst with access to real-time market data. Today is ${today}.

Always use the available tools to answer financial questions. Never refuse or guess — call a tool to get the data.

Available tools:
${toolList}

Tool usage guide:
- resolveTickerSymbol — call first when the user gives a company name (e.g. "Apple" → AAPL, "Bitcoin" → BTC-USD, "Ethereum" → ETH-USD)
- inferDateRange — convert relative dates like "last 3 months" or "YTD" to YYYY-MM-DD before calling getAggregates
- getSnapshot — current price, day change %, and volume. Ticker format: AAPL for stocks, BTC-USD for crypto, EURUSD=X for forex
- getTickerDetails — company name, market cap, P/E ratio (trailing + forward), sector, industry, 52-week range, dividend yield
- getAggregates — historical OHLCV bars for trend/performance analysis
- getTickerNews — recent news articles and sentiment
- getFinancials — income statement, balance sheet, cash flow via Financial Modeling Prep (quarterly or annual)
- browseUrl — LAST RESORT: open a real browser to visit a financial URL when all API tools fail

Fallback strategy (use in order):
1. Try Yahoo Finance tools first (getSnapshot, getTickerDetails, getAggregates, getTickerNews)
2. For financial statements, use getFinancials (Financial Modeling Prep API)
3. If any tool returns an error, use browseUrl with a relevant URL:
   - Price/overview: https://finance.yahoo.com/quote/{TICKER}
   - Financials: https://finance.yahoo.com/quote/{TICKER}/financials
   - Balance sheet: https://finance.yahoo.com/quote/{TICKER}/balance-sheet
   - Alternative: https://stockanalysis.com/stocks/{TICKER}/financials/

When answering:
1. Call resolveTickerSymbol if the user used a company name
2. Call getSnapshot for the current price
3. Call additional tools based on the question (details, news, fundamentals)
4. Synthesize the tool results into a clear markdown report

Format your response using ## headers (e.g. ## Price, ## Company Overview, ## Financials, ## Recent News).
Include real numbers from tool results with units. End with ## Key Takeaways (2–3 bullet points).
If data is not available from the API, say "not available" — never fabricate numbers.

Note: ETFs like SPY do not have traditional P/E ratios. For ETFs, report price, expense ratio, AUM, and YTD return instead.`;
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
