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
