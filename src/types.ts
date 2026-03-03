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
