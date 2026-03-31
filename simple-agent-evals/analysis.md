# analysis.md

## Overview

In this lab, I evaluated a multi-tool agent using Braintrust on a dataset of 25 single-turn queries across five categories: search, weather, directions, multi_tool, and out_of_scope. The agent integrates three tools: DuckDuckGo search, Open-Meteo weather, and OSRM directions.

The evaluation combines custom heuristic scorers (tool selection, response completeness, latency, no error, and scope awareness) to assess different aspects of agent performance. Due to issues with the LLM-as-judge scorers, the final analysis focuses on the heuristic metrics.

---

## Overall Performance

Overall, the agent performs quite well across most metrics. According to the evaluation results:

* Latency: 1.0 average
* NoError: 1.0 average
* ToolSelection: 1.0 average
* ResponseCompleteness: 0.9762 average
* ScopeAwareness: 0.96 average 

This indicates that the agent is stable, consistently uses the correct tools, and produces complete responses in most cases. The system is also efficient, with all responses falling into the fastest latency bucket.

---

## Category-Level Analysis

### Directions and Weather

The agent performs perfectly on both directions and weather tasks. It consistently calls the correct tools and includes all expected information such as distance, duration, and temperature. All scores in these categories are 1.0 across metrics.

This suggests that tool routing is very reliable for clearly structured queries.

---

### Search

Search queries also achieve strong performance, with perfect completeness and no errors. The agent is able to generate sufficiently detailed answers using the search tool.

However, in some cases (e.g., factual questions like “What is the capital of Australia”), the agent answers correctly without calling a tool. This is acceptable behavior but highlights that tool usage is not always necessary for simple queries.

---

### Multi-Tool

Multi-tool queries show slightly lower performance, especially in ResponseCompleteness (average 0.9167) 

The main issue is that while the agent correctly calls multiple tools, it sometimes does not fully integrate all pieces of information into a cohesive answer. For example, in the Miami trip planning case, the response includes weather but lacks sufficient detail on activities, leading to a lower completeness score (0.5).

This suggests that the main bottleneck is not tool usage, but **how well the agent synthesizes outputs from multiple tools**.

---

### Out-of-Scope

Out-of-scope queries have a ScopeAwareness score of 0.75 

In most cases, the agent correctly declines requests such as booking flights or sending emails. However, one failure occurs when the agent attempts to answer a stock price query instead of recognizing it as out-of-scope.

This indicates that the boundary of what counts as “supported” is not always clearly enforced by the agent.

---

## Error Analysis

Although there were no runtime errors in the evaluation (NoError = 1.0), two main weaknesses were observed:

1. Incomplete responses in multi-tool scenarios
   The agent sometimes retrieves the correct information but fails to present all required components clearly.

2. Imperfect scope awareness
   The agent occasionally attempts to answer questions that should be declined.

These issues are relatively minor but highlight areas for improvement in reasoning and decision-making rather than tool execution.

---

## LLM-as-Judge Scorer Issue

The intended evaluation setup included LLM-based scorers (Factuality and ClosedQA) using Claude Sonnet. However, during execution, these scorers failed due to response format mismatches and async runtime issues (e.g., missing expected keys in outputs and event loop errors).

As a result, the final evaluation relies on heuristic scorers only. Despite this limitation, the heuristic metrics still provide a useful and consistent view of agent performance.

---

## Conclusion

Overall, the agent demonstrates strong performance in tool selection, reliability, and response speed. It handles single-tool queries almost perfectly and performs reasonably well on more complex multi-tool tasks.

The main areas for improvement are:

* Better aggregation of multi-tool outputs
* Clearer handling of out-of-scope queries

With relatively small improvements in response synthesis and decision boundaries, the agent could achieve near-perfect performance across all categories.
