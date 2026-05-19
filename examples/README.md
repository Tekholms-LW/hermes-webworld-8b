# Prompt Templates for Hermes-WebWorld-8B

This folder contains ready-to-use prompt templates for different research and planning scenarios.

These prompts are optimized to work with the `simulate_web_action` tool and the Hermes-tuned WebWorld model.

## Available Templates

| File | Subject | Best For | Recommended Steps |
|------|---------|----------|-------------------|
| `01_ai_research.md` | AI / LLM Research | Papers, techniques, benchmarks | 12–18 |
| `02_defi_yield_research.md` | DeFi & Yield Hunting | Yield opportunities, risk analysis | 15–22 |
| `03_token_and_protocol_research.md` | Token / Protocol DD | New launches, due diligence | 18–25 |
| `04_github_and_codebase_monitoring.md` | GitHub Monitoring | Repo activity, PRs, issues | 10–15 |
| `05_market_and_news_analysis.md` | Market & Narrative | Sentiment, catalysts, outlook | 14–20 |
| `06_long_horizon_agent_planning.md` | Complex Multi-step Tasks | Long research + synthesis | 20–30+ |
| `07_multi_source_fact_check.md` | Fact Checking | Verifying claims across sources | 12–18 |

## How to Use

Copy any template and replace the placeholders in `[BRACKETS]` with your specific topic.

Example:
```
Use the simulate_web_action tool to fully plan and simulate the following research task...
```

The more specific and structured your task description is, the better the simulated trajectory will be.

## Tips for Best Results

- Be explicit about the final output format you want (table, report, ranked list, etc.)
- Mention time windows when relevant ("last 60 days", "last 30 days")
- Ask for risk factors or balanced views when doing investment-style research
- For very long tasks, break them into phases in the prompt

These templates were created based on real usage patterns with the Hermes Agent.