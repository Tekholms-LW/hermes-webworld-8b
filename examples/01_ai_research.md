# Prompt Template: AI / LLM Research

## Goal
Research recent papers or developments in a specific AI subfield and produce a structured summary or comparison.

## Prompt
```
Use the simulate_web_action tool to fully plan and simulate the following research task. Show the complete step-by-step simulated trajectory with the predicted next state after each action.

Task: Research the latest advances in [SUBJECT, e.g. "efficient inference for large language models"] published in the last 60 days. Focus on methods that improve speed, reduce memory usage, or maintain quality. Simulate a full research session and at the end produce a clean comparison table of the top 3 techniques, including key metrics (tokens/sec, VRAM, quality retention) and one-sentence summaries.
```

## Recommended Settings
- Simulate 12–18 steps
- Ask for a markdown table at the end
- Good for: LLM inference, training techniques, evaluation methods

## Example Subjects
- Efficient LLM inference
- Mixture-of-Experts optimization
- Long-context handling
- Agentic workflows and tool use