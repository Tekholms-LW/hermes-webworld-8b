# Prompt Template: Multi-Source Fact Checking

## Goal
Verify a claim or piece of information by cross-referencing multiple sources.

## Prompt
```
Use the simulate_web_action tool to fully plan and simulate the following fact-checking task. Show the complete step-by-step simulated trajectory with the predicted next state after each action.

Task: Verify the claim: "[INSERT CLAIM, e.g. 'Qwen3-235B achieves 2x faster inference than Llama-3.1-405B on the same hardware']". Check at least 4 different sources (papers, benchmarks, forum discussions, official announcements). Simulate the full verification process and produce a final verdict with confidence level and key supporting evidence.
```

## Recommended Settings
- Simulate 12–18 steps
- Ask for verdict + evidence summary
- Excellent for reducing hallucinations in research