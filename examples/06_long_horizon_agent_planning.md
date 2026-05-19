# Prompt Template: Long-Horizon Agent Planning

## Goal
Test the model's ability to simulate complex, multi-step agent workflows.

## Prompt
```
Use the simulate_web_action tool to fully plan and simulate the following complex multi-step task. Show the complete step-by-step simulated trajectory with the predicted next state after each action.

Task: [Describe a complex goal, e.g. "Build a comparison of the top 5 open-source coding models released in 2026, including benchmarks, context length, and fine-tuning difficulty"]. Break the goal into logical steps, simulate the full research and synthesis process, and at the end deliver a polished final deliverable (table, report, or structured summary).
```

## Recommended Settings
- Simulate 20–30+ steps
- Ask for final polished output
- Best for testing long-horizon reasoning

## Example Complex Tasks
- Compare multiple models across benchmarks
- Research + summarize a new protocol launch
- Plan and simulate a full competitive analysis
- Multi-source fact-checking on a claim