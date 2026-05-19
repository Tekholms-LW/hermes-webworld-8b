# Prompt Template: GitHub & Codebase Monitoring

## Goal
Monitor a repository for new developments, issues, or PRs and summarize changes.

## Prompt
```
Use the simulate_web_action tool to fully plan and simulate the following GitHub research task. Show the complete step-by-step simulated trajectory with the predicted next state after each action.

Task: Monitor the repository [REPO, e.g. "unslothai/unsloth"] for recent activity. Check: new issues in the last 7 days, recently merged PRs, open discussions about [TOPIC], and any major roadmap updates. Simulate the full monitoring session and produce a clean summary of the most important updates with links and one-sentence descriptions.
```

## Recommended Settings
- Simulate 10–15 steps
- Ask for bullet-point summary
- Good for: Staying updated on open-source projects, tracking dependencies

## Example Repos
- unslothai/unsloth
- huggingface/transformers
- ethereum/go-ethereum
- Any project you contribute to or depend on