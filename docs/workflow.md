# Workflow

## Branch Roles

- `dev/agent`: active development branch for agent-assisted work, exploration, and intermediate artifacts
- `main`: production branch for stable, curated code only

## Promotion Rule

All exploratory work starts on `dev/agent`. Once a change is production-ready, promote only the necessary files or commits onto `main`.

Preferred promotion options:

1. Cherry-pick a clean commit from `dev/agent` onto `main`.
2. Merge `dev/agent` only if the branch contains no agent-only files or exploratory history you want to exclude.

## Agent-Only Material

Keep items like these off `main` unless they are intentionally part of the shipped project:

- scratch notes produced for agent iteration
- temporary research artifacts
- one-off debugging files
- prompt logs or agent-specific workflow files

## Practical Rule

If a file exists only to help an agent or an exploratory development loop, keep it on `dev/agent`. If it is part of the stable pipeline, documentation, or test suite, it can be promoted to `main` when ready.
