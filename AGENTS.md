# Multi-Agent Entry Guide

This repository supports multiple coding agents.

## Canonical Knowledge Location

- Agent documentation root: `docs/agents/`
- veRL upgrade docs: `docs/agents/verl_upgrade/`

## Agent-Specific Templates

- Copilot instructions: `.github/instructions/verl-upgrade.instructions.md`
- Claude skill: `.claude/skills/verl-upgrade/SKILL.md`
- Codex template: `.codex/AGENTS.md`

## Shared Rule

All agents should follow this order for veRL upgrades: read checklist first, generate/select a version-specific migration plan from current->target version, then review only target-version detailed content during execution.
