# Codex Repository Guide

## Canonical Documentation Roots

- Agent documentation root: `docs/agents/`

## Agent Entry Files

- Workspace-level guide: `AGENTS.md`
- Codex-specific guide: `.codex/AGENTS.md`

## Repository Convention

Treat `docs/agents/` as the single source of truth for agent-facing process and navigation documents.

## Local Python Environment

- Always use the repository virtual environment for Python commands: `.venv/bin/python`.
- Run Python tools through that interpreter, for example `.venv/bin/python -m pytest ...`, instead of relying on globally installed commands.
