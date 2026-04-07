---
applyTo: "trinity/trainer/verl/**/*.py,docs/agents/**/*.md"
description: "Use veRL migration guardrails and docs navigation when editing Trinity veRL upgrade related files"
---

# veRL Upgrade Instructions

When the task is related to veRL upgrade/migration in Trinity:

1. Read `docs/agents/verl_upgrade/verl_upgrade_checklist.md` first.
2. Use current version and target version to generate or select a version-specific plan in `docs/agents/verl_upgrade/` following `verl_*_migration_plan.md` naming.
3. During implementation/review execution, focus only on detailed content for the target upgrade version.
4. Preserve Trinity boundaries:
   - Do not restore reward/rollout/validation main loops into Trinity trainer path by default.
   - Avoid whole-file overwrite from upstream snapshots.
5. Prefer three-way merge reasoning:
   - Trinity current vs old upstream baseline
   - old upstream vs new upstream
   - then current Trinity vs new upstream
6. If a subclass override is identical to upstream parent behavior, prefer removing the override.
