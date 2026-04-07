---
name: verl-upgrade
description: "Use when handling veRL version upgrades in Trinity, including three-way merge strategy, boundary checks, and retained customization decisions"
---

# veRL Upgrade Skill

## Primary Sources

1. `docs/agents/verl_upgrade/verl_upgrade_checklist.md`
2. A version-specific migration plan in `docs/agents/verl_upgrade/` matching `verl_*_migration_plan.md`

## Workflow

1. Read `docs/agents/verl_upgrade/verl_upgrade_checklist.md` first.
2. Confirm current version, target version, upgrade scope, and target files.
3. Generate or select the corresponding version-specific migration plan (`verl_*_migration_plan.md`).
4. During execution, review only detailed content for the target upgrade version.
5. Run three-way comparison against `trinity/trainer/verl/build/<version>/` snapshots.
6. Preserve Trinity responsibility boundaries.
7. Keep required Trinity customizations and remove redundant upstream copies.
8. Validate config-to-implementation wiring and output-field contracts.
9. Export remote GPU regression checklist after local static checks.

## Hard Constraints

1. Do not do whole-file overwrite from upstream.
2. Do not reintroduce reward/rollout/validation trainer logic unless responsibilities changed.
3. Keep checkpoint monitor/synchronizer collaboration where required.
