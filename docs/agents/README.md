# Agent Knowledge Hub

This directory stores agent-oriented documentation for upgrade workflows, runbooks, and operating constraints.

## Structure

- `verl_upgrade/`: veRL upgrade knowledge, including planning, checklist, and future postmortems.

## How To Use

1. Start from `verl_upgrade/verl_upgrade_checklist.md` before a version upgrade.
2. Use current version and target version to generate or select the corresponding plan in `verl_upgrade/` following `verl_*_migration_plan.md` naming.
3. During execution, review only the detailed content for the target upgrade version.
4. Add new version migration records in `verl_upgrade/` using versioned file names.

## Naming Convention

- Checklist: `verl_upgrade_checklist_<version_or_scope>.md`
- Plan: `verl_<from_version>_to_<to_version>_migration_plan.md`
- Postmortem: `verl_upgrade_postmortem_<date>.md`
