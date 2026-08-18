# CLAUDE.md — Orchestration & Harness Policy

<!-- Loaded into EVERY session and EVERY non-Explore/Plan subagent. Keep it lean. -->
<!-- Team-neutral rules live in AGENTS.md (imported below). Detailed role
     instructions live in .claude/agents/*.md, not here. -->

@AGENTS.md
@docs/lessons-learned.md

## Model routing (cost control)

- Run the **main session on Fable 5** (`/model fable`). The main session is the
  **orchestrator / supervisor**: it plans, decomposes work, reviews results,
  resolves escalations, and writes the final summary. Planning stays on Fable.
- **Worker subagents default to Opus 5** (`model: opus` in their frontmatter) —
  implementation, review, and research are real reasoning work. A subagent's
  `model` defaults to `inherit`, so leaving it unset makes the worker run on
  Fable and cost more. Always pin workers explicitly.
- **Truly trivial slices go to Sonnet 5**: mechanical renames, boilerplate,
  bulk search-and-list, log triage. Override per dispatch (pass
  `model: sonnet` when spawning) instead of repointing the agent files.
- Escalate to Fable (main session) only for genuinely hard reasoning: ambiguous
  architecture, subtle concurrency/security bugs, cross-cutting refactors.
- Quick session-wide override: launch with `CLAUDE_CODE_SUBAGENT_MODEL=opus`.

## Delegate vs. do inline

Delegate to subagents when work is parallelizable and independent, produces
high-volume output (test logs, search dumps) you won't reuse, or needs
tool/permission isolation. Do it inline for small, iterative, tightly-coupled
changes where delegation overhead isn't worth it.

## Standard workflow

1. **Plan (Fable, main).** Restate the goal and split the task into *independent*
   slices. Do NOT parallelize slices that depend on each other's in-progress
   output — chain those sequentially.
2. **Fan out (Opus).** Spawn one subagent per independent slice, in parallel.
   Subagents start fresh and see only your prompt plus CLAUDE.md, so each
   delegation prompt MUST be self-contained: exact file paths, the interface to
   implement, constraints, and "return a concise summary, not raw diffs/logs".
3. **Review (`reviewer` subagent).** After workers finish, delegate to `reviewer`
   to code-review the diffs. It reports defects and proposes do-not-repeat
   lessons; it never edits code.
4. **Verify (Fable, main).** Read the findings, resolve ESCALATE items, run the
   build/tests, decide done / needs-fix, and append reviewer-approved lessons to
   `docs/lessons-learned.md`. Loop back to step 2 for fixes.
5. **Document.** After pushing, run `/log` to record the work into the team's
   Notion 작업 DB — one page per user request, updated on subsequent pushes
   (rules live in `.claude/commands/log.md`).

## Subagent roster

- `researcher` (Opus, read-only) — parallel codebase exploration; findings only.
- `implementer` (Opus) — implements one independent slice; returns a summary.
- `reviewer` (Opus, read-only) — code-reviews diffs, proposes lessons.

Spawn multiple `researcher` / `implementer` instances concurrently for
independent slices.

## Keep the main context clean

Subagent results return into the main context, so require concise summaries
(files changed, key decisions, open TODOs — not raw diffs or logs). This is the
single biggest lever for keeping a long session both cheap and sharp.
