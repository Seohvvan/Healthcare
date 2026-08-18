# AGENTS.md

<!-- Team-shared, tool-neutral rules. Reviewed in PRs; the single source of truth
     for policy every AI tool must follow. CLAUDE.md imports this via `@AGENTS.md`.
     Keep Claude-Code-specific orchestration out of here — that lives in CLAUDE.md. -->

## Shared project rules (all AI tools)

- All code, comments, and filenames are in English.
- Follow existing conventions; don't introduce new patterns unprompted.
- Run the closest build/lint/test before reporting a change complete.
- Never commit secrets or API keys.
- Record recurring mistakes as one generalizable rule in `docs/lessons-learned.md`.

## Build / test / lint commands

- test: `uv run pytest`
- lint: `uv run ruff check .`
- build: `uv build`
