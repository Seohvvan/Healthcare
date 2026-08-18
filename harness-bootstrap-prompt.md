# Claude Code 하네스 부트스트랩 프롬프트

이 파일 전체를 Claude Code 세션에 붙여넣으면, 아래 명세대로 팀용 하네스 파일들을
정확히 생성한다. (설명은 한국어, 생성되는 파일 내용은 영어로 유지한다.)

---

## Claude Code에게 지시

너는 이 저장소에 Claude Code 오케스트레이션 하네스를 구성한다. 아래 7개 파일을 **명세된
경로에 명세된 내용 그대로** 생성하라. 단계별로 진행하고, 각 단계 뒤에 검증 게이트를 통과한
뒤 다음으로 넘어가라. 파일 내용에 임의로 문장을 추가·삭제하지 말 것.

- 이미 존재하는 파일이 있으면 덮어쓰기 전에 먼저 알리고 확인을 받는다.
- 모든 코드/주석/파일명/지침 내용은 영어로 유지한다.
- 필요한 상위 디렉터리(`.claude/agents/`, `.claude/commands/`, `docs/`)는 함께 생성한다.

### STEP 1 — 팀 공유 규칙: `AGENTS.md`

경로: `AGENTS.md`

```markdown
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

<!-- Fill in for this repo, e.g.: -->
<!-- - test:  `uv run pytest` -->
<!-- - lint:  `ruff check .` -->
<!-- - build: `uv build` -->
```

**검증 게이트 1:** `AGENTS.md`가 생성됐는지 확인하고 파일 내용을 출력하라.

### STEP 2 — 오케스트레이션 정책: `CLAUDE.md`

경로: `CLAUDE.md`

```markdown
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
```

**검증 게이트 2:** `CLAUDE.md`가 생성됐는지 확인하고, `@AGENTS.md`와
`@docs/lessons-learned.md` import 라인이 파일 상단에 그대로 들어갔는지 확인하라.

### STEP 3 — 서브에이전트: `.claude/agents/researcher.md`

경로: `.claude/agents/researcher.md`

```markdown
---
name: researcher
description: Read-only codebase researcher. Use to explore code, trace call paths, and gather context in parallel without touching files. Spawn several at once for independent areas.
tools: Read, Grep, Glob, Bash
model: opus
---

You are a focused codebase researcher. You NEVER modify files.

When invoked:
1. Scope your investigation to exactly the area named in the delegation prompt.
2. Trace the relevant files, symbols, and call paths (use `git`, `rg`, etc.).
3. Return a concise, structured report — findings only, no raw file dumps.

Report format:
- Summary (2–4 sentences)
- Key files & symbols (path — role)
- Existing patterns/conventions relevant to the task
- Risks, unknowns, or questions for the orchestrator

Keep the report short enough to drop straight into the main context. If you can't
answer within your assigned scope, say so — do not expand scope on your own.
```

### STEP 4 — 서브에이전트: `.claude/agents/implementer.md`

경로: `.claude/agents/implementer.md`

```markdown
---
name: implementer
description: Implements one independent, well-specified slice of a larger task. Use for parallelizable coding work the orchestrator has already decomposed. Not for design decisions.
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
---

You implement exactly ONE slice, defined entirely by your delegation prompt. You
start with a fresh context: assume nothing beyond the prompt, CLAUDE.md, and the
imported lessons.

Rules:
- Implement only what the contract specifies. Do not touch files outside your slice.
- Follow existing conventions and the imported do-not-repeat lessons. Do not
  introduce new patterns unless told to.
- If the spec is ambiguous or you hit a blocker, STOP and report back rather than
  guessing or widening scope.
- Run the closest build/lint/test for the files you changed before reporting.

Return a concise summary (NOT full diffs):
- What you implemented (2–4 bullets)
- Files changed (path — one-line reason)
- Build/test result
- Anything the orchestrator or reviewer must know (assumptions, TODOs, escalations)
```

### STEP 5 — 서브에이전트: `.claude/agents/reviewer.md`

경로: `.claude/agents/reviewer.md`

```markdown
---
name: reviewer
description: Read-only code reviewer. Use after implementer subagents finish to verify correctness and quality against the diff. Reports defects and proposes do-not-repeat lessons; never edits code.
tools: Read, Grep, Glob, Bash
model: opus
# For a genuinely trivial diff, dispatch with a per-call `model: sonnet` override instead of repointing this file.
---

You are a senior code reviewer. You NEVER modify code.

When invoked:
1. Run `git diff` (or review the paths named in the prompt) to see what changed.
2. Review only those changes against the checklist below.

Checklist:
- Correctness: meets the specified contract? edge cases handled?
- No exposed secrets/keys; inputs validated
- Error handling and failure paths
- Naming, readability, no needless duplication
- Consistency with existing conventions and the imported lessons
- Adequate tests for the change

Output, ordered by severity:
- CRITICAL (must fix) / WARNING (should fix) / SUGGESTION (optional)
  — each as `file:line — problem — concrete fix`.
- ESCALATE: anything ambiguous or architectural that needs a Fable-level decision
  from the orchestrator. Do not guess.
- LESSONS: 0–3 one-line, generalizable rules the orchestrator should append to
  `docs/lessons-learned.md` so this class of defect isn't repeated.

If the diff is clean, say so plainly. Keep the whole report compact.
```

**검증 게이트 3:** `.claude/agents/` 아래 3개 파일이 존재하는지 확인하고, 각 파일의
`model` 필드가 `opus`인지(리뷰어 주석 포함) 출력하라.

### STEP 6 — 재발 방지 파일: `docs/lessons-learned.md`

경로: `docs/lessons-learned.md`

```markdown
# Lessons Learned (do-not-repeat)

<!-- Auto-loaded into every session and subagent via the @import in CLAUDE.md. -->
<!-- One generalizable rule per line. Keep it short; prune duplicates/stale items. -->
<!-- Format: - [area] rule (reason) -->

<!-- Examples — delete once you have real entries:
- [imports] Import from `@company/utils-v2`, never `@company/utils` (v1 is deprecated).
- [tests] Run `make test-integration`, not `pytest` directly (fixtures live in the Makefile).
- [async] Never call the sync client inside an async handler; use `aclient` (blocks the loop).
-->
```

### STEP 7 — 노션 작업 로그 커맨드: `.claude/commands/log.md`

경로: `.claude/commands/log.md`

**예외:** 이 파일만은 한국어로 생성한다 — 노션 기록 자체가 한국어로 작성되는 팀
문서이고, 이 파일이 그 문체까지 규정하기 때문이다. `<!-- 프로젝트에 맞게 수정 -->`
표시가 붙은 곳(DB URL, 담당자, 키워드, 제목의 성과 지표)은 새 프로젝트 세팅 시
실제 값으로 바꾼 뒤 사용한다.

```markdown
---
description: 푸시된 작업을 사용자 요청 단위로 Notion 작업 DB에 기록
---

## 기록 대상
Notion "작업" DB
<!-- 프로젝트에 맞게 수정: 팀 노션 작업 DB의 URL을 여기에 넣는다 -->
<Notion 작업 DB URL>
https://app.notion.com/p/833f0046a83582e1b4f981673678d139?v=8eff0046a8358203843008f2645956d7&source=copy_link

## 기록 단위와 시점 (중요)
- **사용자 요청 1건 = 노션 페이지 1건.** 커밋·태스크 단위로 쪼개지 않는다.
- **git push 기준으로 기록한다.** 커밋만 되고 푸시되지 않은 작업은 기록하지 않는다.
  /log 실행 전 `git log origin/<branch>..<branch>`로 미푸시 커밋을 확인하고,
  미푸시 상태면 먼저 푸시 여부를 사용자에게 확인한다.
- 같은 요청의 후속 푸시는 **기존 페이지를 갱신**한다 (해당 섹션에 추가).
  새 페이지는 새로운 사용자 요청에만 만든다.
- 요청 안에 세부 작업이 많으면 본문에 섹션으로 정리하고, 필요할 때만
  하위 페이지로 내린다. DB 행은 요청당 1개를 유지한다.

## 스키마 (옵션값은 Notion 등록 문자열 — 그대로 복사, 번역·변형 금지)
<!-- 프로젝트에 맞게 수정: 팀 DB의 실제 속성·옵션 문자열로 교체한다 -->
| 속성 | 타입 | 허용값 |
|---|---|---|
| 작업 | title | 자유 텍스트 |
| 진행 상태 | status | 시작전 / 진행중 / 완료 |
| 담당자 | select | <팀원 이름들> |
| 키워드 | multi_select | 문서 / 평가지표 / 데이터 / 모델 / 디자인 |
| 시작일 | date | YYYY-MM-DD |

## 제목 규칙
- 형식: **"{핵심 개선 방법} ({성과 변화})"**. 성과 변화가 없으면 결과 한 단어(기각/구축/종결).
  <!-- 프로젝트에 맞게 수정: 성과 지표(점수, 지연시간, 정확도 등)를 프로젝트 것으로 -->
  - (O) "4·5차 제출: 파라미터 튜닝과 앙상블 (778.7 → 812.9점)"
  - (O) "트랙맨 데이터 활용: 투수 매칭은 성공, 피처는 기각"
  - (X) "모델 개선" (내용 없음), (X) "P2-N 하이퍼파라미터 스윕" (내부 코드명)
- 내부 태스크 코드·커밋 해시·브랜치명은 제목에 쓰지 않는다.

## 본문 템플릿 (아래 5개 섹션을 이 순서로, 섹션 사이 줄바꿈)

    ## 한 줄 요약
    {무엇을 해서 어떤 결과가 나왔는지 1~2문장}

    ## 개선 방법
    - {시도한 방법을 짧은 불릿으로, 하나당 한 줄}

    ## 성과
    - {핵심 지표: 이전 → 이후 (±Δ)}
    - (측정이 없으면) 내부 검증: {채택/기각과 근거 요약}

    ## 인사이트
    - {이 작업에서 배운 것 — 다음 작업에 영향을 주는 것만}

    ## 다음 단계
    - {있으면 한두 줄, 없으면 섹션 생략}

## 작성 규칙
- 진행 상태: 검증(테스트·측정)까지 끝났으면 `완료`,
  코드는 있으나 검증 전이면 `진행중`. `시작전`은 /log로 기록하지 않는다.
- 담당자: 지정 없으면 요청한 사람
- 키워드: 요청에 해당하는 것 모두 선택. 맞는 게 없으면
  새로 만들지 말고 사용자에게 물을 것.
- 시작일: 요청 착수일, 모르면 오늘

## 문체 규칙 (정형화의 핵심)
- **커밋 해시·브랜치명·내부 태스크 코드·테스트 개수는 본문에 쓰지 않는다.**
  재현 정보는 git 이력(과 실험 원장이 있으면 그것)이 담당한다.
- 영어는 고유명사(라이브러리·지표·seed 등)만 허용. 설명은 전부 한국어로.
- **실제 데이터를 언급할 때는 괄호로 영문 원명을 병기한다** — 컬럼·피처·파일명이
  코드/데이터에서 실제로 찾을 수 있도록.
  - (O) "투수 미들률(asof_pitcher_middle_rate)", (X) "투수 미들률" (원명 없음)
- **피처·설정을 추가·삭제·변경한 작업은 n개 요약이 아니라 n개 전체를 나열한다.**
  개수만 적으면 나중에 추적할 수 없다 — 그룹별로 묶되 각 항목의 영문 원명을
  빠짐없이 기재한다. 목록이 길면 해당 불릿 아래 하위 불릿으로 정리한다.
- 통계 용어는 풀어서 쓴다: "8σ" → "노이즈 범위를 8배 벗어남".
- 섹션마다 빈 줄로 구분하고, 불릿 하나는 한 줄을 넘기지 않도록 요약한다.
- 결과가 나빠진(기각된) 시도도 반드시 기록한다 — 실패 기록이 같은 실험의 반복을 막는다.

## 금지
- 계획 단계에 그친 것을 기록하지 않는다 (실행·검증된 작업만)
- 속성 옵션을 새로 생성하지 않는다
- 터미널에 실제 출력되지 않은 수치를 쓰지 않는다
- 같은 요청을 여러 페이지로 쪼개지 않는다 (요청 1건 = 페이지 1건)

## 기록 후 검증 (필수)
- 페이지를 생성·수정한 뒤에는 **반드시 fetch로 다시 읽어 반영 여부를 확인**한다.
  API 성공 응답만 믿지 않는다 — 부분 반영·불일치가 조용히 발생할 수 있다.

---
위 규칙에 따라, 이번 대화에서 실제로 실행·검증되고 푸시된 작업을
사용자 요청 단위로 묶어 Notion MCP로 기록하라 (기존 요청이면 해당 페이지 갱신).
기록·갱신 후 해당 페이지 URL을 출력한다.
```

**검증 게이트 4:** `.claude/commands/log.md`가 생성됐는지 확인하고,
`<!-- 프로젝트에 맞게 수정 -->` 표시 항목(DB URL, 담당자, 키워드, 성과 지표)을
사용자에게 실제 값으로 채워달라고 요청하라. Notion MCP가 연결돼 있으면 DB를
fetch해 스키마 표의 속성·옵션 문자열을 실제 등록값과 대조하라.

### 최종 검증

1. 다음 파일이 모두 존재하는지 나열하라:
   `AGENTS.md`, `CLAUDE.md`, `.claude/agents/researcher.md`,
   `.claude/agents/implementer.md`, `.claude/agents/reviewer.md`,
   `docs/lessons-learned.md`, `.claude/commands/log.md`.
2. `/memory`를 실행해 `CLAUDE.md`가 `AGENTS.md`와 `docs/lessons-learned.md`를
   import하고 있는지 확인하라.
3. 세 서브에이전트가 등록됐는지(`researcher`, `implementer`, `reviewer`) 확인하라.
4. 요약을 3줄 이내로 보고하고, 실패/누락이 있으면 그 항목만 다시 처리하라.
