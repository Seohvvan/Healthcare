# Healthcare Agentic AI Challenge 2026 — 시스템 설계안

> Interactive Clinical Trial Recommendation 멀티 에이전트 시스템.
> 2026-08-18 확정. 구현 기준 문서 (구현 코드/주석/파일명은 영어, 문서는 한국어).

## 0. 핵심 전략

- 제공된 `synthetic-patients.json`은 TREC Clinical Trials Track과 **과제 구조가
  동형**(자유 텍스트 환자 서술 → 임상시험 매칭)이고 `num`/`title`은 TREC 계열 토픽
  관례를 따른 것으로 보인다. 단 **포맷·문체가 동일하지는 않다** — TREC CT는 XML에
  5~10문장 입원기록형, 샘플은 1~2문장 교과서형 vignette.
- 검증 축은 3분리한다 (역할을 섞지 않는다):
  1. **TREC 2021/2022 qrels** = 매칭·랭킹 **엔진의 외부 타당도** 벤치마크.
     2021 = 개발/튜닝, 2022 = 홀드아웃. ⚠️ qrels는 2021-04 코퍼스 스냅샷 기준 —
     최신 API 크롤로 평가하면 **커버리지 결손 + 기준 텍스트 드리프트**로 점수가
     무의미하다. 기본은 **judged-subset 재랭킹**(토픽별 판정된 시험만 후보로 사용),
     코퍼스는 TrialGPT 공개 저장소의 전처리 배포본(2021 텍스트) 우선, 불가 시
     judged NCT ID를 API로 수집하되 드리프트 캐비앳을 결과에 명시.
  2. **대회 제공 10건** = 정답 없음. 예상 진단·적합 시험 수동 매핑으로 스모크/회귀.
  3. **masked-field 가상 환자** = 질문-재평가 루프(인터랙션) 성능의 정량 평가.
- 주최 측 채점 방식은 **미공개**이므로 특정 방식을 가정하지 않는다 — 정확성(3분류)·
  랭킹(nDCG 등)·인터랙션(질문 적중률) 지표군을 전부 리포트해 어떤 채점이든 대응.
- 아키텍처는 NIH **TrialGPT**(Nature Communications 2024)의 검증된 3단계
  (Retrieval → Criterion 매칭 → Ranking)를 골격으로 하고, **UNKNOWN 판정 기반
  확인 질문 생성 → 재평가 인터랙티브 루프**를 추가한다. 포지셔닝: "최초"가 아니라
  **기존 접근의 한계로 지목된 지점(전문가의 질문·기준 정제 의존)에 대해, 기준을
  수정하지 않고 환자 측 정보 격차를 질문으로 메우는 답** (인접 연구: PRISM,
  Criteria2Query 3.0 — 발표 인용 전 원문 확인 필요).

## 1. 데이터

| 데이터 | 용도 | 출처/라이선스 |
|---|---|---|
| ClinicalTrials.gov API v2 (`/api/v2/studies`) | 임상시험 코퍼스 (로컬 스냅샷) | 공개, API 키 불필요, 출처 표기 |
| TREC CT 2021/2022 (topics + qrels; 코퍼스는 2021-04 스냅샷) | 엔진 외부 타당도 (judged-subset 재랭킹) | TrialGPT 배포본 우선 / trec-cds.org 등록 |
| 가상 환자 (LLM 증강 생성) | 개발·질문 평가 | 자체 생성 (정답 시트 포함) |

- API v2: JSON, `pageSize` 최대 1000, `pageToken` 커서 페이지네이션.
  핵심 필드: `protocolSection.{eligibilityModule, conditionsModule, descriptionModule,
  identificationModule, statusModule, designModule}`.
- 가상 환자 3종: 정보 충분 / **핵심 정보 누락**(질문 평가용) / 제외 기준 함정 케이스.

## 2. 전처리

1. **Eligibility criteria 구조화**: inclusion/exclusion 블록 분리(규칙 기반) → 개별
   criterion 분해 → LLM 구조화(나이·성별·진단·검사 역치·투약 등 + 원문 보존).
   **시험별 1회 계산 후 캐시** (환자 간 재사용 = 비용 절감 핵심).
2. **하이브리드 검색 인덱스**: BM25 (제목+conditions+criteria) + dense 임베딩
   (MedCPT 권장, 로컬 무료; 초기엔 BM25 단독으로 시작, dense는 optional extra) →
   Reciprocal Rank Fusion.
3. **환자 프로파일 정규화**: 자유 텍스트 → 구조화 스키마, 미기재 필드는 `unknown`
   표기 (질문 생성의 입력).

## 3. 아키텍처: 깔때기 + 인터랙티브 루프

```
환자 서술
 → [1] Patient Profiler Agent   구조화 프로파일 + unknown 필드 + 검색 키워드
 → [2] Retrieval Agent          도구: BM25/임베딩/CT.gov API — 수만 → 후보 30~50건
 → [3] Criteria Parser Agent    criterion 단위 구조화 (캐시)
 → [4] Matching Agent           criterion별 MET / NOT MET / UNKNOWN + 근거 인용
 |                              도구: 나이/BMI/eGFR 계산기, 용어 정규화
 → [5] Gap Detector + Question Agent
 |        판정을 좌우하는 UNKNOWN → 확인 질문 (제외 기준 우선)
 |        ←→ 사용자 또는 Patient Simulator Agent 응답 → [4] 해당 criterion만 재평가
 → [6] Ranking Agent            점수 집계 → Top-k 추천 (집계 규칙은 결정적 코드)
 → [7] Reporter Agent           판정 + 근거 + 추천 + 의료 면책 고지
```

설계 원칙:
- 판정 스키마는 criterion 단위 3분류 **MET / NOT MET / UNKNOWN** (TrialGPT 방식).
- **집계·하드룰은 LLM이 아닌 코드로**: 제외 기준 확정 위반 1건 = excluded,
  선정 충족률·unknown 비율로 점수화.
- **최종 정렬은 라벨 등급 우선**: ELIGIBLE > 미확정 > EXCLUDED > NOT_RELEVANT,
  같은 등급 안에서 점수순. "추천 불가"는 순위가 아니라 리포트의 판정 라벨로
  전달하며, 이 순서는 TREC gain(2 > 1 > 0)과도 일관된다.
- 질문 대상은 **판정 미확정(trial_label 미정) 시험의 UNKNOWN만** — 이미
  excluded/not relevant로 확정된 시험에 질문을 낭비하지 않는다.
- 질문 후 재평가는 전체 재실행이 아니라 해당 criterion만 갱신 (기존 판정은
  병합 유지 — 확정된 MET이 라운드 간에 뒤집히지 않는다).
- **Patient Simulator Agent**로 질문-재평가 루프를 자동 시연·평가.

## 4. 모델

| 역할 | 모델 | 근거 |
|---|---|---|
| 파싱·추출·시뮬레이터 | Claude Haiku 4.5 ($1/$5 /MTok) | 구조화 추출은 소형으로 충분 |
| criterion 매칭·질문 생성 | Claude Sonnet 5 ($3/$15; 인트로 $2/$10 ~8/31) | 정확도 직결 단계 |
| 최종 리포트(선택) | Claude Opus 5 ($5/$25) | 소량만 |

- 비용 레버: **Batch API(50%↓, 오프라인 평가), Prompt caching(시험 기준 재사용 ~0.1×),
  Structured Outputs(JSON 스키마 강제 → 파싱 실패 제거)**.
- 임베딩: MedCPT 또는 BGE-M3 (로컬). BM25: rank_bm25.
- 오케스트레이션: Anthropic SDK tool-use 루프 직접 구현 (의존성 최소).

**프로바이더 이중화 (실험 경로).** 위 Claude 티어링이 **기준 플랜**이고, 비교 실험용으로
Google Gemini 경로를 추가한다 (`--provider gemini`).

| 역할 | Claude (기준) | Gemini (실험) |
|---|---|---|
| 파싱·추출·시뮬레이터 | Haiku 4.5 | Gemini 2.5 Flash |
| criterion 매칭·질문 생성 | Sonnet 5 | Gemini 2.5 Pro |
| 최종 리포트 | Sonnet 5 / Opus 5 | Gemini 2.5 Flash |

- 두 프로바이더는 `structured`/`text` **동일 인터페이스**(`trialmatch/llm.py`)로 감싸므로
  에이전트·파이프라인 코드는 프로바이더를 알지 못한다. 모델 ID는 `config.py`의
  `ModelConfig` / `GEMINI_MODELS` 한 곳에서만 바뀐다.
- 캐싱 차이: Claude는 명시적 prompt caching 브레이크포인트(`cache_system=True`),
  Gemini 2.5는 암묵적 캐싱이라 같은 인자를 받되 무시한다.
- 자격증명은 각 SDK가 환경변수에서 읽는다 (`ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`).
  §5.2 벤치를 두 프로바이더로 각각 돌려 정확도·비용을 비교하는 것이 실험 목적.

## 5. 평가지표

1. **Criterion 매칭**: 3분류 accuracy / macro-F1 / Cohen's κ — 수동 라벨 200~500쌍
   (2인 교차 라벨, 일치도 보고). 목표선: TrialGPT 87.3%.
2. **Trial 판정·랭킹**: TREC qrels 대비 3분류 정확도; **NDCG@10**(eligible gain 2,
   excluded gain 1), P@10, MRR; Retrieval 단계는 Recall@50.
   기본 실험은 **judged-subset 재랭킹**(토픽별 판정 시험만 후보로; LLM 매칭은
   토픽·후보 수를 예산에 맞게 캡). 전체 코퍼스 retrieval 평가는 여력 있을 때 확장.
   벤치마크 경로(2021 데이터)와 데모/대회 경로(최신 API v2 스냅샷)는 분리 운영.
3. **질문 생성**: **Masked-field recovery** — 완전 정보 가상 환자에서 필드 마스킹 →
   (a) 질문 적중률, (b) 재평가 후 정확도 상승·UNKNOWN 감소율. 보조: LLM-as-judge +
   소수 사람 검증.
4. **효율·ablation**: 토큰 비용/지연; 멀티에이전트 vs 단일 프롬프트, 도구 유무,
   하이브리드 vs BM25 단독.

## 6. 검증

- TREC 2021로 튜닝 → **TREC 2022 홀드아웃** 최종 1~2회 평가 (오버피팅 방지).
  TREC 결과는 "엔진의 외부 타당도"로만 주장한다 — 대회 10건의 성능 증명이 아니다.
- 제공 샘플 10건: 예상 진단·적합 시험 수동 매핑 → 상시 회귀 스모크 테스트.
- 재현성: 프롬프트·모델 버전 고정, 전체 실행 로그 저장, README 재현 절차.
- 오류 유형화(수치 해석·시간 조건·약어·근거 없는 추론)를 발표에 포함.

## 7. 일정 (8/18 → 8/30)

- 1–3일: 데이터 확보·인덱스·API 연결 / 4–7일: 깔때기 E2E + TREC 2021 1차 측정
- 8–10일: 질문 루프 + 시뮬레이터 + masked-field 실험 / 11–12일: 홀드아웃·ablation
- 13–14일: 발표 자료·데모·README·라이선스·면책 고지·제출

우선순위: 깔때기 몸통(매칭 30% 직결) > 질문 루프(차별화) > ablation.

## 8. 산출물 요건 대응

- 에이전트 구성도(위 다이어그램), 재현 가능한 코드 + README + 의존성 명세,
  데이터 출처·라이선스 명시, 출력물에 의료 면책 고지 포함.
