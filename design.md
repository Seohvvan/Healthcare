# Healthcare Agentic AI Challenge 2026 — 시스템 설계안

> Interactive Clinical Trial Recommendation 멀티 에이전트 시스템.
> 2026-08-18 확정. 구현 기준 문서 (구현 코드/주석/파일명은 영어, 문서는 한국어).

## 0. 핵심 전략

- 제공된 `synthetic-patients.json`은 **TREC Clinical Trials Track 토픽 포맷**과 동일
  (`topics` / `num` / `title` 자유 텍스트 환자 서술).
- 따라서 **TREC 2021/2022 qrels(의사 판정 정답: eligible / excluded / not relevant)를
  정량 검증의 축**으로 사용한다. TREC 2021 = 개발/튜닝, TREC 2022 = 홀드아웃.
- 아키텍처는 NIH **TrialGPT**(Nature Communications 2024)의 검증된 3단계
  (Retrieval → Criterion 매칭 → Ranking)를 골격으로 하고, 공모전 차별화 요소인
  **UNKNOWN 판정 기반 확인 질문 생성 → 재평가 인터랙티브 루프**를 추가한다.

## 1. 데이터

| 데이터 | 용도 | 출처/라이선스 |
|---|---|---|
| ClinicalTrials.gov API v2 (`/api/v2/studies`) | 임상시험 코퍼스 (로컬 스냅샷) | 공개, API 키 불필요, 출처 표기 |
| TREC CT 2021/2022 (topics + qrels, ~37.5만 시험 코퍼스) | 정량 검증 정답 | 무료 등록, 연구 목적 |
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
- 질문 후 재평가는 전체 재실행이 아니라 해당 criterion만 갱신.
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

## 5. 평가지표

1. **Criterion 매칭**: 3분류 accuracy / macro-F1 / Cohen's κ — 수동 라벨 200~500쌍
   (2인 교차 라벨, 일치도 보고). 목표선: TrialGPT 87.3%.
2. **Trial 판정·랭킹**: TREC qrels 대비 3분류 정확도; **NDCG@10**(eligible gain 2,
   excluded gain 1), P@10, MRR; Retrieval 단계는 Recall@50.
3. **질문 생성**: **Masked-field recovery** — 완전 정보 가상 환자에서 필드 마스킹 →
   (a) 질문 적중률, (b) 재평가 후 정확도 상승·UNKNOWN 감소율. 보조: LLM-as-judge +
   소수 사람 검증.
4. **효율·ablation**: 토큰 비용/지연; 멀티에이전트 vs 단일 프롬프트, 도구 유무,
   하이브리드 vs BM25 단독.

## 6. 검증

- TREC 2021로 튜닝 → **TREC 2022 홀드아웃** 최종 1~2회 평가 (오버피팅 방지).
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
