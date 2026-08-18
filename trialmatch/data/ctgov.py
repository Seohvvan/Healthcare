"""ClinicalTrials.gov API v2 client and snapshot downloader.

The public API needs no key. We only read `/api/v2/studies` and store a local
JSONL snapshot so the rest of the pipeline runs fully offline and reproducibly.
Source: https://clinicaltrials.gov/data-api/api (public domain data, cite source).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Self

import httpx

from trialmatch.data.store import write_jsonl
from trialmatch.schemas import TrialRecord

API_URL = "https://clinicaltrials.gov/api/v2/studies"

DEFAULT_STATUSES: tuple[str, ...] = (
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "COMPLETED",
)

MAX_PAGE_SIZE = 1000
_RETRY_ATTEMPTS = 2
_RETRY_BASE_SLEEP = 1.0


class CTGovClient:
    """Thin paginating client over the ClinicalTrials.gov v2 studies endpoint.

    Pass `transport` (e.g. `httpx.MockTransport`) or a pre-built `client` to
    exercise this offline in tests.
    """

    def __init__(
        self,
        base_url: str = API_URL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = client or httpx.Client(timeout=timeout, transport=transport)
        self._owns_client = client is None

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- requests --------------------------------------------------------

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        """GET with a small retry on timeouts and 5xx; other errors raise."""
        last_exc: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                response = self._client.get(self.base_url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
            else:
                if response.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                else:
                    response.raise_for_status()
                    return response.json()
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BASE_SLEEP * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def iter_studies(
        self,
        query_cond: str | None = None,
        query_term: str | None = None,
        statuses: list[str] | None = None,
        page_size: int = MAX_PAGE_SIZE,
        max_studies: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw study dicts, following `nextPageToken` cursor pagination."""
        params: dict[str, Any] = {
            "format": "json",
            "pageSize": min(page_size, MAX_PAGE_SIZE),
        }
        if query_cond:
            params["query.cond"] = query_cond
        if query_term:
            params["query.term"] = query_term
        if statuses:
            params["filter.overallStatus"] = "|".join(statuses)

        yielded = 0
        page_token: str | None = None
        while True:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token
            payload = self._get(page_params)
            studies = payload.get("studies") or []
            for study in studies:
                yield study
                yielded += 1
                if max_studies is not None and yielded >= max_studies:
                    return
            page_token = payload.get("nextPageToken")
            if not page_token or not studies:
                return


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a protocolSection submodule, or {} when absent."""
    protocol = raw.get("protocolSection") or {}
    module = protocol.get(name)
    return module if isinstance(module, dict) else {}


def parse_study(raw: dict[str, Any]) -> TrialRecord:
    """Map a raw v2 study dict onto `TrialRecord`; missing fields use defaults."""
    identification = _section(raw, "identificationModule")
    conditions_module = _section(raw, "conditionsModule")
    description = _section(raw, "descriptionModule")
    eligibility = _section(raw, "eligibilityModule")
    status = _section(raw, "statusModule")
    design = _section(raw, "designModule")

    conditions = conditions_module.get("conditions") or []
    phases = design.get("phases") or []

    return TrialRecord(
        nct_id=identification.get("nctId") or "",
        title=identification.get("briefTitle") or "",
        conditions=[str(c) for c in conditions],
        brief_summary=description.get("briefSummary") or "",
        eligibility_text=eligibility.get("eligibilityCriteria") or "",
        sex=eligibility.get("sex"),
        minimum_age=eligibility.get("minimumAge"),
        maximum_age=eligibility.get("maximumAge"),
        overall_status=status.get("overallStatus"),
        phase=[str(p) for p in phases],
    )


def download_snapshot(
    queries: list[str],
    out_path: Path,
    statuses: Sequence[str] = DEFAULT_STATUSES,
    max_per_query: int | None = None,
    client: CTGovClient | None = None,
) -> int:
    """Fetch trials for each condition query into a JSONL snapshot.

    Trials are deduped by `nct_id` across queries. Returns records written.
    """
    out_path = Path(out_path)
    owns_client = client is None
    ctgov = client or CTGovClient()
    seen: set[str] = set()
    total = 0
    try:
        for index, query in enumerate(queries):
            batch: list[TrialRecord] = []
            for raw in ctgov.iter_studies(
                query_cond=query,
                statuses=list(statuses),
                max_studies=max_per_query,
            ):
                record = parse_study(raw)
                if not record.nct_id or record.nct_id in seen:
                    continue
                seen.add(record.nct_id)
                batch.append(record)
            total += write_jsonl(batch, out_path, append=index > 0)
    finally:
        if owns_client:
            ctgov.close()
    return total
