"""Data-layer tests: CT.gov client (mocked transport), TREC loaders, JSONL store.

No network and no API key: every HTTP call goes through `httpx.MockTransport`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from trialmatch.config import PROJECT_ROOT
from trialmatch.data.ctgov import CTGovClient, download_snapshot, parse_study
from trialmatch.data.store import load_trials, read_jsonl, write_jsonl
from trialmatch.data.trec import load_qrels, load_topics_json, load_topics_xml
from trialmatch.schemas import TrialRecord

FIXTURES = Path(__file__).parent / "fixtures"


def _study(nct_id: str) -> dict:
    """A realistic-but-minimal API v2 study payload."""
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Study {nct_id}"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "descriptionModule": {"briefSummary": "A summary."},
            "conditionsModule": {"conditions": ["Migraine", "Headache"]},
            "designModule": {"phases": ["PHASE2", "PHASE3"]},
            "eligibilityModule": {
                "eligibilityCriteria": (
                    "Inclusion Criteria:\n* Age 18+\n\nExclusion Criteria:\n* Pregnancy"
                ),
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "75 Years",
            },
        }
    }


def _paged_transport(pages: list[dict], recorder: list[httpx.Request] | None = None):
    """MockTransport that serves `pages` in order, one per request."""
    state = {"index": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        page = pages[min(state["index"], len(pages) - 1)]
        state["index"] += 1
        return httpx.Response(200, json=page)

    return httpx.MockTransport(handler)


# --- CTGovClient ---------------------------------------------------------


def test_iter_studies_follows_cursor_pagination():
    pages = [
        {"studies": [_study("NCT00000001"), _study("NCT00000002")], "nextPageToken": "tok2"},
        {"studies": [_study("NCT00000003")]},
    ]
    requests: list[httpx.Request] = []
    with CTGovClient(transport=_paged_transport(pages, requests)) as client:
        studies = list(client.iter_studies(query_cond="migraine", statuses=["RECRUITING"]))

    assert [s["protocolSection"]["identificationModule"]["nctId"] for s in studies] == [
        "NCT00000001",
        "NCT00000002",
        "NCT00000003",
    ]
    assert len(requests) == 2
    first, second = (dict(r.url.params) for r in requests)
    assert first["query.cond"] == "migraine"
    assert first["filter.overallStatus"] == "RECRUITING"
    assert first["format"] == "json"
    assert "pageToken" not in first
    assert second["pageToken"] == "tok2"


def test_iter_studies_joins_statuses_and_respects_max_studies():
    pages = [{"studies": [_study(f"NCT{i:08d}") for i in range(5)], "nextPageToken": "tok2"}]
    requests: list[httpx.Request] = []
    with CTGovClient(transport=_paged_transport(pages, requests)) as client:
        studies = list(
            client.iter_studies(
                query_term="phase 2",
                statuses=["RECRUITING", "COMPLETED"],
                page_size=5,
                max_studies=2,
            )
        )

    assert len(studies) == 2  # stops mid-page, no second request
    assert len(requests) == 1
    params = dict(requests[0].url.params)
    assert params["filter.overallStatus"] == "RECRUITING|COMPLETED"
    assert params["query.term"] == "phase 2"
    assert params["pageSize"] == "5"


def test_iter_studies_stops_on_empty_page():
    pages = [{"studies": [], "nextPageToken": "tok2"}]
    with CTGovClient(transport=_paged_transport(pages)) as client:
        assert list(client.iter_studies(query_cond="none")) == []


def test_get_raises_on_client_error():
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "bad"}))
    with CTGovClient(transport=transport) as client, pytest.raises(httpx.HTTPStatusError):
        list(client.iter_studies(query_cond="x"))


def test_get_retries_once_on_server_error(monkeypatch):
    monkeypatch.setattr("trialmatch.data.ctgov.time.sleep", lambda _seconds: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"studies": [_study("NCT00000009")]})

    with CTGovClient(transport=httpx.MockTransport(handler)) as client:
        studies = list(client.iter_studies(query_cond="x"))

    assert calls["n"] == 2
    assert len(studies) == 1


def test_parse_study_maps_all_fields():
    record = parse_study(_study("NCT00000001"))
    assert record.nct_id == "NCT00000001"
    assert record.title == "Study NCT00000001"
    assert record.conditions == ["Migraine", "Headache"]
    assert record.brief_summary == "A summary."
    assert record.eligibility_text.startswith("Inclusion Criteria:")
    assert record.sex == "ALL"
    assert record.minimum_age == "18 Years"
    assert record.maximum_age == "75 Years"
    assert record.overall_status == "RECRUITING"
    assert record.phase == ["PHASE2", "PHASE3"]


def test_parse_study_tolerates_missing_modules():
    record = parse_study({"protocolSection": {"identificationModule": {"nctId": "NCT1"}}})
    assert record.nct_id == "NCT1"
    assert record.title == ""
    assert record.conditions == []
    assert record.eligibility_text == ""
    assert record.sex is None
    assert record.phase == []

    empty = parse_study({})
    assert empty.nct_id == "" and empty.conditions == []


def test_download_snapshot_dedupes_across_queries(tmp_path):
    pages = [
        {"studies": [_study("NCT00000001"), _study("NCT00000002")]},
        {"studies": [_study("NCT00000002"), _study("NCT00000003")]},
    ]
    out_path = tmp_path / "nested" / "trials.jsonl"
    with CTGovClient(transport=_paged_transport(pages)) as client:
        written = download_snapshot(["migraine", "headache"], out_path, client=client)

    assert written == 3
    assert sorted(load_trials(out_path)) == ["NCT00000001", "NCT00000002", "NCT00000003"]


# --- TREC loaders --------------------------------------------------------


def test_load_topics_json_reads_synthetic_patients():
    topics = load_topics_json(PROJECT_ROOT / "synthetic-patients.json")
    assert len(topics) == 10
    assert topics[0][0] == "S001"
    assert "epigastric pain" in topics[0][1]
    assert all(isinstance(t, str) and isinstance(x, str) for t, x in topics)


def test_load_topics_xml_strips_whitespace():
    topics = load_topics_xml(FIXTURES / "topics_sample.xml")
    assert [t[0] for t in topics] == ["1", "2"]
    assert topics[0][1] == (
        "A 45-year-old woman with a history of migraine presents with photophobia and nausea."
    )


def test_load_qrels_parses_and_skips_bad_lines():
    qrels = load_qrels(FIXTURES / "qrels_sample.txt")
    assert qrels == {
        "1": {"NCT00000102": 2, "NCT00000104": 1, "NCT00000106": 0},
        "2": {"NCT00000102": 1, "NCT00000112": 2},
    }


# --- store ---------------------------------------------------------------


def test_write_read_round_trip(tmp_path):
    records = [
        TrialRecord(nct_id="NCT1", title="One", conditions=["Migraine"]),
        TrialRecord(nct_id="NCT2", title="Two", eligibility_text="Age 18+"),
    ]
    path = tmp_path / "sub" / "trials.jsonl"

    assert write_jsonl(records, path) == 2
    assert list(read_jsonl(path)) == records

    # One JSON object per line.
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["nct_id"] == "NCT1"


def test_write_jsonl_append_vs_truncate(tmp_path):
    path = tmp_path / "trials.jsonl"
    write_jsonl([TrialRecord(nct_id="NCT1")], path)
    write_jsonl([TrialRecord(nct_id="NCT2")], path, append=True)
    assert [r.nct_id for r in read_jsonl(path)] == ["NCT1", "NCT2"]

    write_jsonl([TrialRecord(nct_id="NCT3")], path)
    assert [r.nct_id for r in read_jsonl(path)] == ["NCT3"]


def test_load_trials_keys_by_nct_id(tmp_path):
    path = tmp_path / "trials.jsonl"
    write_jsonl([TrialRecord(nct_id="NCT1", title="One"), TrialRecord(nct_id="NCT2")], path)
    trials = load_trials(path)
    assert set(trials) == {"NCT1", "NCT2"}
    assert trials["NCT1"].title == "One"


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text('{"nct_id": "NCT1"}\n\n{"nct_id": "NCT2"}\n', encoding="utf-8")
    assert [r.nct_id for r in read_jsonl(path)] == ["NCT1", "NCT2"]
