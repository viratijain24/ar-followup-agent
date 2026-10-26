"""API tests. The pipeline is replaced by a stub, so no Gemini calls are made."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api, coordinator
from src.coordinator import Outcome

PAYMENTS = b"payment_id,payment_date,customer_id,invoice_reference,amount,currency,method\n"
CUSTOMERS = b"customer_id,name,contact_name,contact_email,payment_terms_days,currency,country\n"
PDF = b"%PDF-1.4 minimal"


@pytest.fixture
def client(monkeypatch):
    seen = {}

    def fake_run(data_dir: Path, as_of, verbose=False):
        seen["files"] = sorted(p.name for p in (data_dir / "invoices").iterdir())
        seen["as_of"] = as_of
        return [Outcome("INV-1", "ready_to_send", stage="friendly", draft={"subject": "s", "body": "b"},
                        trace=[{"agent": "draft"}]),
                Outcome("INV-2", "no_action", reason="paid in full", stage="settled")]

    monkeypatch.setattr(coordinator, "run", fake_run)
    c = TestClient(api.app)
    c.seen = seen
    return c


def _files(pdf=PDF, payments=PAYMENTS, customers=CUSTOMERS, name="INV-1.pdf"):
    return [("invoices", (name, pdf, "application/pdf")), ("payments", ("payments.csv", payments, "text/csv")),
            ("customers", ("customers.csv", customers, "text/csv"))]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_run_returns_summary_and_results(client):
    r = client.post("/run", files=_files(), data={"as_of": "2026-10-18"})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"] == {"ready_to_send": 1, "no_action": 1}
    assert body["results"][0]["draft"]["subject"] == "s"
    assert "trace" not in body["results"][0]
    assert client.seen["files"] == ["INV-1.pdf"] and str(client.seen["as_of"]) == "2026-10-18"


def test_trace_on_request(client):
    r = client.post("/run", files=_files(), data={"include_trace": "true"})
    assert r.json()["results"][0]["trace"] == [{"agent": "draft"}]


def test_rejects_non_pdf(client):
    r = client.post("/run", files=_files(pdf=b"hello"))
    assert r.status_code == 400 and "not a PDF" in r.json()["detail"]


def test_rejects_csv_missing_columns(client):
    r = client.post("/run", files=_files(payments=b"payment_id,amount\n"))
    assert r.status_code == 400 and "missing columns" in r.json()["detail"]


def test_strips_directories_from_filenames(client):
    client.post("/run", files=_files(name="../../evil.pdf"))
    assert client.seen["files"] == ["evil.pdf"]
