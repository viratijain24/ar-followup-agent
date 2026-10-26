"""HTTP interface: POST /run takes invoice PDFs + payments.csv + customers.csv, returns drafts and flags.

Run from the project root:  uvicorn src.api:app --reload   (docs at http://127.0.0.1:8000/docs)
The knowledge base (policy, notes, emails) is server-side in data/corpus.
"""
from __future__ import annotations

import csv
import io
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src import coordinator

MAX_PDFS = 50
MAX_FILE_BYTES = 5 * 1024 * 1024
REQUIRED_COLUMNS = {
    "payments": {"payment_id", "payment_date", "customer_id", "invoice_reference", "amount", "currency"},
    "customers": {"customer_id", "name", "contact_name", "contact_email", "payment_terms_days", "currency"},
}

app = FastAPI(title="AR Follow-up Agent", version="0.1.0",
              description="Demo on synthetic data: extract invoices, reconcile payments, draft verified reminders.")


def _read(upload: UploadFile) -> bytes:
    data = upload.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{upload.filename} is larger than {MAX_FILE_BYTES // 1024 // 1024} MB")
    return data


def _check_csv(kind: str, data: bytes) -> None:
    try:
        header = next(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    except (UnicodeDecodeError, StopIteration):
        raise HTTPException(400, f"{kind}.csv must be a non-empty UTF-8 CSV file")
    missing = REQUIRED_COLUMNS[kind] - {h.strip() for h in header}
    if missing:
        raise HTTPException(400, f"{kind}.csv is missing columns: {sorted(missing)}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run(invoices: list[UploadFile] = File(..., description="Invoice PDFs"),
        payments: UploadFile = File(..., description="payments.csv"),
        customers: UploadFile = File(..., description="customers.csv"),
        as_of: date = Form(coordinator.AS_OF, description="Date to compute days overdue from (YYYY-MM-DD)"),
        include_trace: bool = Form(False)) -> dict:
    """Run the full pipeline. Synchronous: an uncached run takes ~15 s of model time per invoice."""
    if len(invoices) > MAX_PDFS:
        raise HTTPException(400, f"at most {MAX_PDFS} invoices per run")
    pdfs = {}
    for f in invoices:
        data = _read(f)
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, f"{f.filename} is not a PDF")
        pdfs[Path(f.filename or "invoice.pdf").name] = data  # .name drops any client-supplied directories
    tables = {"payments": _read(payments), "customers": _read(customers)}
    for kind, data in tables.items():
        _check_csv(kind, data)

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "invoices").mkdir()
        for name, data in pdfs.items():
            (data_dir / "invoices" / name).write_bytes(data)
        for kind, data in tables.items():
            (data_dir / f"{kind}.csv").write_bytes(data)
        outcomes = coordinator.run(data_dir, as_of, verbose=False)

    results = []
    for o in outcomes:
        item = asdict(o)
        item.pop("first_draft")
        if not include_trace:
            item.pop("trace")
        results.append(item)
    summary: dict[str, int] = {}
    for o in outcomes:
        summary[o.status] = summary.get(o.status, 0) + 1
    return {"as_of": as_of.isoformat(), "summary": summary, "results": results}
