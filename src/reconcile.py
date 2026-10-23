"""Reconciliation: match payments to invoices and label each invoice. Plain pandas, no LLM.

For each invoice: amount paid, balance, status (paid / partial / unpaid), days overdue, and the
collections-policy stage (see data/corpus/collections_policy.md section 1).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

TOLERANCE = 0.005  # half a cent

# Policy section 1: (lowest days overdue, stage). Checked from the top.
STAGES = [(60, "final"), (30, "firm"), (7, "friendly"), (1, "grace")]
ACTIONABLE_STAGES = {"friendly", "firm", "final"}  # stages where policy says "send a reminder"


def normalize_ref(ref: str) -> str:
    """'inv 1008' / ' INV-1008 ' / 'INV1008' -> 'INV-1008'."""
    m = re.fullmatch(r"INV[\s\-_]*(\d+)", str(ref).strip().upper())
    return f"INV-{m.group(1)}" if m else str(ref).strip().upper()


def stage_for(days_overdue: int, balance: float) -> str:
    if balance <= TOLERANCE:
        return "settled"
    for threshold, stage in STAGES:
        if days_overdue >= threshold:
            return stage
    return "not_due"


def load_payments(path: Path = Path("data/payments.csv")) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"amount": float}, parse_dates=["payment_date"])


def reconcile(invoices: list[dict], payments: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (one row per invoice with status, payments that couldn't be applied).

    `invoices` need: invoice_number, customer_id, currency, total, issue_date, due_date.
    """
    inv = pd.DataFrame(invoices)[["invoice_number", "customer_id", "currency", "total", "issue_date", "due_date"]]
    inv["invoice_number"] = inv["invoice_number"].map(normalize_ref)
    inv["due_date"] = pd.to_datetime(inv["due_date"])

    pay = payments.copy()
    pay["invoice_number"] = pay["invoice_reference"].map(normalize_ref)
    pay = pay.merge(inv[["invoice_number", "currency"]], on="invoice_number", how="left", suffixes=("", "_invoice"))

    # A payment is only applied if it references a known invoice in the same currency.
    pay["problem"] = None
    pay.loc[pay["currency_invoice"].isna(), "problem"] = "no matching invoice"
    pay.loc[pay["currency_invoice"].notna() & (pay["currency"] != pay["currency_invoice"]), "problem"] = "currency mismatch"
    applied = pay[pay["problem"].isna()]
    unapplied = pay[pay["problem"].notna()].drop(columns=["currency_invoice"])

    paid = applied.groupby("invoice_number").agg(amount_paid=("amount", "sum"),
                                                 payment_ids=("payment_id", lambda s: ",".join(s)))
    out = inv.merge(paid, on="invoice_number", how="left")
    out["amount_paid"] = out["amount_paid"].fillna(0.0).round(2)
    out["payment_ids"] = out["payment_ids"].fillna("")
    out["balance"] = (out["total"] - out["amount_paid"]).round(2)
    out["status"] = "unpaid"
    out.loc[out["amount_paid"] > TOLERANCE, "status"] = "partial"
    out.loc[out["balance"] <= TOLERANCE, "status"] = "paid"
    out["overpaid"] = out["balance"] < -TOLERANCE
    days_past_due = (pd.Timestamp(as_of) - out["due_date"]).dt.days
    out["days_overdue"] = days_past_due.clip(lower=0).where(out["balance"] > TOLERANCE, 0).astype(int)
    out["overdue"] = out["days_overdue"] > 0
    out["stage"] = [stage_for(d, b) for d, b in zip(out["days_overdue"], out["balance"])]
    out["due_date"] = out["due_date"].dt.date.astype(str)
    return out, unapplied


def main() -> None:
    import json

    truth = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    table, unapplied = reconcile(truth["invoices"], load_payments(), date.fromisoformat(truth["as_of"]))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[["invoice_number", "customer_id", "currency", "total", "amount_paid", "balance",
                     "status", "due_date", "days_overdue", "stage"]].to_string(index=False))
    print(f"\nunapplied payments: {len(unapplied)}")


if __name__ == "__main__":
    main()
