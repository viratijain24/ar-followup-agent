"""Reconciliation tests: agreement with generator ground truth, plus hand-built edge cases."""
from datetime import date

import pandas as pd
import pytest

from src.generate_data import generate
from src.reconcile import load_payments, normalize_ref, reconcile, stage_for

AS_OF = date(2026, 10, 18)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    truth = generate(out, seed=42)
    return truth, load_payments(out / "payments.csv")


def test_matches_ground_truth(generated):
    truth, payments = generated
    table, unapplied = reconcile(truth["invoices"], payments, date.fromisoformat(truth["as_of"]))
    assert unapplied.empty
    rows = table.set_index("invoice_number")
    for inv in truth["invoices"]:
        exp, got = inv["expected"], rows.loc[inv["invoice_number"]]
        assert got["status"] == exp["status"], inv["invoice_number"]
        assert got["amount_paid"] == pytest.approx(exp["amount_paid"], abs=0.005)
        assert got["balance"] == pytest.approx(exp["balance"], abs=0.005)
        assert got["days_overdue"] == exp["days_overdue"]


@pytest.mark.parametrize("days,stage", [(0, "not_due"), (1, "grace"), (6, "grace"), (7, "friendly"),
                                        (29, "friendly"), (30, "firm"), (59, "firm"), (60, "final"), (95, "final")])
def test_stage_boundaries(days, stage):
    assert stage_for(days, balance=100.0) == stage


def test_paid_invoice_is_settled_even_if_old():
    assert stage_for(0, balance=0.0) == "settled"


@pytest.mark.parametrize("raw", ["INV-1008", "inv-1008", " INV 1008 ", "INV1008", "inv_1008"])
def test_normalize_ref(raw):
    assert normalize_ref(raw) == "INV-1008"


def _invoice(number="INV-1", total=100.0, due="2026-10-01", currency="USD"):
    return {"invoice_number": number, "customer_id": "C1", "currency": currency, "total": total,
            "issue_date": "2026-09-01", "due_date": due}


def _payments(*rows):
    return pd.DataFrame([{"payment_id": f"P{i}", "payment_date": pd.Timestamp("2026-10-05"), "customer_id": "C1",
                          "invoice_reference": ref, "amount": amt, "currency": cur, "method": "ACH"}
                         for i, (ref, amt, cur) in enumerate(rows)])


def test_split_payments_sum_to_paid():
    table, _ = reconcile([_invoice()], _payments(("INV-1", 60.0, "USD"), ("inv 1", 40.0, "USD")), AS_OF)
    row = table.iloc[0]
    assert (row["status"], row["balance"], row["days_overdue"], row["stage"]) == ("paid", 0.0, 0, "settled")
    assert row["payment_ids"] == "P0,P1"


def test_overpayment_flagged():
    table, _ = reconcile([_invoice()], _payments(("INV-1", 120.0, "USD")), AS_OF)
    assert table.iloc[0]["status"] == "paid" and table.iloc[0]["overpaid"]


def test_unknown_reference_and_wrong_currency_not_applied():
    table, unapplied = reconcile([_invoice()], _payments(("INV-999", 50.0, "USD"), ("INV-1", 50.0, "EUR")), AS_OF)
    assert table.iloc[0]["status"] == "unpaid" and table.iloc[0]["days_overdue"] == 17
    assert sorted(unapplied["problem"]) == ["currency mismatch", "no matching invoice"]
