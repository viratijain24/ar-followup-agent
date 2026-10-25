"""Deterministic verifier tests: value extraction across formats, and the checks on a draft."""
from datetime import date

import pytest

from src.draft import Draft, DraftContext
from src.rag import Chunk, Retrieved
from src.verify import check_values, extract_amounts, extract_dates, extract_invoice_numbers

AS_OF = date(2026, 10, 18)


@pytest.mark.parametrize("text,expected", [
    ("balance of USD 3,758.40", [(3758.40, "USD")]),
    ("balance of $3,758.40", [(3758.40, "USD")]),
    ("balance of 3,758.40 USD", [(3758.40, "USD")]),
    ("balance of 3758.40", [(3758.40, None)]),
    ("Rs. 2,71,195.86 is due", [(271195.86, "INR")]),
    ("£887.33 and €12.00", [(887.33, "GBP"), (12.00, "EUR")]),
    ("USD 6,264 total", [(6264.00, "USD")]),
    ("a charge of 1.5% per month, within 7 days, account 000123456789", []),
])
def test_extract_amounts(text, expected):
    assert extract_amounts(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("due on 13 September 2026", date(2026, 9, 13)),
    ("due on September 13, 2026", date(2026, 9, 13)),
    ("due on 2026-09-13", date(2026, 9, 13)),
    ("due on 13.09.2026", date(2026, 9, 13)),
    ("due on 13th of Sept 2026", date(2026, 9, 13)),
    ("received on 16 September", date(2026, 9, 16)),  # no year -> as-of year
])
def test_extract_dates(text, expected):
    assert [d for d, _ in extract_dates(text, AS_OF.year)] == [expected]


def test_extract_dates_ignores_non_dates_and_flags_invalid():
    assert extract_dates("please pay within 7 days; 2 data audits", AS_OF.year) == []
    assert extract_dates("due on 31 September 2026", AS_OF.year)[0][0] is None


def test_extract_invoice_numbers():
    assert extract_invoice_numbers("Re: INV-1008 (see also inv 1009 and INV1010)") == {"INV-1008", "INV-1009", "INV-1010"}


def _ctx(customer_chunks=()):
    invoice = {"invoice_number": "INV-1008", "customer_name": "Bluefin Logistics LLC", "currency": "USD",
               "issue_date": "2026-07-30", "due_date": "2026-09-13", "subtotal": 5800.0, "tax": 464.0,
               "total": 6264.0, "line_items": [{"description": "Data pipeline setup", "quantity": 1,
                                                "unit_price": 2200.0, "amount": 2200.0}]}
    row = {"invoice_number": "INV-1008", "status": "partial", "amount_paid": 2505.6, "balance": 3758.4,
           "days_overdue": 35, "stage": "firm"}
    payments = [{"amount": 2505.6, "payment_date": "2026-09-16"}]
    return DraftContext(row, invoice, {"contact_name": "Marcus Ortega"}, payments,
                        Retrieved(list(customer_chunks), [], "", {}), AS_OF)


CLEAN_BODY = ("Dear Marcus,\n\nInvoice INV-1008 for USD 6,264.00 was due on 13 September 2026. Thank you for "
              "your payment of USD 2,505.60 received on 16 September 2026. The outstanding balance is "
              "USD 3,758.40. Please pay within 7 days by bank transfer to Example Bank, account 000123456789, "
              "quoting INV-1008.\n\nAccounts Receivable, Kestrel Data Systems Ltd.")


def _draft(body=CLEAN_BODY, subject="Reminder: INV-1008 is overdue", action="send"):
    return Draft(action=action, hold_reason="", subject=subject, body=body)


def test_clean_draft_passes():
    result = check_values(_draft(), _ctx())
    assert result.ok, result.problems


@pytest.mark.parametrize("old,new,expected_problem", [
    ("USD 3,758.40", "USD 3,785.40", "amount 3,785.40"),                 # transposed digits
    ("13 September 2026", "15 September 2026", "date '15 September 2026'"),  # wrong date
    ("quoting INV-1008", "quoting INV-1009", "other invoice number"),   # another invoice
    ("USD 3,758.40", "EUR 3,758.40", "written in EUR"),                 # wrong currency
    ("The outstanding balance is USD 3,758.40. ", "", "missing the outstanding balance"),
    ("account 000123456789", "our account", "missing payment instructions"),
])
def test_corruptions_are_caught(old, new, expected_problem):
    result = check_values(_draft(CLEAN_BODY.replace(old, new)), _ctx())
    assert not result.ok
    assert any(expected_problem in p for p in result.problems), result.problems


def test_date_from_retrieved_email_is_allowed():
    promise = Chunk("correspondence:x", "correspondence", "x", "We will pay by 2026-10-25.", customer_id="C002")
    body = CLEAN_BODY.replace("Please pay", "You mentioned payment by 25 October 2026. Please pay")
    assert check_values(_draft(body), _ctx([promise])).ok
    assert not check_values(_draft(body), _ctx()).ok


def test_held_draft_is_not_value_checked():
    assert check_values(Draft(action="hold_for_review", hold_reason="disputed", subject="", body=""), _ctx()).ok


def test_role_swap_is_a_known_blind_spot():
    """Asking for the original total instead of the balance uses a real value, so the value check can't see it."""
    body = CLEAN_BODY.replace("The outstanding balance is USD 3,758.40.",
                              "The outstanding balance is USD 6,264.00 (balance USD 3,758.40 after payment).")
    assert check_values(_draft(body), _ctx()).ok
