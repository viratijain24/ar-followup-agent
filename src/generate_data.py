"""Synthetic data generator: invoice PDFs, customers.csv, payments.csv, ground_truth.json.

Ground truth is written from the same Invoice objects that render the PDFs, never by parsing them.
Run from the project root:  python -m src.generate_data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

rl_config.invariant = 1  # no timestamps/random IDs inside PDFs -> byte-identical reruns

SEED = 42
AS_OF = date(2026, 10, 18)  # fixed "today" so days-overdue never drifts between runs
SELLER = {
    "name": "Kestrel Data Systems Ltd.",
    "address": "4th Floor, 12 Harbour Street, Bengaluru 560001, India",
    "email": "billing@kestreldata.example",
}

CUSTOMERS = [
    # id, name, contact, email, terms (days), currency, country
    ("C001", "Acme Retail Inc.", "Dana Whitfield", "ap@acmeretail.example", 30, "USD", "US"),
    ("C002", "Bluefin Logistics LLC", "Marcus Ortega", "payables@bluefin.example", 45, "USD", "US"),
    ("C003", "Hoffmann Maschinenbau GmbH", "Katrin Hoffmann", "buchhaltung@hoffmann-mb.example", 30, "EUR", "DE"),
    ("C004", "Thistle & Co Ltd", "Fiona MacLeod", "accounts@thistleco.example", 30, "GBP", "GB"),
    ("C005", "Kaveri Textiles Pvt Ltd", "Arjun Menon", "finance@kaveritextiles.example", 45, "INR", "IN"),
    ("C006", "Pinecrest Dental Group", "Laura Kim", "billing@pinecrestdental.example", 15, "USD", "US"),
]

FX = {"USD": Decimal("1"), "EUR": Decimal("0.92"), "GBP": Decimal("0.79"), "INR": Decimal("83")}
TAX = {"USD": ("Sales tax", Decimal("0.08")), "EUR": ("VAT", Decimal("0.19")),
       "GBP": ("VAT", Decimal("0.20")), "INR": ("GST", Decimal("0.18"))}
SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "Rs. "}  # ₹ isn't in the built-in PDF fonts

CATALOG = [  # description, base unit price in USD
    ("Analytics dashboard licence (monthly)", 450),
    ("Data pipeline setup", 2200),
    ("Support hours", 95),
    ("Model training compute (GPU-hours)", 3),
    ("Onboarding workshop", 1200),
    ("Custom report development", 780),
    ("API access tier - Pro (monthly)", 300),
    ("Data quality audit", 1500),
]

# (status, days past due as of AS_OF; negative = not yet due). Fixed so every policy stage appears.
SCENARIOS = (
    [("paid", d) for d in (-20, -5, 10, 15, 25, 40, 55, 70)]
    + [("partial", d) for d in (-5, 8, 20, 35, 62)]
    + [("unpaid", d) for d in (-12, 3, 10, 31, 45, 70, 95)]
)
LAYOUTS = ("classic", "modern", "compact")


def money(x: Decimal) -> Decimal:
    """Round to cents, half-up (how invoices round, unlike Python's banker's rounding)."""
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: Decimal
    amount: Decimal


@dataclass
class Customer:
    customer_id: str
    name: str
    contact_name: str
    contact_email: str
    payment_terms_days: int
    currency: str
    country: str


@dataclass
class Invoice:
    invoice_number: str
    customer: Customer
    currency: str
    issue_date: date
    due_date: date
    line_items: list[LineItem]
    tax_label: str
    tax_rate: Decimal
    layout: str
    scenario: str  # planned payment status: paid / partial / unpaid
    subtotal: Decimal = field(init=False)
    tax: Decimal = field(init=False)
    total: Decimal = field(init=False)

    def __post_init__(self) -> None:
        self.subtotal = sum((li.amount for li in self.line_items), Decimal("0"))
        self.tax = money(self.subtotal * self.tax_rate)
        self.total = self.subtotal + self.tax


@dataclass
class Payment:
    payment_id: str
    payment_date: date
    customer_id: str
    invoice_reference: str
    amount: Decimal
    currency: str
    method: str


# ---------- building the data ----------

def make_line_items(rng: random.Random, currency: str) -> list[LineItem]:
    items = []
    for desc, base in rng.sample(CATALOG, k=rng.randint(1, 4)):
        qty = rng.randint(20, 400) if "GPU" in desc else rng.randint(1, 40) if "hours" in desc else rng.randint(1, 3)
        price = money(Decimal(base) * FX[currency])
        if currency == "INR":
            price = price.quantize(Decimal("1"))  # rupee prices are whole numbers
        items.append(LineItem(desc, qty, money(price), money(price * qty)))
    return items


def build_invoices(rng: random.Random, customers: list[Customer]) -> list[Invoice]:
    scenarios = SCENARIOS[:]
    rng.shuffle(scenarios)
    drafts = []
    for i, (status, days_past_due) in enumerate(scenarios):
        cust = customers[i % len(customers)]
        due = AS_OF - timedelta(days=days_past_due)
        issue = due - timedelta(days=cust.payment_terms_days)
        drafts.append((issue, cust, due, status))
    drafts.sort(key=lambda d: (d[0], d[1].customer_id))  # number invoices in issue-date order, like a real ledger
    invoices = []
    for n, (issue, cust, due, status) in enumerate(drafts):
        label, rate = TAX[cust.currency]
        invoices.append(Invoice(
            invoice_number=f"INV-{1001 + n}", customer=cust, currency=cust.currency,
            issue_date=issue, due_date=due, line_items=make_line_items(rng, cust.currency),
            tax_label=label, tax_rate=rate, layout=LAYOUTS[n % len(LAYOUTS)], scenario=status,
        ))
    return invoices


def build_payments(rng: random.Random, invoices: list[Invoice]) -> list[Payment]:
    payments = []
    for inv in invoices:
        if inv.scenario == "unpaid":
            continue
        if inv.scenario == "paid":
            amount = inv.total
            latest = min(AS_OF, inv.due_date + timedelta(days=15))
        else:
            amount = money(inv.total * Decimal(str(rng.choice([0.25, 0.4, 0.5, 0.6]))))
            latest = AS_OF
        span = max((latest - inv.issue_date).days, 1)
        pay_date = inv.issue_date + timedelta(days=rng.randint(1, span))
        payments.append(Payment("", pay_date, inv.customer.customer_id, inv.invoice_number,
                                amount, inv.currency, rng.choice(["bank transfer", "ACH", "card", "cheque"])))
    payments.sort(key=lambda p: (p.payment_date, p.invoice_reference))
    for k, p in enumerate(payments, start=1):
        p.payment_id = f"PAY-{k:04d}"
    return payments


def expected_reconciliation(inv: Invoice, payments: list[Payment]) -> dict:
    """What reconciliation (step 3) should conclude for this invoice as of AS_OF."""
    paid = sum((p.amount for p in payments if p.invoice_reference == inv.invoice_number), Decimal("0"))
    balance = inv.total - paid
    status = "paid" if balance <= 0 else "partial" if paid > 0 else "unpaid"
    days_overdue = max(0, (AS_OF - inv.due_date).days) if balance > 0 else 0
    return {"status": status, "amount_paid": float(paid), "balance": float(balance),
            "overdue": days_overdue > 0, "days_overdue": days_overdue}


# ---------- rendering PDFs (3 layouts) ----------

def fmt_date(d: date, layout: str) -> str:
    if layout == "classic":
        return f"{d:%B} {d.day}, {d.year}"  # October 3, 2026
    if layout == "modern":
        return d.isoformat()  # 2026-10-03
    return f"{d:%d.%m.%Y}"  # 03.10.2026


def fmt_money(x: Decimal, currency: str, layout: str) -> str:
    if layout == "classic":
        return f"{SYMBOL[currency]}{x:,.2f}"  # $4,850.00
    if layout == "modern":
        return f"{currency} {x:,.2f}"  # USD 4,850.00
    return f"{x:,.2f} {currency}"  # 4,850.00 USD


def render_pdf(inv: Invoice, path: Path) -> None:
    styles = getSampleStyleSheet()
    L = inv.layout
    m = lambda x: fmt_money(x, inv.currency, L)  # noqa: E731
    d = lambda x: fmt_date(x, L)  # noqa: E731
    c = inv.customer
    story = []

    if L == "classic":
        story += [Paragraph(f"<b>{SELLER['name']}</b>", styles["Title"]),
                  Paragraph(f"{SELLER['address']}<br/>{SELLER['email']}", styles["Normal"]),
                  Spacer(1, 8 * mm), Paragraph("<b>INVOICE</b>", styles["Heading1"])]
        meta = [["Invoice #", inv.invoice_number], ["Date", d(inv.issue_date)],
                ["Due Date", d(inv.due_date)], ["Terms", f"Net {c.payment_terms_days}"]]
        bill_to = Paragraph(f"<b>Bill To:</b><br/>{c.name}<br/>Attn: {c.contact_name}<br/>{c.contact_email}",
                            styles["Normal"])
        story += [Table([[bill_to, Table(meta)]], colWidths=[95 * mm, 75 * mm]), Spacer(1, 8 * mm)]
        header = ["Description", "Qty", "Unit Price", "Amount"]
        grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)])
    elif L == "modern":
        banner = Table([[Paragraph(f"<font color='white' size='18'><b>{SELLER['name']}</b></font>", styles["Normal"]),
                         Paragraph(f"<font color='white' size='14'>Invoice No. {inv.invoice_number}</font>",
                                   styles["Normal"])]], colWidths=[105 * mm, 65 * mm], rowHeights=[18 * mm])
        banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1f4e79")),
                                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story += [banner, Spacer(1, 6 * mm),
                  Paragraph(f"Issued: {d(inv.issue_date)} &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"Payment due: <b>{d(inv.due_date)}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"Currency: {inv.currency}", styles["Normal"]),
                  Spacer(1, 4 * mm),
                  Paragraph(f"<b>Customer</b>: {c.name} ({c.customer_id})<br/>"
                            f"Contact: {c.contact_name}, {c.contact_email}", styles["Normal"]),
                  Spacer(1, 6 * mm)]
        header = ["Item", "Quantity", "Rate", "Line total"]
        grid = TableStyle([("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.HexColor("#1f4e79")),
                           ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.lightgrey)])
    else:  # compact
        story += [Table([[Paragraph(f"<b>{SELLER['name']}</b><br/><font size='8'>{SELLER['address']}</font>",
                                    styles["Normal"]),
                          Paragraph(f"<font size='9'>Ref: <b>{inv.invoice_number}</b><br/>"
                                    f"Inv. date: {d(inv.issue_date)}<br/>Due: {d(inv.due_date)}</font>",
                                    styles["Normal"])]], colWidths=[110 * mm, 60 * mm]),
                  Spacer(1, 5 * mm),
                  Paragraph(f"<font size='9'>To: {c.name}, {c.contact_name} &lt;{c.contact_email}&gt; "
                            f"- payable within {c.payment_terms_days} days</font>", styles["Normal"]),
                  Spacer(1, 4 * mm)]
        header = ["Pos. / Description", "Qty", "Price", "Total"]
        grid = TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BOX", (0, 0), (-1, -1), 0.5, colors.black)])

    rows = [header] + [[li.description, str(li.quantity), m(li.unit_price), m(li.amount)] for li in inv.line_items]
    rows += [["", "", "Subtotal", m(inv.subtotal)],
             ["", "", f"{inv.tax_label} {inv.tax_rate * 100:.0f}%", m(inv.tax)],
             ["", "", "TOTAL DUE" if L != "compact" else "Total", m(inv.total)]]
    table = Table(rows, colWidths=[80 * mm, 20 * mm, 35 * mm, 35 * mm])
    table.setStyle(grid)
    table.setStyle(TableStyle([("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("FONTNAME", (2, -1), (-1, -1), "Helvetica-Bold")]))
    story += [table, Spacer(1, 10 * mm),
              Paragraph(f"Please quote {inv.invoice_number} with your payment. Bank: Example Bank, "
                        f"A/C 000123456789.", styles["Italic"])]
    SimpleDocTemplate(str(path), pagesize=A4, title=inv.invoice_number,
                      leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm).build(story)


# ---------- writing outputs ----------

def invoice_truth(inv: Invoice, payments: list[Payment]) -> dict:
    return {
        "invoice_number": inv.invoice_number,
        "customer_id": inv.customer.customer_id,
        "customer_name": inv.customer.name,
        "currency": inv.currency,
        "issue_date": inv.issue_date.isoformat(),
        "due_date": inv.due_date.isoformat(),
        "subtotal": float(inv.subtotal),
        "tax": float(inv.tax),
        "total": float(inv.total),
        "line_items": [{"description": li.description, "quantity": li.quantity,
                        "unit_price": float(li.unit_price), "amount": float(li.amount)} for li in inv.line_items],
        "layout": inv.layout,
        "pdf_file": f"invoices/{inv.invoice_number}.pdf",
        "expected": expected_reconciliation(inv, payments),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(out_dir: Path, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    customers = [Customer(*row) for row in CUSTOMERS]
    invoices = build_invoices(rng, customers)
    payments = build_payments(rng, invoices)

    (out_dir / "invoices").mkdir(parents=True, exist_ok=True)
    for inv in invoices:
        render_pdf(inv, out_dir / "invoices" / f"{inv.invoice_number}.pdf")
    write_csv(out_dir / "customers.csv", [asdict(c) for c in customers])
    write_csv(out_dir / "payments.csv", [{**asdict(p), "payment_date": p.payment_date.isoformat(),
                                          "amount": f"{p.amount:.2f}"} for p in payments])
    truth = {"seed": seed, "as_of": AS_OF.isoformat(), "seller": SELLER["name"],
             "invoices": [invoice_truth(inv, payments) for inv in invoices]}
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    truth = generate(args.out, args.seed)
    for t in truth["invoices"]:
        e = t["expected"]
        print(f"{t['invoice_number']}  {t['customer_id']}  {t['layout']:<8} {t['currency']} {t['total']:>12,.2f}  "
              f"due {t['due_date']}  {e['status']:<8} days_overdue={e['days_overdue']}")
    print(f"\nWrote {len(truth['invoices'])} PDFs, customers.csv, payments.csv, ground_truth.json to {args.out}/")


if __name__ == "__main__":
    main()
