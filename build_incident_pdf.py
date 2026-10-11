"""Build reports/incident_response.pdf: the routing-fairness incident-response
table plus the current status of each trigger.

Usage:
    python build_incident_pdf.py
"""

import os

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

DASHBOARD_URL = "https://claude.ai/artifact/Tb995318rSCxCwaScEweRG"
INK = colors.HexColor("#1B2430")
MUTED = colors.HexColor("#5E6573")
RULE = colors.HexColor("#D9D7D0")
HEAD_BG = colors.HexColor("#EDEBE5")
CRIT_BG = colors.HexColor("#FBEAE7")
CRIT = colors.HexColor("#A72C29")

title = ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=17, leading=21, textColor=INK)
sub = ParagraphStyle("sub", fontName="DejaVu", fontSize=9, leading=12, textColor=MUTED)
h2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=14, textColor=INK, spaceBefore=4)
cell = ParagraphStyle("cell", fontName="DejaVu", fontSize=7.6, leading=9.2, textColor=INK, alignment=TA_LEFT)
head = ParagraphStyle("head", parent=cell, fontName="DejaVu-Bold")
note = ParagraphStyle("note", fontName="DejaVu", fontSize=8, leading=10.5, textColor=MUTED)

TRIGGERS = [
    ("1", "<b>Four-fifths ratio</b> (tier's Fast-Track rate ÷ best tier's) below 0.80", "High", "Compliance lead",
     "5 working days",
     "Confirm the data is complete. Re-run the ratio adjusted for claim amount and car age. Open an incident record. "
     "Notify the head of claims and the model owner.",
     "Fairness committee. Senior leadership if it fails 2 months in a row.",
     "Ratio is 0.80 or higher for 3 months in a row and the cause is written up."),
    ("2", "<b>Over-routing</b> (actual Queue B rate ÷ expected) above 1.25", "High", "Model owner (data science)",
     "5 working days",
     "Find which routing rules or model inputs send the tier to Queue B; check for location-linked inputs. "
     "Test the model without them. Report the impact on fraud catch rate.",
     "Fairness committee and Compliance",
     "Ratio is 1.10 or lower for 2 months and the fix is approved."),
    ("3", "<b>False investigations</b> per 100 claims, 5 or more above the best tier", "High", "SIU manager",
     "5 working days",
     "Sample 25 to 50 investigated claims from the tier that were paid in full and record why each was flagged. "
     "Start a second review before flagged-tier claims enter Queue B.",
     "Head of claims, Compliance, and Customer experience (outreach to affected customers)",
     "Gap is 3 per 100 or less for 2 months."),
    ("4", "<b>Fraud confirmed</b> per 100 investigations, a tier 35% or more below the best tier", "Medium",
     "SIU manager", "10 working days",
     "Compare confirmed-fraud rates by tier. If a tier is investigated more but caught less, the extra scrutiny "
     "isn't justified: lower the routing weight for that tier.",
     "Model owner and fairness committee",
     "Tier's rate is within 20% of the best tier's."),
    ("5", "<b>Repeated breach</b>: the same tier drifts from its baseline (3+ pts, p &lt; 0.05) two months running, "
     "or fails the four-fifths test in 3 of the last 6 months", "Critical", "Fairness committee chair (senior sponsor)",
     "2 working days",
     "Freeze routing-rule changes. Run a full review of the routing model. Agree a fix plan with deadlines. "
     "Ask Legal whether regulators must be told.",
     "Executive leadership and Legal",
     "Fix is in place, the monitor stays clean for 3 months, and the review is signed off."),
]

STATUS = [
    ("1", "Four-fifths ratio", "Tier 5 at 0.70 (Tier 4 at 0.81, just passing)", "Open"),
    ("2", "Over-routing", "Tier 5 at 2.1×, Tier 4 at 1.7× expected", "Open"),
    ("3", "False investigations", "Tier 5 +16 per 100, Tier 4 +10 per 100, vs Tier 2", "Open"),
    ("4", "Fraud confirmed", "No confirmed-fraud field in the data", "Can't check yet"),
    ("5", "Repeated breach", "Tier 5 below 0.80 in all 6 months", "Open, Critical (most urgent)"),
]


def table(rows, widths, crit_rows=(), crit_col=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in crit_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), CRIT_BG))
    t.setStyle(TableStyle(style))
    return t


def p(text, style=cell, color=None):
    if color is not None:
        style = ParagraphStyle("c", parent=style, textColor=color, fontName="DejaVu-Bold")
    return Paragraph(text, style)


def build(out="reports/incident_response.pdf"):
    doc = SimpleDocTemplate(out, pagesize=landscape(letter), leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.4 * inch, bottomMargin=0.4 * inch,
                            title="Claim Routing Incident Response Plan", author="Claims fairness monitoring")
    W = doc.width
    story = [
        Paragraph("Claim routing fairness: incident response plan", title),
        Spacer(1, 3),
        Paragraph("Covers the ZIP-tier routing monitor (synthetic claims, January–June 2026). Queue A = Fast-Track, "
                  f"Queue B = Special Investigation. Live dashboard: <a href='{DASHBOARD_URL}' color='#1F5FAE'>"
                  f"{DASHBOARD_URL}</a>", sub),
        Spacer(1, 6),
        Paragraph("Triggers, owners and actions", h2),
        Spacer(1, 4),
    ]

    hdr = ["#", "Metric and trigger", "Severity", "Owner (accountable)", "Respond within", "Immediate actions",
           "Escalate to", "Close when"]
    fr = [0.025, 0.19, 0.068, 0.10, 0.07, 0.227, 0.14, 0.18]
    rows = [[p(h, head) for h in hdr]]
    for r in TRIGGERS:
        sev_color = CRIT if r[2] == "Critical" else None
        rows.append([p(r[0]), p(r[1]), p(r[2], color=sev_color), p(r[3]), p(r[4]), p(r[5]), p(r[6]), p(r[7])])
    story.append(table(rows, [W * f for f in fr], crit_rows=[5]))
    story.append(Spacer(1, 5))
    story.append(Paragraph("In every incident: whoever spots the alert logs it within 1 working day; each incident "
                           "has one named owner, even when several teams help; closing it needs a short write-up of "
                           "the cause, the fix, and how to prevent a repeat.", note))
    story.append(Spacer(1, 8))

    srows = [[p(h, head) for h in ["#", "Trigger", "Current value", "Status"]]]
    for r in STATUS:
        color = CRIT if r[3].startswith("Open") else MUTED
        srows.append([p(r[0]), p(r[1]), p(r[2]), p(r[3], color=color)])
    story.append(KeepTogether([
        Paragraph("Where we stand today: 4 of 5 triggers open", h2),
        Spacer(1, 4),
        table(srows, [W * f for f in (0.03, 0.2, 0.47, 0.3)], crit_rows=[5]),
        Spacer(1, 6),
        Paragraph("Thresholds are suggested starting points; Compliance and Legal set the final numbers. "
                  "The four-fifths rule is a screening heuristic, not a legal finding. Per-100 gaps are measured "
                  "against Tier 2 (9.9 per 100).", note),
    ]))
    doc.build(story)
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
