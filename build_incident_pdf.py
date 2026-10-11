"""Build reports/incident_response.pdf: the routing-fairness incident-response
table (top five metrics to watch).

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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

INK = colors.HexColor("#1B2430")
MUTED = colors.HexColor("#5E6573")
RULE = colors.HexColor("#D9D7D0")
HEAD_BG = colors.HexColor("#EDEBE5")
CRIT_BG = colors.HexColor("#FBEAE7")
CRIT = colors.HexColor("#A72C29")

title = ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=17, leading=21, textColor=INK)
sub = ParagraphStyle("sub", fontName="DejaVu", fontSize=9, leading=12, textColor=MUTED)
cell = ParagraphStyle("cell", fontName="DejaVu", fontSize=9, leading=11.5, textColor=INK, alignment=TA_LEFT)
head = ParagraphStyle("head", parent=cell, fontName="DejaVu-Bold")
note = ParagraphStyle("note", fontName="DejaVu", fontSize=9, leading=12, textColor=MUTED)

TRIGGERS = [
    ("1", "<b>Four-fifths ratio</b> (tier's Fast-Track rate ÷ the best tier's) below 0.80", "High", "Compliance lead",
     "5 business days",
     "Confirm the data is complete. Re-run the ratio adjusted for claim amount and car age. Open an incident record. "
     "Notify the head of claims and the model owner.",
     "Fairness committee. Senior leadership if it fails 2 months in a row.",
     "The ratio is 0.80 or higher for 3 months in a row and the cause is written up."),
    ("2", "<b>Over-routing</b> (actual Queue B rate ÷ expected) above 1.25", "High", "Model owner (data science)",
     "5 business days",
     "Find which routing rules or model inputs send the tier to Queue B, and check for location-linked inputs. "
     "Test the model without them. Report the impact on fraud catch rate.",
     "Fairness committee and Compliance",
     "The ratio is 1.10 or lower for 2 months and the fix is approved."),
    ("3", "<b>False investigations</b> per 100 claims, 5 or more above the best tier", "High", "SIU manager",
     "5 business days",
     "Sample 25 to 50 investigated claims from the tier that were paid in full and record why each was flagged. "
     "Start a second review before flagged-tier claims enter Queue B.",
     "Head of claims, Compliance, and Customer experience (for outreach to affected customers)",
     "The gap is 3 or less for 2 months."),
    ("4", "<b>Fraud confirmed per 100 investigations</b>, a tier 35% or more below the best tier", "Medium",
     "SIU manager", "10 business days",
     "Compare confirmed-fraud rates by tier. If high tiers are investigated more but caught less, the extra "
     "scrutiny isn't justified, so lower the routing weight for that tier.",
     "Model owner and fairness committee",
     "The tier's rate is within 20% of the best tier's."),
    ("5", "<b>Repeated breach</b>: the same tier drifts from its baseline (≥3 pts, p &lt; 0.05) two months running, "
     "or fails the four-fifths test in 3 of the last 6 months", "Critical", "Fairness committee chair (senior sponsor)",
     "2 business days",
     "Freeze routing-rule changes and run a full review of the routing model. Agree a fix plan with deadlines. "
     "Ask legal whether regulators must be told.",
     "Executive leadership and Legal",
     "The fix is in place, the monitor stays clean for 3 months, and the review is signed off."),
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
        Paragraph("Incident response: top 5 metrics to watch", title),
        Spacer(1, 4),
        Paragraph("These are the five to watch first, chosen because together they catch both unfair routing and "
                  "real harm to customers.", sub),
        Spacer(1, 10),
    ]

    hdr = ["#", "Metric and trigger", "Severity", "Owner (accountable)", "Respond within", "Immediate actions",
           "Escalate to", "Close when"]
    fr = [0.025, 0.18, 0.075, 0.115, 0.075, 0.215, 0.14, 0.175]
    rows = [[p(h, head) for h in hdr]]
    for r in TRIGGERS:
        sev_color = CRIT if r[2] == "Critical" else None
        rows.append([p(r[0]), p(r[1]), p(r[2], color=sev_color), p(r[3]), p(r[4]), p(r[5]), p(r[6]), p(r[7])])
    story.append(table(rows, [W * f for f in fr], crit_rows=[5]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("In every incident, the person who spots the alert logs it within 1 business day. Each "
                           "incident has one named owner, even when several teams help. Closing it requires a short "
                           "write-up of the cause, the fix, and how to prevent a repeat.", note))
    doc.build(story)
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
