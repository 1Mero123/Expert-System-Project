"""
report.py
=========
Export helpers for the dashboard:
  * alerts_dataframe()  - every fired rule as a flat table (CSV export)
  * build_pdf_report()  - a printable multi-page PDF match report

Note: the PDF uses the built-in Latin-1 fonts, so characters outside Latin-1
(e.g. Arabic player names) are printed as "?" in the PDF only; the dashboard
and the CSV exports keep them intact.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF
from fpdf.enums import TableBordersLayout, XPos, YPos
from fpdf.fonts import FontFace

from expert_system import DX_PREFIX, SEVERITY_ORDER, FiredRule, InferenceResult

ASSETS = Path(__file__).parent / "assets"
SEV_RGB = {"Critical": (239, 68, 68), "High": (245, 158, 11), "Medium": (59, 130, 246),
           "Info": (16, 185, 129), "OK": (16, 185, 129)}
INK, MUTED, SLATE, EMERALD = (31, 41, 55), (107, 114, 128), (17, 24, 39), (16, 185, 129)

_REPLACEMENTS = {"·": "|", "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "→": "->",
                 "≤": "<=", "≥": ">=", "➡": "->", "↳": "->", "•": "-", "…": "...", "✅": "", "⚠": "!", "🔗": ""}


def safe(value) -> str:
    """Make text printable with the PDF's built-in Latin-1 fonts."""
    text = str(value)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def fmt(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _is_chained(f: FiredRule) -> bool:
    return any(c.fact.startswith(DX_PREFIX) for c, _ in f.evidence)


# ---------------------------------------------------------------------------
# CSV: all fired rules as one flat table
# ---------------------------------------------------------------------------
ALERT_COLUMNS = ["Minute", "Player", "Position", "Rule", "Scope", "Severity", "Diagnosis",
                 "Recommendation", "Status", "Chained", "Cycle"]


def alerts_dataframe(rows: list[dict], results: list[InferenceResult]) -> pd.DataFrame:
    """One line per fired rule. `rows` holds the input rows (for the Minute), aligned with `results`."""
    out = []
    for row, res in zip(rows, results):
        wm = res.working_memory
        for f in sorted(res.fired, key=lambda f: (SEVERITY_ORDER[f.rule.severity], f.cycle)):
            out.append({
                "Minute": row.get("Minute"), "Player": wm["Player_Name"], "Position": wm["Player_Position"],
                "Rule": f.rule.rule_id, "Scope": f.rule.scope, "Severity": f.rule.severity,
                "Diagnosis": f.rule.diagnosis, "Recommendation": f.recommendation,
                "Status": "Active" if f.superseded_by is None else f"Merged into {f.superseded_by}",
                "Chained": "yes" if _is_chained(f) else "no", "Cycle": f.cycle,
            })
    return pd.DataFrame(out, columns=ALERT_COLUMNS)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
class _ReportPDF(FPDF):
    def header(self):
        logo = ASSETS / "logo_for_light_bg.png"
        if logo.exists():
            self.image(str(logo), x=10, y=6.5, w=12)
        self.set_xy(25, 8)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*SLATE)
        self.cell(0, 6, "PitchSense AI")
        self.set_xy(25, 14.5)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*MUTED)
        self.cell(0, 4, "Tactical match report - rule-based expert system")
        self.set_draw_color(*EMERALD)
        self.set_line_width(0.6)
        self.line(10, 21.5, 200, 21.5)
        self.set_xy(10, 26)
        self.set_text_color(*INK)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"Generated {datetime.now():%Y-%m-%d %H:%M}  |  PitchSense AI  |  Page {self.page_no()}/{{nb}}",
                  align="C")


def _heading(pdf: _ReportPDF, text: str) -> None:
    if pdf.will_page_break(30):
        pdf.add_page()
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12.5)
    pdf.set_text_color(*SLATE)
    pdf.cell(0, 7, safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*EMERALD)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 60, pdf.get_y())
    pdf.ln(2.5)
    pdf.set_text_color(*INK)


def _paragraph(pdf: _ReportPDF, text: str, size: float = 9, style: str = "", color=INK, font: str = "Helvetica") -> None:
    pdf.set_font(font, style, size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, size * 0.52, safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*INK)


def _table(pdf: _ReportPDF, header: list[str], rows: list[list], widths: list[float], aligns: list[str] | None = None,
           sev_col: int | None = None, size: float = 8) -> None:
    """Draw a table; if sev_col is given, that column is rendered as a coloured severity badge."""
    pdf.set_font("Helvetica", "", size)
    head = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=SLATE)
    with pdf.table(col_widths=widths, text_align=aligns or tuple("LEFT" for _ in widths), line_height=size * 0.5 + 1.2,
                   headings_style=head, borders_layout=TableBordersLayout.HORIZONTAL_LINES,
                   padding=1.2) as table:
        r = table.row()
        for h in header:
            r.cell(safe(h))
        for row in rows:
            r = table.row()
            for i, v in enumerate(row):
                if sev_col is not None and i == sev_col and str(v) in SEV_RGB:
                    r.cell(safe(v), style=FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=SEV_RGB[str(v)]))
                else:
                    r.cell(safe(fmt(v)))
    pdf.ln(1)


def build_pdf_report(result: InferenceResult, *, source: str, minute: int | None = None,
                     timeline: pd.DataFrame | None = None, squad: pd.DataFrame | None = None,
                     team_alerts: list[FiredRule] | None = None) -> bytes:
    """Render the analysis of one time window (plus optional timeline / squad tables) as a PDF."""
    wm = result.working_memory
    fired = sorted(result.fired, key=lambda f: (SEVERITY_ORDER[f.rule.severity], f.cycle))
    active = [f for f in fired if f.superseded_by is None]
    merged = [f for f in fired if f.superseded_by is not None]

    pdf = _ReportPDF(format="A4")
    pdf.set_margins(10, 10, 10)
    pdf.set_auto_page_break(True, margin=16)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ---- title block --------------------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*SLATE)
    pdf.cell(0, 10, "Match Tactical Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    window = f"{minute - 15}'-{minute}'" if minute is not None else "Manual simulation"
    _table(pdf, ["Player", "Position", "Analysed window", "Data source"],
           [[wm["Player_Name"], wm["Player_Position"], window, source]], [48, 40, 42, 60], size=9)

    # ---- summary ---------------------------------------------------------------------------------------
    _heading(pdf, "1. Summary")
    counts = {s: sum(1 for f in active if f.rule.severity == s) for s in SEVERITY_ORDER}
    if active:
        parts = ", ".join(f"{n} {s.lower()}" for s, n in counts.items() if n)
        _paragraph(pdf, f"{len(active)} active alert(s): {parts}.", 10, "B", SEV_RGB[active[0].rule.severity])
        _table(pdf, ["Severity", "Rule", "Diagnosis", "Recommendation"],
               [[f.rule.severity, f.rule.rule_id, f.rule.diagnosis, f.recommendation] for f in active],
               [20, 13, 58, 99], sev_col=0)
    else:
        _paragraph(pdf, "No tactical alerts: the current metrics trigger none of the rules.", 10, "B", SEV_RGB["OK"])
    if merged:
        _paragraph(pdf, "Merged into a broader recommendation: "
                   + "; ".join(f"{f.rule.rule_id} -> {f.superseded_by}" for f in merged) + ".", 8.5, "I", MUTED)

    # ---- metrics -----------------------------------------------------------------------------------------
    _heading(pdf, "2. Match metrics (working memory)")
    rows = [
        ["Sprint fatigue", f'{wm["Player_Sprint_Fatigue"]} %', "R2 > 85 | R9 > 80"],
        ["Pass accuracy under press", f'{wm["Pass_Accuracy_Under_Press"]} %', "R7 < 70"],
        ["Distance covered", f'{wm["Distance_Covered_km"]:.1f} km', "R9 > 11.5"],
        ["Ball losses (per 15')", f'{wm["Ball_Loss_Count"]} ({wm["Ball_Loss_Turnovers"]})', "High >= 8"],
        ["Opponent build-up", f'{wm["Opponent_Build_Up_Time_s"]:.1f} s ({wm["Opponent_Build_Up_Speed"]})', "Fast <= 8 s"],
        ["Our pressing (PPDA)", f'{wm["Our_PPDA"]:.1f} ({wm["Our_Pressing_Intensity"]})', "High <= 8 | Low > 12"],
        ["Space between lines", f'{wm["Space_Between_Lines"]:.1f} m', "R3 > 15 | R10 > 16"],
        ["Opponent through balls (per 15')", f'{wm["Opponent_Through_Ball_Count"]} ({wm["Opponent_Through_Balls"]})', "Frequent >= 4"],
        ["Opponent xG (per 15')", f'{wm["Opponent_xG_Created"]:.2f} ({wm["Opponent_Threat_Level"]})', "R8 > 0.75"],
        ["Midfield line height", f'{wm["Midfield_Line_Height_m"]:.0f} m ({wm["Midfield_Line_Height"]})', "Low < 35 m"],
        ["Substitutions available", wm["Substitutes_Available"], "R4 needs > 0"],
    ]
    _table(pdf, ["Metric", "Value", "Related threshold"], rows, [70, 60, 60])

    # ---- squad ---------------------------------------------------------------------------------------------
    if squad is not None and len(squad):
        _heading(pdf, f"3. Squad view at {minute}'")
        if team_alerts:
            _paragraph(pdf, "Team-wide alerts: " + "; ".join(f"{f.rule.rule_id} {f.rule.diagnosis}" for f in team_alerts) + ".",
                       8.5, "I", MUTED)
        cols = ["Rank", "Player", "Position", "Status", "Fatigue %", "Pass acc. %", "Distance km", "Ball losses", "Player alerts"]
        _table(pdf, cols, squad[cols].values.tolist(), [11, 27, 26, 17, 15, 16, 17, 14, 47], sev_col=3, size=7.5)
        top = squad.iloc[0]
        if top["Status"] != "OK":
            _paragraph(pdf, f'Highest player risk: {top["Player"]} ({top["Status"]}) - {top["Recommendation"]}', 9, "B",
                       SEV_RGB[top["Status"]])

    # ---- timeline --------------------------------------------------------------------------------------------
    if timeline is not None and len(timeline):
        _heading(pdf, f'4. Match timeline - {wm["Player_Name"]}')
        cols = ["Minute", "Fatigue %", "Pass acc. %", "Distance km", "Ball losses", "Opp. xG", "Alerts", "Worst", "Diagnoses"]
        data = timeline.assign(Worst=timeline["Worst"].fillna("OK"),
                               **{"Opp. xG": timeline["Opp. xG"].map(lambda v: f"{v:.2f}")})[cols].values.tolist()
        _table(pdf, cols, data, [13, 15, 16, 17, 15, 13, 12, 17, 72], sev_col=7, size=7.5)

    # ---- explanation -----------------------------------------------------------------------------------------
    _heading(pdf, "5. Explanation: why each rule fired")
    if not fired:
        _paragraph(pdf, "No rule fired, so there is nothing to explain.", 9)
    for f in fired:
        if pdf.will_page_break(55):
            pdf.add_page()
        tags = ("  [chained]" if _is_chained(f) else "") + (f"  [covered by {f.superseded_by}]" if f.superseded_by else "")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*SEV_RGB[f.rule.severity])
        pdf.cell(0, 6, safe(f"{f.rule.rule_id} - {f.rule.diagnosis} ({f.rule.severity}){tags}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        _paragraph(pdf, f.rule.as_text(), 7.5, "", MUTED, font="Courier")
        _table(pdf, ["Condition (IF-part)", "Actual value"],
               [[c.describe(), "-" if c.fact.startswith(DX_PREFIX) else str(a)] for c, a in f.evidence], [130, 60], size=8)
        _paragraph(pdf, "Football reasoning: " + f.rule.rationale, 8.5, "I", INK)
        pdf.ln(2)

    if result.not_fired:
        _heading(pdf, "6. Rules that did not fire")
        _table(pdf, ["Rule", "Diagnosis", "Unmet condition(s)"],
               [[r.rule_id, r.diagnosis, "; ".join(reasons)] for r, reasons in result.not_fired], [13, 55, 122], size=7.5)

    # ---- method note --------------------------------------------------------------------------------------------
    _heading(pdf, "About this report")
    _paragraph(pdf, "Generated by a rule-based expert system: raw tracking metrics are abstracted into symbolic facts, "
               "then production rules are applied by forward chaining until no new conclusion can be drawn. "
               "Thresholds are expert-knowledge heuristics (not statistically fitted values) and should be tuned "
               "to the league and team being analysed.", 8.5, "", MUTED)
    return bytes(pdf.output())
