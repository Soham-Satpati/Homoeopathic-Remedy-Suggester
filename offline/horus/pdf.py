"""Treatment-plan PDF. Ported from the demo's ``build_pdf``.

Changes: renders to memory instead of a temp file (the demo never deleted its temp files,
which would leave patient PDFs on disk), and the date can be fixed for reproducible output.
Layout and wording are unchanged.
"""

from __future__ import annotations

import copy
import html as _html
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf(patient_id: str, categorised: dict, report: dict,
              patient_details: dict = None,
              report_enhanced: dict = None,
              added_symptoms: list = None,
              comparative_summary: str = None,
              generated_on: datetime | None = None,
              sources_note: str | None = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=55, leftMargin=55, topMargin=70, bottomMargin=60,
    )
    styles = getSampleStyleSheet()
    S = lambda name, **kw: ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_s  = S("T",  fontSize=18, fontName="Helvetica-Bold", textColor=colors.HexColor("#111"), spaceAfter=4)
    sub_s    = S("Su", fontSize=9,  textColor=colors.HexColor("#888"), spaceAfter=20)
    label_s  = S("Lb", fontSize=7,  fontName="Helvetica-Bold", textColor=colors.HexColor("#999"),
                 spaceBefore=14, spaceAfter=5, leading=10)
    body_s   = S("Bd", fontSize=10, textColor=colors.HexColor("#222"), leading=16, spaceAfter=8)
    rem_s    = S("RN", fontSize=12, fontName="Helvetica-Bold", textColor=colors.HexColor("#111"), spaceAfter=3)
    bullet_s = S("Bl", fontSize=10, textColor=colors.HexColor("#333"), leading=14, leftIndent=12, spaceAfter=3)
    small_s  = S("Sm", fontSize=9,  textColor=colors.HexColor("#666"), leading=13)
    foot_s   = S("Ft", fontSize=7,  textColor=colors.HexColor("#aaa"))
    h2_s     = S("H2", fontSize=13, fontName="Helvetica-Bold", textColor=colors.HexColor("#111"),
                 spaceBefore=18, spaceAfter=6)

    patient_line = f"Patient {patient_id}"
    if patient_details:
        if patient_details.get("name"):
            patient_line = f"{patient_details['name']} ({patient_id})"
        extras = []
        if patient_details.get("age"):    extras.append(f"Age {patient_details['age']}")
        if patient_details.get("gender"): extras.append(patient_details["gender"])
        if extras:
            patient_line += " · " + " · ".join(extras)

    # FIX: build elems list, pass explicitly to inner fn to avoid closure rebind bug
    elems = [
        Paragraph("HoRUS 3 — Treatment Plan", title_s),
        Paragraph(f"{patient_line} · {(generated_on or datetime.now()).strftime('%d %B %Y')}", sub_s),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd"), spaceAfter=14),
    ]

    cell_s = S("Cl", fontSize=8, textColor=colors.HexColor("#222"), leading=10)
    head_s = S("Hd", fontSize=8, fontName="Helvetica-Bold", textColor=colors.white, leading=10)
    flag_t = S("FT", fontSize=12, fontName="Helvetica-Bold", textColor=colors.HexColor("#b91c1c"), spaceAfter=4)
    flag_b = S("FB", fontSize=9.5, textColor=colors.HexColor("#7f1d1d"), leading=13, spaceAfter=3)

    def P(text, style=None):
        return Paragraph(_html.escape(str(text or "")), style or cell_s)

    def _red_flag_box(flags: list) -> list:
        if not flags:
            return []
        inner = [Paragraph("SEEK URGENT MEDICAL CARE", flag_t),
                 Paragraph("This case contains warning signs that need conventional medical assessment first. "
                           "This plan must not delay it.", flag_b)]
        inner += [Paragraph(f"<b>{_html.escape(f['title'])}.</b> {_html.escape(f['advice'])}", flag_b) for f in flags]
        box = Table([[inner]], colWidths=[6.4 * inch])
        box.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#dc2626")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return [box, Spacer(1, 12)]

    def _add_report_section(elems_ref: list, rpt: dict, cat: dict, section_label: str):
        """Inner fn receives elems_ref explicitly — no closure rebind issue."""
        elems_ref.append(Paragraph(section_label, h2_s))

        elems_ref.append(Paragraph("CASE ESSENCE", label_s))
        elems_ref.append(P(rpt.get("caseEssence", ""), body_s))
        elems_ref.append(Paragraph("MIASM (RUBRIC-BASED)", label_s))
        elems_ref.append(P(rpt.get("miasmaticAnalysis", ""), body_s))

        # Symptom table
        elems_ref.append(Paragraph("SYMPTOM SUMMARY", label_s))
        rows = [[P("Category", head_s), P("Symptom", head_s), P("Worse", head_s), P("Better", head_s)]]
        _cm = cat.get("case_modalities") or {}
        if _cm.get("worse") or _cm.get("better"):
            rows.append([P("Whole case"), P("All complaints"),
                         P(", ".join(_cm.get("worse", [])) or "—"), P(", ".join(_cm.get("better", [])) or "—")])
        for c in ("psychological", "general", "physical"):
            for s in cat.get(c, []):
                if s.get("unverified"):
                    continue
                rows.append([
                    P(c.title()), P(s["symptom"]),
                    P(", ".join(s.get("worse", [])) or "—"),
                    P(", ".join(s.get("better", [])) or "—"),
                ])
        if len(rows) > 1:
            t = Table(rows, colWidths=[1*inch, 2.4*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#111")),
                ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
                ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
                ("TOPPADDING",     (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
                ("LEFTPADDING",    (0,0), (-1,-1), 6),
                ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ]))
            elems_ref.extend([t, Spacer(1, 14)])

        # Top 10 remedies table
        top10 = rpt.get("top10Remedies", [])
        pr = rpt.get("primaryRemedy", {})
        elems_ref.append(Paragraph("TOP 10 REMEDIES", label_s))
        if top10:
            rem_rows = [[P("#", head_s), P("Remedy", head_s), P("Coverage", head_s), P("Why it ranked", head_s)]]
            for i, r in enumerate(top10, 1):
                rem_rows.append([
                    P(i),
                    P(r.get("name", "")),
                    P(r.get("role", "")),
                    P(r.get("rationale", "")),
                ])
            rt = Table(rem_rows, colWidths=[0.3*inch, 1.4*inch, 1.1*inch, 3.6*inch])
            rt.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#111")),
                ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
                ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
                ("TOPPADDING",     (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
                ("LEFTPADDING",    (0,0), (-1,-1), 5),
                ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ]))
            elems_ref.extend([rt, Spacer(1, 10)])
        else:
            # Fallback: primary + secondary
            elems_ref.extend([
                Paragraph("SIMILLIMUM", label_s),
                Paragraph(pr.get("name", ""), rem_s),
                Paragraph(pr.get("why", ""), body_s),
            ])

        if pr.get("keyIndications"):
            elems_ref.append(Paragraph("Dataset matches for the top remedy:", small_s))
            for ind in pr["keyIndications"]:
                elems_ref.append(P(f"• {ind}", bullet_s))
        if pr.get("followedBy"):
            elems_ref.append(Paragraph(f"Followed well by: {', '.join(pr['followedBy'])}", small_s))
        elems_ref.append(Spacer(1, 10))

        if rpt.get("remedyRelationships"):
            elems_ref.extend([Paragraph("HOW THE TOP REMEDIES DIFFER", label_s), P(rpt["remedyRelationships"], body_s)])
        mon = rpt.get("monitoringPoints", [])
        if mon:
            elems_ref.append(Paragraph("MONITORING POINTS", label_s))
            for pt in mon:
                elems_ref.append(P(f"• {pt}", bullet_s))
        elems_ref.append(Paragraph(
            "Remedies, order, scores and miasm were computed from the HoRUS dataset and rubric; "
            "explanations are AI-written and filtered.", small_s))

    # Base report
    _add_report_section(elems, report, categorised, "A. Original Symptom Analysis")

    # Enhanced report (if present)
    if report_enhanced and added_symptoms:
        elems.extend([Spacer(1, 20), HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#ddd"), spaceAfter=14)])

        elems.append(Paragraph("B. AI-Suggested Additional Symptoms", h2_s))
        elems.append(Paragraph("Symptoms suggested by cluster model, added by clinician:", small_s))
        for s in added_symptoms:
            worse  = ", ".join(s.get("worse",  [])) or "—"
            better = ", ".join(s.get("better", [])) or "—"
            prob   = s.get("avg_prob", 0)
            trigger = s.get("triggered_by", "")
            trigger_str = f" ← {trigger}" if trigger else ""
            elems.append(Paragraph(
                f"• {s['symptom']}  [P={prob:.0%}]{trigger_str} | ↓ {worse} | ↑ {better}",
                bullet_s
            ))
        elems.append(Spacer(1, 10))

        enhanced_cat = copy.deepcopy(categorised)
        for s in added_symptoms:
            enhanced_cat["physical"].append({
                "symptom": s["symptom"],
                "worse": s.get("worse", []),
                "better": s.get("better", []),
            })

        _add_report_section(elems, report_enhanced, enhanced_cat, "C. Enhanced Analysis (with suggested symptoms)")

        if comparative_summary:
            elems.extend([
                Spacer(1, 14),
                HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd"), spaceAfter=10),
                Paragraph("D. Comparative Clinical Summary", h2_s),
                Paragraph(comparative_summary, body_s),
            ])

    if sources_note:
        elems.extend([Spacer(1, 10), Paragraph("SOURCES AND WEIGHTING", label_s), P(sources_note, small_s)])
    elems.extend([
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#ddd"), spaceAfter=6),
        Paragraph("For clinical reference only. Prescribing decisions rest with the practitioner.", foot_s),
    ])
    doc.build(elems)
    return buf.getvalue()
