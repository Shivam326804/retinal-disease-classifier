from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors

import tempfile
import numpy as np
import cv2
import uuid
from datetime import datetime
import qrcode
import os

from src.reports.report_generator import MedicalReportGenerator


def generate_medical_report(
    image,
    prediction,
    confidence,
    probabilities,
    class_names,
    heatmap=None
):

    generator = MedicalReportGenerator()
    report_data = generator.generate(
        label=prediction,
        confidence=confidence,
        probabilities=probabilities
    )

    summary = report_data["summary"]
    clinical = report_data["clinical"]
    recommendation = report_data["recommendation"]
    probs = report_data["probabilities"]

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        name="Header",
        fontSize=16,
        leading=20,
        textColor=colors.white
    )

    small_grey = ParagraphStyle(
        name="SmallGrey",
        fontSize=9,
        textColor=colors.grey
    )

    elements = []

    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    # ---------------------------------------------------
    # HEADER BAR (BRANDING)
    # ---------------------------------------------------
    header_table = Table(
        [["AI Retinal Screening System"]],
        colWidths=[6*inch]
    )

    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("AI-Assisted Clinical Screening Report", styles["Normal"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"<b>Report ID:</b> {report_id}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date:</b> {now}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    # ---------------------------------------------------
    # PATIENT INFO
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Patient Information</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    patient_table = Table([
        ["Patient ID", "N/A"],
        ["Age", "N/A"],
        ["Gender", "N/A"],
        ["Scan Type", "Fundus Image"]
    ], colWidths=[2.5*inch, 3*inch])

    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 16))

    # ---------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Clinical Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    summary_table = Table([
        ["Condition", summary["condition"]],
        ["Confidence", summary["confidence"]],
        ["Risk Level", summary["risk_level"]],
        ["Urgency", summary["urgency"]]
    ], colWidths=[2.5*inch, 3*inch])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    # ---------------------------------------------------
    # DIAGNOSIS + IMPRESSION
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Diagnosis</b>", styles["Heading2"]))
    elements.append(Paragraph(
        f"Predicted: <b>{summary['condition']}</b> ({summary['confidence']})",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Clinical Impression</b>", styles["Heading2"]))
    elements.append(Paragraph(clinical["impression"], styles["Normal"]))
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------
    # FINDINGS
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Findings</b>", styles["Heading2"]))
    for f in clinical["findings"]:
        elements.append(Paragraph(f"• {f}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------
    # RECOMMENDATION
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Recommendation</b>", styles["Heading2"]))
    elements.append(Paragraph(recommendation["action"], styles["Normal"]))
    elements.append(Paragraph(f"<b>Follow-up:</b> {recommendation['follow_up']}", styles["Normal"]))
    elements.append(Spacer(1, 16))

    # ---------------------------------------------------
    # PROBABILITIES
    # ---------------------------------------------------
    table_data = [["Class", "Probability"]]
    for cls, prob in probs.items():
        table_data.append([cls, prob])

    prob_table = Table(table_data, colWidths=[3*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(prob_table)
    elements.append(Spacer(1, 16))

    # ---------------------------------------------------
    # IMAGE
    # ---------------------------------------------------
    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(img_temp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elements.append(Paragraph("<b>Fundus Image</b>", styles["Heading2"]))
    elements.append(RLImage(img_temp.name, width=4*inch, height=4*inch))
    elements.append(Spacer(1, 16))

    # ---------------------------------------------------
    # GRAD-CAM
    # ---------------------------------------------------
    if heatmap is not None:
        heat_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_TURBO)
        cv2.imwrite(heat_temp.name, heatmap_colored)

        elements.append(Paragraph("<b>AI Attention Map</b>", styles["Heading2"]))
        elements.append(Paragraph("Highlighted regions influenced the prediction.", small_grey))
        elements.append(RLImage(heat_temp.name, width=4*inch, height=4*inch))
        elements.append(Spacer(1, 16))

    # ---------------------------------------------------
    # QR CODE (VERIFICATION)
    # ---------------------------------------------------
    qr = qrcode.make(f"Report ID: {report_id}")
    qr_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    qr.save(qr_path)

    elements.append(Paragraph("<b>Verification Code</b>", styles["Heading2"]))
    elements.append(RLImage(qr_path, width=1.5*inch, height=1.5*inch))
    elements.append(Spacer(1, 16))

    # ---------------------------------------------------
    # SIGNATURE BLOCK
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Authorized By</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("AI Diagnostic System", styles["Normal"]))
    elements.append(Paragraph("Validated Clinical AI Pipeline", small_grey))
    elements.append(Spacer(1, 20))

    # ---------------------------------------------------
    # FOOTER
    # ---------------------------------------------------
    elements.append(Paragraph(
        "This report is AI-generated for screening purposes only and should not replace clinical diagnosis.",
        small_grey
    ))

    doc.build(elements)

    return temp_pdf.name