from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import tempfile
import numpy as np
import cv2
import uuid


def generate_medical_report(
    image,
    prediction,
    confidence,
    probabilities,
    class_names,
    heatmap=None
):
    """
    Hospital-grade DR report (Dashboard-style PDF)
    """

    # ---------------------------------------------------
    # FILE SETUP
    # ---------------------------------------------------
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)

    styles = getSampleStyleSheet()
    elements = []

    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    # ---------------------------------------------------
    # HEADER
    # ---------------------------------------------------
    elements.append(Paragraph("<b>AI Retinal Screening Report</b>", styles["Title"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(f"<b>Report ID:</b> {report_id}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date:</b> {now}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------
    # DIAGNOSIS SECTION
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Diagnosis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(f"<b>Condition:</b> {prediction}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence*100:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------
    # PROBABILITY TABLE (CLEAN LOOK)
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Class Probability Distribution</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    table_data = [["Class", "Probability"]]

    for cls, prob in zip(class_names, probabilities):
        table_data.append([cls, f"{prob*100:.2f}%"])

    table = Table(table_data, colWidths=[3*inch, 2*inch])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 15))

    # ---------------------------------------------------
    # FINDINGS (CLINICAL STYLE)
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Findings</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    findings_map = {
        "No DR": "No visible retinal abnormalities detected.",
        "Mild NPDR": "Microaneurysms present indicating early retinal damage.",
        "Moderate NPDR": "Hemorrhages and vascular abnormalities detected.",
        "Severe NPDR": "Significant vessel blockage and retinal damage observed.",
        "Proliferative DR": "Neovascularization detected indicating advanced disease."
    }

    base_label = prediction.split(" (")[0]
    findings = findings_map.get(base_label, "No findings available.")

    elements.append(Paragraph(findings, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------
    # IMPRESSION
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Clinical Impression</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    impression_map = {
        "No DR": "No diabetic retinopathy detected.",
        "Mild NPDR": "Early stage diabetic retinopathy.",
        "Moderate NPDR": "Moderate disease progression.",
        "Severe NPDR": "Severe non-proliferative stage.",
        "Proliferative DR": "Advanced proliferative stage with high risk of vision loss."
    }

    impression = impression_map.get(base_label, "No impression available.")

    elements.append(Paragraph(impression, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------
    # RECOMMENDATION
    # ---------------------------------------------------
    elements.append(Paragraph("<b>Recommendation</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    recommendation_map = {
        "No DR": "Routine yearly screening recommended.",
        "Mild NPDR": "Monitor regularly and maintain glycemic control.",
        "Moderate NPDR": "Consult ophthalmologist within 3–6 months.",
        "Severe NPDR": "Urgent specialist consultation required.",
        "Proliferative DR": "Immediate medical intervention required."
    }

    recommendation = recommendation_map.get(base_label, "Consult specialist.")

    elements.append(Paragraph(recommendation, styles["Normal"]))
    elements.append(Spacer(1, 15))

    # ---------------------------------------------------
    # IMAGE
    # ---------------------------------------------------
    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(img_temp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elements.append(Paragraph("<b>Input Fundus Image</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(RLImage(img_temp.name, width=4*inch, height=4*inch))
    elements.append(Spacer(1, 15))

    # ---------------------------------------------------
    # HEATMAP
    # ---------------------------------------------------
    if heatmap is not None:
        heat_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_TURBO
        )

        cv2.imwrite(heat_temp.name, heatmap_colored)

        elements.append(Paragraph("<b>AI Attention Map (Grad-CAM)</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        elements.append(RLImage(heat_temp.name, width=4*inch, height=4*inch))
        elements.append(Spacer(1, 15))

    # ---------------------------------------------------
    # DISCLAIMER
    # ---------------------------------------------------
    elements.append(Paragraph(
        "<b>Disclaimer:</b> This AI system is intended for screening only and should not replace clinical diagnosis.",
        styles["Normal"]
    ))

    # ---------------------------------------------------
    # BUILD PDF
    # ---------------------------------------------------
    doc.build(elements)

    return temp_pdf.name