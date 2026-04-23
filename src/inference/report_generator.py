from datetime import datetime


class MedicalReportGenerator:
    """
    Hospital-grade AI Medical Report Generator
    """

    def __init__(self):

        self.class_data = {
            "No DR": {
                "severity": "No Diabetic Retinopathy",
                "risk": "Low",
                "timeline": "Annual screening recommended",
                "action": "Maintain healthy lifestyle and regular monitoring"
            },
            "Mild NPDR": {
                "severity": "Mild Non-Proliferative DR",
                "risk": "Low to Moderate",
                "timeline": "Follow-up in 6–12 months",
                "action": "Tight glycemic control and monitoring"
            },
            "Moderate NPDR": {
                "severity": "Moderate Non-Proliferative DR",
                "risk": "Moderate",
                "timeline": "Follow-up in 3–6 months",
                "action": "Ophthalmology consultation recommended"
            },
            "Severe NPDR": {
                "severity": "Severe Non-Proliferative DR",
                "risk": "High",
                "timeline": "Immediate specialist referral",
                "action": "Urgent retinal evaluation required"
            },
            "Proliferative DR": {
                "severity": "Proliferative Diabetic Retinopathy",
                "risk": "Very High",
                "timeline": "Immediate intervention required",
                "action": "High risk of vision loss — urgent treatment needed"
            }
        }

    # ---------------------------------------------------
    def generate(self, label, confidence, probabilities, image_name="Uploaded Image"):

        base_label = label.split(" (")[0]
        data = self.class_data.get(base_label, {})

        now = datetime.now().strftime("%d %b %Y, %H:%M")

        confidence_text = self._confidence_level(confidence)

        findings = self._generate_findings(base_label)
        impression = self._generate_impression(base_label)

        report = {
            "meta": {
                "report_time": now,
                "image": image_name,
                "model": "EfficientNetB3 DR Classifier"
            },
            "diagnosis": {
                "label": base_label,
                "severity": data.get("severity"),
                "confidence": f"{confidence*100:.2f}%",
                "confidence_level": confidence_text
            },
            "risk_assessment": {
                "risk_level": data.get("risk"),
                "clinical_impression": impression
            },
            "findings": findings,
            "recommendation": {
                "action": data.get("action"),
                "follow_up": data.get("timeline")
            },
            "probabilities": self._format_probs(probabilities),
            "disclaimer": "This AI system is intended for screening purposes only and must not replace clinical diagnosis."
        }

        return report

    # ---------------------------------------------------
    def _confidence_level(self, conf):
        if conf > 0.75:
            return "High"
        elif conf > 0.5:
            return "Moderate"
        return "Low"

    # ---------------------------------------------------
    def _generate_findings(self, label):

        mapping = {
            "No DR": [
                "No visible microaneurysms",
                "No hemorrhages detected",
                "Retinal vasculature appears normal"
            ],
            "Mild NPDR": [
                "Microaneurysms present",
                "Early vascular changes detected"
            ],
            "Moderate NPDR": [
                "Hemorrhages and exudates visible",
                "Moderate vascular abnormalities"
            ],
            "Severe NPDR": [
                "Extensive hemorrhages detected",
                "Significant vascular blockage"
            ],
            "Proliferative DR": [
                "Neovascularization observed",
                "High-risk proliferative changes"
            ]
        }

        return mapping.get(label, ["No findings available"])

    # ---------------------------------------------------
    def _generate_impression(self, label):

        impressions = {
            "No DR": "No diabetic retinopathy detected.",
            "Mild NPDR": "Early-stage retinal damage present.",
            "Moderate NPDR": "Disease progression observed.",
            "Severe NPDR": "Severe retinal ischemia suspected.",
            "Proliferative DR": "Advanced proliferative disease with high vision risk."
        }

        return impressions.get(label, "No impression available")

    # ---------------------------------------------------
    def _format_probs(self, probs):

        class_names = [
            "No DR",
            "Mild NPDR",
            "Moderate NPDR",
            "Severe NPDR",
            "Proliferative DR"
        ]

        return {
            cls: f"{p*100:.2f}%"
            for cls, p in zip(class_names, probs)
        }