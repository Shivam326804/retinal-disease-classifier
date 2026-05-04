from datetime import datetime


class MedicalReportGenerator:
    """
    Enhanced Clinical Intelligence Layer for AI Medical Report
    """

    def __init__(self):

        self.class_data = {
            "No DR": {
                "severity": "No Diabetic Retinopathy",
                "risk": "Low",
                "urgency": "Routine",
                "timeline": "Annual screening recommended",
                "action": "Maintain healthy lifestyle and regular monitoring"
            },
            "Mild NPDR": {
                "severity": "Mild Non-Proliferative DR",
                "risk": "Low to Moderate",
                "urgency": "Low",
                "timeline": "Follow-up in 6–12 months",
                "action": "Tight glycemic control and periodic monitoring"
            },
            "Moderate NPDR": {
                "severity": "Moderate Non-Proliferative DR",
                "risk": "Moderate",
                "urgency": "Medium",
                "timeline": "Follow-up in 3–6 months",
                "action": "Ophthalmology consultation recommended"
            },
            "Severe NPDR": {
                "severity": "Severe Non-Proliferative DR",
                "risk": "High",
                "urgency": "High",
                "timeline": "Immediate specialist referral",
                "action": "Urgent retinal evaluation required"
            },
            "Proliferative DR": {
                "severity": "Proliferative Diabetic Retinopathy",
                "risk": "Very High",
                "urgency": "Critical",
                "timeline": "Immediate intervention required",
                "action": "High risk of vision loss — urgent treatment required"
            }
        }

    # ---------------------------------------------------
    def generate(self, label, confidence, probabilities, image_name="Uploaded Image"):

        base_label = label.split(" (")[0]
        data = self.class_data.get(base_label, {})

        now = datetime.now().strftime("%d %b %Y, %H:%M")

        confidence_level = self._confidence_level(confidence)

        report = {
            "meta": {
                "report_time": now,
                "image": image_name,
                "model": "EfficientNetB3 DR Classifier"
            },

            # 🔥 NEW: Structured summary
            "summary": {
                "condition": base_label,
                "severity": data.get("severity"),
                "confidence": f"{confidence*100:.2f}%",
                "confidence_level": confidence_level,
                "risk_level": data.get("risk"),
                "urgency": data.get("urgency")
            },

            "diagnosis": {
                "label": base_label,
                "severity": data.get("severity")
            },

            "clinical": {
                "impression": self._generate_impression(base_label),
                "findings": self._generate_findings(base_label)
            },

            "recommendation": {
                "action": self._generate_action(base_label),
                "follow_up": data.get("timeline")
            },

            "probabilities": self._format_probs(probabilities),

            "disclaimer": (
                "This AI-generated report is intended for screening support only "
                "and should not be used as a sole basis for clinical decision-making. "
                "Consult a qualified ophthalmologist."
            )
        }

        return report

    # ---------------------------------------------------
    def _confidence_level(self, conf):
        if conf > 0.8:
            return "High"
        elif conf > 0.6:
            return "Moderate"
        return "Low"

    # ---------------------------------------------------
    def _generate_findings(self, label):

        mapping = {
            "No DR": [
                "No microaneurysms detected",
                "No retinal hemorrhages",
                "Normal retinal vasculature"
            ],
            "Mild NPDR": [
                "Microaneurysms present",
                "Early vascular changes observed"
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
            "No DR": "No evidence of diabetic retinopathy.",
            "Mild NPDR": "Early-stage retinal damage consistent with mild NPDR.",
            "Moderate NPDR": "Moderate disease progression with vascular compromise.",
            "Severe NPDR": "Severe retinal ischemia and vascular damage.",
            "Proliferative DR": "Advanced proliferative disease with high risk of vision loss."
        }

        return impressions.get(label, "No clinical impression available.")

    # ---------------------------------------------------
    def _generate_action(self, label):

        actions = {
            "No DR": "Continue routine screening and preventive care.",
            "Mild NPDR": "Maintain glycemic control and monitor regularly.",
            "Moderate NPDR": "Consult ophthalmologist within 3–6 months.",
            "Severe NPDR": "Urgent specialist referral recommended.",
            "Proliferative DR": "Immediate ophthalmologic intervention required."
        }

        return actions.get(label, "Consult specialist.")

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