class ApprovalEngine:

    def evaluate(
        self,
        customer_probability,
        pipeline_result,
        document_result,
        consistency_result
    ):

        reasons = []
        fraud_flags = []

        # -----------------------------
        # SAFE EXTRACTION
        # -----------------------------
        parts = pipeline_result.get("parts") or []
        total_cost = float(pipeline_result.get("total_estimated_cost") or 0)

        # If no parts detected → immediate fraud flag
        if len(parts) == 0:
            return {
                "decision": "FRAUD REVIEW",
                "approval_score": 0,
                "risk_level": "HIGH",
                "reasons": ["No damage parts detected"]
            }

        damage_confidences = []
        damage_types = []

        # -----------------------------
        # SAFE LOOP (IMPORTANT FIX)
        # -----------------------------
        for p in parts:
            damage = p.get("damage", {})

            conf = float(damage.get("Confidence") or 0)
            dtype = damage.get("Damage_Type") or "unknown"

            damage_confidences.append(conf)
            damage_types.append(dtype)

        max_damage_confidence = max(damage_confidences) if damage_confidences else 0

        # -----------------------------
        # DOCUMENT / CONSISTENCY SAFETY
        # -----------------------------
        document_score = float(document_result.get("score") or 0)
        consistency_score = float(consistency_result.get("score") or 0)
        issues = consistency_result.get("issues") or []

        # -----------------------------
        # FRAUD DETECTION
        # -----------------------------

        if "no_damage" in damage_types:
            fraud_flags.append("No damage detected but claim raised")

        if consistency_score < 50:
            fraud_flags.append("Major document mismatch")

        if len(issues) >= 2:
            fraud_flags.append("Multiple inconsistencies found")

        if total_cost <= 0:
            fraud_flags.append("Invalid or zero repair estimate")

        # If fraud found → stop immediately
        if fraud_flags:
            return {
                "decision": "FRAUD REVIEW",
                "approval_score": 0,
                "risk_level": "HIGH",
                "reasons": fraud_flags
            }

        # -----------------------------
        # AUTO APPROVAL LOGIC
        # -----------------------------
        if (
            customer_probability >= 0.85
            and document_score >= 80
            and consistency_score >= 80
            and max_damage_confidence >= 80
            and total_cost <= 50000
        ):

            reasons = [
                "Low customer risk",
                "Documents verified",
                "Damage detected with high confidence",
                "Consistency checks passed",
                "Repair cost within threshold"
            ]

            return {
                "decision": "AUTO APPROVED",
                "approval_score": round(customer_probability * 100, 2),
                "risk_level": "LOW",
                "reasons": reasons
            }

        # -----------------------------
        # SURVEYOR REVIEW
        # -----------------------------
        if total_cost > 50000:

            return {
                "decision": "SURVEYOR REVIEW",
                "approval_score": round(customer_probability * 100, 2),
                "risk_level": "MEDIUM",
                "reasons": ["High repair estimate"]
            }

        # -----------------------------
        # MANUAL REVIEW
        # -----------------------------
        reasons = issues if issues else ["Additional validation required"]

        return {
            "decision": "MANUAL REVIEW",
            "approval_score": round(customer_probability * 100, 2),
            "risk_level": "MEDIUM",
            "reasons": reasons
        }