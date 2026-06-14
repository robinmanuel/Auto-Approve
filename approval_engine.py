class ApprovalEngine:
    """
    Final approval decision engine that integrates:
    - Customer risk assessment from risk_engine
    - Damage analysis from pipeline
    - Document verification
    - Consistency checks
    """

    def evaluate(
        self,
        risk_assessment,
        pipeline_result,
        document_result,
        consistency_result
    ):
        """
        Args:
            risk_assessment: Full risk engine output dict (not just probability)
            pipeline_result: Damage detection results
            document_result: Document verification results
            consistency_result: Consistency check results
        """

        reasons = []
        fraud_flags = []

        # =============================================
        # EXTRACT DATA SAFELY
        # =============================================
        risk_score = risk_assessment.get("risk_score", 0)
        risk_level = risk_assessment.get("risk_level", "VERY_HIGH")
        
        parts = pipeline_result.get("parts", [])
        total_cost = float(pipeline_result.get("total_estimated_cost", 0))
        
        document_score = float(document_result.get("score", 0))
        document_missing = document_result.get("missing", [])
        
        consistency_score = float(consistency_result.get("score", 0))
        consistency_issues = consistency_result.get("issues", [])

        # =============================================
        # FRAUD DETECTION - HARD STOPS
        # =============================================
        
        # No damage detected but claim filed
        if len(parts) == 0 and total_cost > 0:
            fraud_flags.append("⚠️ Claim filed but no damage detected in image")
        
        # Document missing
        if len(document_missing) > 2:
            fraud_flags.append(f"⚠️ Critical documents missing: {', '.join(document_missing)}")
        
        # Very low consistency
        if consistency_score < 30:
            fraud_flags.append("⚠️ Major document inconsistencies detected")
        
        # Claim too large relative to vehicle
        if total_cost > 0 and pipeline_result.get("total_estimated_cost", 0) > 0:
            claim_ratio = total_cost / max(risk_assessment.get("vehicle_value", 1), 1)
            if claim_ratio > 1.0:
                fraud_flags.append("⚠️ Claim exceeds vehicle value")
        
        # If critical fraud flags → reject immediately
        if fraud_flags:
            return {
                "decision": "FRAUD REVIEW",
                "approval_score": 0,
                "risk_level": "VERY_HIGH",
                "reasons": fraud_flags,
                "next_steps": ["Manual fraud investigation required"]
            }

        # =============================================
        # AUTO APPROVAL - STRICT CRITERIA
        # =============================================
        
        auto_approve_checks = {
            "✅ Low customer risk": risk_score >= 80,
            "✅ Good documents": document_score >= 75,
            "✅ Consistent documentation": consistency_score >= 80,
            "✅ Damage confirmed": len(parts) > 0,
            "✅ Reasonable claim amount": total_cost <= 100000
        }
        
        passed_checks = sum(1 for v in auto_approve_checks.values() if v)
        
        if passed_checks == 5:  # All checks passed
            reasons = [
                f"Customer Risk Score: {risk_score}/100 (LOW)",
                f"Document Quality: {document_score}/100",
                f"Consistency Score: {consistency_score}/100",
                f"Damages verified: {len(parts)} areas detected",
                f"Estimated cost: ₹{total_cost:,.0f} (within limits)"
            ]
            
            return {
                "decision": "AUTO APPROVED ✅",
                "approval_score": round((risk_score / 100) * 100, 2),
                "risk_level": "LOW",
                "reasons": reasons,
                "next_steps": ["Proceed with payment processing"]
            }

        # =============================================
        # SURVEYOR REVIEW - HIGH COST OR MEDIUM RISK
        # =============================================
        
        surveyor_reasons = []
        
        if total_cost > 100000:
            surveyor_reasons.append(f"High repair estimate: ₹{total_cost:,.0f}")
        
        if risk_score >= 60 and risk_score < 80:
            surveyor_reasons.append(f"Medium customer risk (score: {risk_score}/100)")
        
        if document_score < 75:
            surveyor_reasons.append(f"Document quality needs verification ({document_score}/100)")
        
        if consistency_score < 80:
            surveyor_reasons.append(f"Some inconsistencies detected ({consistency_score}/100)")
        
        if surveyor_reasons:
            return {
                "decision": "SURVEYOR REVIEW 🔍",
                "approval_score": round((risk_score / 100) * 100, 2),
                "risk_level": "MEDIUM",
                "reasons": surveyor_reasons,
                "next_steps": ["Assign to surveyor for physical inspection"]
            }

        # =============================================
        # MANUAL REVIEW - EDGE CASES
        # =============================================
        
        manual_reasons = []
        
        if risk_score < 60:
            manual_reasons.append(f"High customer risk (score: {risk_score}/100)")
        
        if len(parts) == 0:
            manual_reasons.append("No damage areas detected - needs verification")
        
        if len(consistency_issues) > 0:
            manual_reasons.append(f"Document inconsistencies: {', '.join(consistency_issues[:2])}")
        
        if not manual_reasons:
            manual_reasons.append("Additional review recommended")
        
        return {
            "decision": "MANUAL REVIEW 👤",
            "approval_score": round((risk_score / 100) * 100, 2),
            "risk_level": risk_level,
            "reasons": manual_reasons,
            "next_steps": ["Route to claims specialist for review"]
        }