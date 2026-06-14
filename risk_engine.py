# risk_engine.py

class CustomerRiskEngine:

    def __init__(self):
        pass

    # -----------------------------
    # MAIN SCORING
    # -----------------------------
    def evaluate(
        self,
        customer_age,
        driving_experience,
        vehicle_age,
        vehicle_value,
        previous_claims,
        annual_premium,
        claim_amount,
        document_score,
        consistency_score,
        damage_count,
        severe_damage_count
    ):

        score = 100
        reasons = []
        factor_breakdown = {}  # Track each factor's impact

        # ==================================
        # DRIVER AGE
        # ==================================
        age_impact = 0
        age_reason = f"Driver age: {customer_age} years"

        if customer_age < 21:
            age_impact = -25
            reasons.append("Very young driver (< 21)")

        elif customer_age < 25:
            age_impact = -15
            reasons.append("Young driver (< 25)")

        elif customer_age > 75:
            age_impact = -10
            reasons.append("Senior driver (> 75)")
        else:
            age_reason += " (optimal range)"

        score += age_impact
        factor_breakdown["Driver Age"] = {
            "value": customer_age,
            "impact": age_impact,
            "explanation": age_reason
        }

        # ==================================
        # EXPERIENCE
        # ==================================
        exp_impact = 0
        exp_reason = f"Driving experience: {driving_experience} years"

        if driving_experience < 2:
            exp_impact = -20
            reasons.append("Very low driving experience (< 2 years)")

        elif driving_experience < 5:
            exp_impact = -10
            reasons.append("Low driving experience (< 5 years)")

        elif driving_experience > 15:
            exp_impact = 5
            reasons.append("Highly experienced driver (> 15 years)")
        else:
            exp_reason += " (moderate experience)"

        score += exp_impact
        factor_breakdown["Driving Experience"] = {
            "value": driving_experience,
            "impact": exp_impact,
            "explanation": exp_reason
        }

        # ==================================
        # CLAIM HISTORY
        # ==================================
        claim_impact = 0
        claim_reason = f"Previous claims: {previous_claims}"

        if previous_claims >= 5:
            claim_impact = -35
            reasons.append("Frequent claimant (≥ 5 claims)")

        elif previous_claims >= 3:
            claim_impact = -20
            reasons.append("Multiple claims (≥ 3 claims)")

        elif previous_claims == 0:
            claim_impact = 10
            reasons.append("Clean claim history (0 claims)")
        else:
            claim_reason += " (acceptable history)"

        score += claim_impact
        factor_breakdown["Claim History"] = {
            "value": previous_claims,
            "impact": claim_impact,
            "explanation": claim_reason
        }

        # ==================================
        # VEHICLE AGE
        # ==================================
        vehicle_age_impact = 0
        vehicle_age_reason = f"Vehicle age: {vehicle_age} years"

        if vehicle_age > 15:
            vehicle_age_impact = -10
            reasons.append("Old vehicle (> 15 years)")

        elif vehicle_age > 10:
            vehicle_age_impact = -5
            reasons.append("Aging vehicle (> 10 years)")
        else:
            vehicle_age_reason += " (relatively new)"

        score += vehicle_age_impact
        factor_breakdown["Vehicle Age"] = {
            "value": vehicle_age,
            "impact": vehicle_age_impact,
            "explanation": vehicle_age_reason
        }

        # ==================================
        # PREMIUM
        # ==================================
        premium_impact = 0
        premium_reason = f"Annual premium: ₹{annual_premium:,.0f}"

        if annual_premium > 50000:
            premium_impact = 5
            reasons.append("High premium (> ₹50,000)")

        elif annual_premium < 10000:
            premium_impact = -5
            reasons.append("Very low premium (< ₹10,000)")
        else:
            premium_reason += " (reasonable premium)"

        score += premium_impact
        factor_breakdown["Annual Premium"] = {
            "value": annual_premium,
            "impact": premium_impact,
            "explanation": premium_reason
        }

        # ==================================
        # CLAIM VS VEHICLE VALUE
        # ==================================
        claim_value_impact = 0
        claim_value_reason = f"Vehicle value: ₹{vehicle_value:,.0f}, Claim: ₹{claim_amount:,.0f}"

        if vehicle_value > 0:
            ratio = claim_amount / vehicle_value
            claim_value_reason = f"Claim-to-value ratio: {ratio:.2%}"

            if ratio > 0.8:
                claim_value_impact = -40
                reasons.append("Claim close to total loss (> 80%)")

            elif ratio > 0.5:
                claim_value_impact = -25
                reasons.append("Substantial claim (> 50% of vehicle value)")

            elif ratio > 0.3:
                claim_value_impact = -10
                reasons.append("Significant claim (> 30% of vehicle value)")
            else:
                claim_value_reason += " (reasonable claim amount)"

        score += claim_value_impact
        factor_breakdown["Claim vs Vehicle Value"] = {
            "value": f"{claim_amount:,.0f} / {vehicle_value:,.0f}",
            "impact": claim_value_impact,
            "explanation": claim_value_reason
        }

        # ==================================
        # DAMAGE ANALYSIS
        # ==================================
        damage_impact = 0
        damage_reason = f"Damage count: {damage_count}, Severe: {severe_damage_count}"

        if damage_count > 10:
            damage_impact -= 20
            reasons.append("Extensive damage (> 10 areas)")

        elif damage_count > 5:
            damage_impact -= 10
            reasons.append("Multiple damage areas (> 5)")
        else:
            damage_reason += " (damage extent acceptable)"

        if severe_damage_count >= 3:
            damage_impact -= 20
            reasons.append("Multiple severe damages (≥ 3)")

        elif severe_damage_count >= 1:
            damage_impact -= 10
            reasons.append("Severe damage detected")
        else:
            damage_reason += " (no severe damage)"

        score += damage_impact
        factor_breakdown["Damage Analysis"] = {
            "value": f"{damage_count} total, {severe_damage_count} severe",
            "impact": damage_impact,
            "explanation": damage_reason
        }

        # ==================================
        # DOCUMENT SCORE
        # ==================================
        doc_impact = 0
        doc_reason = f"Document quality score: {document_score}/100"

        if document_score < 50:
            doc_impact = -35
            reasons.append("Poor document quality (< 50)")

        elif document_score < 70:
            doc_impact = -15
            reasons.append("Fair document quality (< 70)")

        elif document_score > 90:
            doc_impact = 5
            reasons.append("Excellent document quality (> 90)")
        else:
            doc_reason += " (acceptable quality)"

        score += doc_impact
        factor_breakdown["Document Quality"] = {
            "value": document_score,
            "impact": doc_impact,
            "explanation": doc_reason
        }

        # ==================================
        # CONSISTENCY
        # ==================================
        consistency_impact = 0
        consistency_reason = f"Consistency score: {consistency_score}/100"

        if consistency_score < 50:
            consistency_impact = -30
            reasons.append("Major document inconsistencies (< 50)")

        elif consistency_score < 70:
            consistency_impact = -15
            reasons.append("Minor inconsistencies detected (< 70)")

        elif consistency_score > 90:
            consistency_impact = 5
            reasons.append("Excellent document consistency (> 90)")
        else:
            consistency_reason += " (good consistency)"

        score += consistency_impact
        factor_breakdown["Document Consistency"] = {
            "value": consistency_score,
            "impact": consistency_impact,
            "explanation": consistency_reason
        }

        # ==================================
        # NORMALIZE
        # ==================================

        score = max(0, min(100, score))

        approval_probability = round(
            score / 100,
            2
        )

        # ==================================
        # RISK LEVEL
        # ==================================

        if score >= 80:
            risk_level = "LOW"

        elif score >= 60:
            risk_level = "MEDIUM"

        elif score >= 40:
            risk_level = "HIGH"

        else:
            risk_level = "VERY_HIGH"

        return {
            "risk_score": score,
            "approval_probability": approval_probability,
            "risk_level": risk_level,
            "reasons": reasons,
            "factor_breakdown": factor_breakdown  # Detailed breakdown for explainability
        }


risk_engine = CustomerRiskEngine()