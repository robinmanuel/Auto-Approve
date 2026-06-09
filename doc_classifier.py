import re


class DocumentClassifier:

    def classify(self, text: str):

        text = text.lower()

        scores = {
            "RC": 0,
            "DRIVING_LICENSE": 0,
            "POLICY": 0,
            "INVOICE": 0
        }

        # -------------------------
        # RC (Registration Certificate)
        # -------------------------
        rc_keywords = [
            "registration certificate",
            "rc book",
            "registration number",
            "vehicle registration",
            "chassis number",
            "engine number",
            "rc no",
            "registration authority"
        ]

        for kw in rc_keywords:
            if kw in text:
                scores["RC"] += 2

        # -------------------------
        # Driving License
        # -------------------------
        dl_keywords = [
            "driving licence",
            "driving license",
            "dl no",
            "license number",
            "date of birth",
            "issuing authority",
            "transport department"
        ]

        for kw in dl_keywords:
            if kw in text:
                scores["DRIVING_LICENSE"] += 2

        # -------------------------
        # Insurance Policy
        # -------------------------
        policy_keywords = [
            "insurance policy",
            "policy number",
            "sum insured",
            "premium",
            "policy period",
            "insurer",
            "coverage"
        ]

        for kw in policy_keywords:
            if kw in text:
                scores["POLICY"] += 2

        # -------------------------
        # Invoice / Bill
        # -------------------------
        invoice_keywords = [
            "invoice",
            "bill",
            "total amount",
            "service charge",
            "gst",
            "garage",
            "repair estimate",
            "payable amount"
        ]

        for kw in invoice_keywords:
            if kw in text:
                scores["INVOICE"] += 2

        # pick best match
        best_doc = max(scores, key=scores.get)

        if scores[best_doc] == 0:
            return {
                "document_type": "UNKNOWN",
                "confidence": 0
            }

        return {
            "document_type": best_doc,
            "confidence": scores[best_doc] * 10,
            "all_scores": scores
        }


classifier = DocumentClassifier()