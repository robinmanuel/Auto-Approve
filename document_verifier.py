import re
import os
import pdfplumber
import pytesseract
from PIL import Image

from bert_doc_classifier import classifier   # 🔥 BERT MODEL

# IMPORTANT: Windows path fix
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class DocumentVerifier:

    # -------------------------
    # OCR EXTRACTION
    # -------------------------
    def extract_text(self, file_path):

        extension = os.path.splitext(file_path)[1].lower()
        text = ""

        if extension == ".pdf":

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        else:

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)

        return text

    # -------------------------
    # 🔥 BERT-BASED CLASSIFICATION (REPLACES KEYWORDS)
    # -------------------------
    def detect_document_type(self, text):

        result = classifier.classify(text)

        doc_type = result["document_type"]

        # fallback safety
        if not doc_type:
            return "UNKNOWN"

        return doc_type

    # -------------------------
    # FIELD EXTRACTION
    # -------------------------
    def extract_vehicle_number(self, text):

        pattern = r"[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}"

        match = re.search(pattern, text.upper())

        return match.group() if match else None

    def extract_policy_number(self, text):

        patterns = [
            r"Policy\s*No\.?\s*[:\-]?\s*([A-Z0-9\-]+)",
            r"Policy\s*Number\s*[:\-]?\s*([A-Z0-9\-]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def extract_license_number(self, text):

        patterns = [
            r"[A-Z]{2}\d{13}",
            r"[A-Z]{2}-\d{13}"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()

        return None

    def extract_amount(self, text):

        amounts = re.findall(
            r"\d+(?:,\d+)*(?:\.\d+)?",
            text
        )

        if not amounts:
            return None

        amounts = [
            float(a.replace(",", ""))
            for a in amounts
        ]

        return max(amounts)

    # -------------------------
    # FIELD MAPPING
    # -------------------------
    def extract_fields(self, document_type, text):

        fields = {}

        if document_type == "RC":
            fields["vehicle_number"] = self.extract_vehicle_number(text)

        elif document_type == "DRIVING_LICENSE":
            fields["license_number"] = self.extract_license_number(text)

        elif document_type == "POLICY":
            fields["policy_number"] = self.extract_policy_number(text)
            fields["vehicle_number"] = self.extract_vehicle_number(text)

        elif document_type == "INVOICE":
            fields["estimated_amount"] = self.extract_amount(text)

        return fields

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    def verify_document(self, file_path):

        text = self.extract_text(file_path)

        # 🔥 BERT CLASSIFICATION
        document_type = self.detect_document_type(text)

        fields = self.extract_fields(document_type, text)

        return {
            "document_type": document_type,
            "fields": fields,
            "confidence": classifier.classify(text).get("confidence", 0),
            "ocr_text": text[:2000]
        }


verifier = DocumentVerifier()


def verify_document(file_path):
    return verifier.verify_document(file_path)