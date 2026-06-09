import os
import re
import platform
import pdfplumber
import pytesseract
from PIL import Image

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# TESSERACT CONFIG (Platform-aware)
# -------------------------
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
# On Linux/Streamlit Cloud, tesseract is in PATH by default


# -------------------------
# BERT DOCUMENT CLASSIFIER
# -------------------------
class BERTDocumentClassifier:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.labels = {
            "RC": """
                registration certificate rc book vehicle registration number
                chassis number engine number transport department
            """,
            "DRIVING_LICENSE": """
                driving licence driving license dl number transport authority
                date of birth driving permit license number
            """,
            "POLICY": """
                insurance policy policy number insured premium sum insured
                coverage insurer policy period
            """,
            "INVOICE": """
                invoice bill gst total amount repair estimate garage service charge
                repair cost payment invoice number
            """
        }

        self.label_embeddings = {
            k: self.model.encode(v)
            for k, v in self.labels.items()
        }

    def classify(self, text):

        if not text or len(text.strip()) < 10:
            return {"document_type": "UNKNOWN", "confidence": 0}

        text_emb = self.model.encode(text)

        scores = {}

        for label, emb in self.label_embeddings.items():

            score = cosine_similarity(
                [text_emb],
                [emb]
            )[0][0]

            scores[label] = float(score)

        best = max(scores, key=scores.get)
        confidence = scores[best] * 100

        if confidence < 35:
            return {
                "document_type": "UNKNOWN",
                "confidence": confidence,
                "scores": scores
            }

        return {
            "document_type": best,
            "confidence": confidence,
            "scores": scores
        }


classifier = BERTDocumentClassifier()


# -------------------------
# OCR + VERIFIER
# -------------------------
class DocumentVerifier:

    # CLEAN TEXT
    def clean_text(self, text):
        return (
            text.upper()
            .replace("\n", " ")
            .replace("\t", " ")
        )

    # OCR EXTRACTION
    def extract_text(self, file_path):

        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        if ext == ".pdf":

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        else:

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)

        return self.clean_text(text)

    # DOCUMENT TYPE (BERT)
    def detect_document_type(self, text):
        return classifier.classify(text)["document_type"]

    # -------------------------
    # FIELD EXTRACTION (ROBUST)
    # -------------------------
    def extract_vehicle_number(self, text):

        patterns = [
            r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}"
        ]

        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group().replace(" ", "")

        return None

    def extract_policy_number(self, text):

        patterns = [
            r"POLICY\s*(NO|NUMBER)?\s*[:\-]?\s*([A-Z0-9\-]{6,})"
        ]

        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(2)

        return None

    def extract_license_number(self, text):

        patterns = [
            r"[A-Z]{2}\d{13}",
            r"[A-Z]{2}-\d{13}"
        ]

        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group()

        return None

    def extract_amount(self, text):

        nums = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)

        if not nums:
            return None

        nums = [float(n.replace(",", "")) for n in nums]

        return max(nums)

    # FIELD MAPPER
    def extract_fields(self, doc_type, text):

        fields = {}

        if doc_type == "RC":
            fields["vehicle_number"] = self.extract_vehicle_number(text)

        elif doc_type == "DRIVING_LICENSE":
            fields["license_number"] = self.extract_license_number(text)

        elif doc_type == "POLICY":
            fields["policy_number"] = self.extract_policy_number(text)
            fields["vehicle_number"] = self.extract_vehicle_number(text)

        elif doc_type == "INVOICE":
            fields["estimated_amount"] = self.extract_amount(text)

        return fields

    # MAIN FUNCTION
    def verify_document(self, file_path):

        text = self.extract_text(file_path)

        doc_type = self.detect_document_type(text)

        fields = self.extract_fields(doc_type, text)

        return {
            "document_type": doc_type,
            "confidence": classifier.classify(text)["confidence"],
            "fields": fields,
            "ocr_text": text[:2000]
        }


verifier = DocumentVerifier()


def verify_document(file_path):
    return verifier.verify_document(file_path)