import os
import re
import pdfplumber
import pytesseract
from PIL import Image


# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    return text.upper().replace("\n", " ").replace("\t", " ")


# -------------------------
# OCR EXTRACTION
# -------------------------
def extract_text(file_path):

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

    return clean_text(text)


# -------------------------
# SIMPLE RULE-BASED CLASSIFIER (SAFE VERSION)
# -------------------------
def detect_document_type(text):

    if "POLICY" in text or "INSURANCE" in text:
        return "POLICY"

    if "DL" in text or "DRIVING" in text:
        return "DRIVING_LICENSE"

    if "RC" in text or "REGISTRATION" in text:
        return "RC"

    if "INVOICE" in text or "BILL" in text:
        return "INVOICE"

    return "UNKNOWN"


# -------------------------
# FIELD EXTRACTION
# -------------------------
def extract_vehicle_number(text):
    pattern = r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}"
    m = re.search(pattern, text)
    return m.group().replace(" ", "") if m else None


def extract_policy_number(text):
    pattern = r"POLICY\s*(NO|NUMBER)?\s*[:\-]?\s*([A-Z0-9\-]{6,})"
    m = re.search(pattern, text)
    return m.group(2) if m else None


def extract_license_number(text):
    pattern = r"[A-Z]{2}\d{13}"
    m = re.search(pattern, text)
    return m.group() if m else None


def extract_amount(text):
    nums = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)
    if not nums:
        return None
    nums = [float(n.replace(",", "")) for n in nums]
    return max(nums)


# -------------------------
# FIELD MAPPER
# -------------------------
def extract_fields(doc_type, text):

    fields = {}

    if doc_type == "RC":
        fields["vehicle_number"] = extract_vehicle_number(text)

    elif doc_type == "DRIVING_LICENSE":
        fields["license_number"] = extract_license_number(text)

    elif doc_type == "POLICY":
        fields["policy_number"] = extract_policy_number(text)
        fields["vehicle_number"] = extract_vehicle_number(text)

    elif doc_type == "INVOICE":
        fields["estimated_amount"] = extract_amount(text)

    return fields


# -------------------------
# MAIN FUNCTION (SAFE)
# -------------------------
def verify_document(file_path):

    text = extract_text(file_path)
    doc_type = detect_document_type(text)
    fields = extract_fields(doc_type, text)

    return {
        "document_type": doc_type,
        "confidence": 100,
        "fields": fields,
        "ocr_text": text[:2000]
    }