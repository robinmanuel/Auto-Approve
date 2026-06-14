import os
import re
import cv2
import json
import hashlib
import platform
import pdfplumber
import numpy as np

from PIL import Image, ExifTags
from datetime import datetime

# HEIC Support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except:
    HEIC_AVAILABLE = False

# OCR
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False

# NLP Classification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# TESSERACT CONFIG
# =====================================================

if TESSERACT_AVAILABLE and platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# =====================================================
# DOCUMENT CLASSIFIER
# =====================================================

class BERTDocumentClassifier:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.labels = {
            "RC": """
            REGISTRATION CERTIFICATE
            RC BOOK
            VEHICLE REGISTRATION

            REGISTRATION NUMBER
            REGN NO
            REGISTRATION NO

            CHASSIS NUMBER
            CHASSIS NO

            ENGINE NUMBER
            ENGINE NO

            OWNER NAME

            VEHICLE CLASS
            VEHICLE TYPE

            FUEL TYPE

            MANUFACTURER
            MAKER

            MODEL

            TRANSPORT DEPARTMENT
            MOTOR VEHICLE
            """,

            "DRIVING_LICENSE": """
            DRIVING LICENCE
            DRIVING LICENSE

            LICENCE
            LICENSE

            LICENCE NO
            LICENSE NO

            DL NO
            DL NUMBER

            DATE OF BIRTH
            DOB

            VALID UPTO
            VALID TILL

            ISSUE DATE
            EXPIRY DATE

            ISSUING AUTHORITY

            TRANSPORT AUTHORITY

            LMV
            MCWG
            MCWOG

            DRIVER
            DRIVING PERMIT
            """,

            "POLICY": """
            INSURANCE POLICY

            POLICY NUMBER
            POLICY NO

            INSURED

            INSURER

            PREMIUM

            SUM INSURED

            NCB

            NO CLAIM BONUS

            POLICY PERIOD

            POLICY EXPIRY

            CLAIM HISTORY
            """,

            "INVOICE": """
            TAX INVOICE

            REPAIR BILL

            GARAGE BILL

            INVOICE NUMBER

            GST

            TOTAL AMOUNT

            GRAND TOTAL

            NET AMOUNT

            SERVICE CHARGE

            LABOUR CHARGE

            REPAIR COST
            """,

            "AADHAAR": """
            AADHAAR

            UIDAI

            GOVERNMENT OF INDIA

            YEAR OF BIRTH

            DOB

            MALE

            FEMALE

            12 DIGIT

            IDENTIFICATION
            """,

            "PAN": """
            PAN CARD

            PERMANENT ACCOUNT NUMBER

            INCOME TAX

            GOVT OF INDIA

            ACCOUNT NUMBER
            """,

            "PUC": """
            POLLUTION UNDER CONTROL

            POLLUTION CERTIFICATE

            PUC

            EMISSION

            VALIDITY
            """
        }

        self.label_embeddings = {
            label: self.model.encode(text)
            for label, text in self.labels.items()
        }

    def classify(self, text):

        if not text or len(text.strip()) < 10:

            return {
                "document_type": "UNKNOWN",
                "confidence": 0,
                "scores": {}
            }

        text_upper = text.upper()

        # RC
        if (
            "REGISTRATION CERTIFICATE" in text_upper
            or "RC BOOK" in text_upper
        ):
            return {
                "document_type": "RC",
                "confidence": 99
            }

        # DL
        if (
            "DRIVING LICENCE" in text_upper
            or "DRIVING LICENSE" in text_upper
            or "DL NO" in text_upper
            or "LICENCE NO" in text_upper
        ):
            return {
                "document_type": "DRIVING_LICENSE",
                "confidence": 99
            }

        # POLICY
        if (
            "POLICY NUMBER" in text_upper
            or "INSURANCE POLICY" in text_upper
        ):
            return {
                "document_type": "POLICY",
                "confidence": 99
            }

        # INVOICE
        if (
            "TAX INVOICE" in text_upper
            or "INVOICE NUMBER" in text_upper
            or "GSTIN" in text_upper
        ):
            return {
                "document_type": "INVOICE",
                "confidence": 99
            }

        # AADHAAR
        if (
            "UIDAI" in text_upper
            or "AADHAAR" in text_upper
        ):
            return {
                "document_type": "AADHAAR",
                "confidence": 99
            }

        # PAN
        if (
            "PERMANENT ACCOUNT NUMBER" in text_upper
            or "INCOME TAX DEPARTMENT" in text_upper
        ):
            return {
                "document_type": "PAN",
                "confidence": 99
            }

        # PUC
        if (
            "POLLUTION UNDER CONTROL" in text_upper
            or "PUC" in text_upper
        ):
            return {
                "document_type": "PUC",
                "confidence": 99
            }

        text_embedding = self.model.encode(text)

        scores = {}

        for label, emb in self.label_embeddings.items():

            score = cosine_similarity(
                [text_embedding],
                [emb]
            )[0][0]

            scores[label] = float(score)

        best_label = max(scores, key=scores.get)

        confidence = float(scores[best_label] * 100)

        if confidence < 35:

            return {
                "document_type": "UNKNOWN",
                "confidence": confidence,
                "scores": scores
            }

        return {
            "document_type": best_label,
            "confidence": confidence,
            "scores": scores
        }


# Lazy singleton
_classifier = None

def get_classifier():

    global _classifier

    if _classifier is None:
        _classifier = BERTDocumentClassifier()

    return _classifier


# =====================================================
# FILE HASHING
# =====================================================

def calculate_file_hash(file_path):

    sha = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# =====================================================
# IMAGE QUALITY ANALYSIS
# =====================================================

class ImageQualityAnalyzer:

    @staticmethod
    def blur_score(file_path):

        try:

            img = cv2.imread(file_path)

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            return float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()
            )

        except:
            return 0

    @staticmethod
    def brightness_score(file_path):

        try:

            img = cv2.imread(file_path)

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            return float(np.mean(gray))

        except:
            return 0

    @staticmethod
    def quality_status(
        blur_score,
        brightness_score
    ):

        status = []

        if blur_score < 50:
            status.append("VERY_BLURRY")

        elif blur_score < 100:
            status.append("BLURRY")

        if brightness_score < 50:
            status.append("TOO_DARK")

        elif brightness_score > 220:
            status.append("OVER_EXPOSED")

        if not status:
            status.append("GOOD")

        return status


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

class ImagePreprocessor:

    @staticmethod
    def preprocess_image(file_path):

        image = cv2.imread(file_path)

        if image is None:
            return None

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2
        )

        return thresh


# =====================================================
# OCR ENGINE
# =====================================================

class OCREngine:

    @staticmethod
    def clean_text(text):

        if not text:
            return ""

        return (
            text.upper()
            .replace("\n", " ")
            .replace("\t", " ")
            .strip()
        )

    @staticmethod
    def extract_text_from_image(file_path):

        if not TESSERACT_AVAILABLE:

            return {
                "text": "",
                "confidence": 0
            }

        image = Image.open(file_path)

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT
        )

        words = []
        confidences = []

        for txt, conf in zip(
            data["text"],
            data["conf"]
        ):

            txt = txt.strip()

            if not txt:
                continue

            words.append(txt)

            try:

                conf = float(conf)

                if conf >= 0:
                    confidences.append(conf)

            except:
                pass

        text = " ".join(words)

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0
        )

        return {
            "text": OCREngine.clean_text(text),
            "confidence": round(
                avg_confidence,
                2
            )
        }

    @staticmethod
    def extract_text_from_pdf(file_path):

        all_text = []

        try:

            with pdfplumber.open(file_path) as pdf:

                for page in pdf.pages:

                    txt = page.extract_text()

                    if txt:
                        all_text.append(txt)

        except:
            pass

        return OCREngine.clean_text(
            "\n".join(all_text)
        )
    
# =====================================================
# FIELD EXTRACTION ENGINE
# =====================================================

class FieldExtractor:

    # -------------------------------------------
    # COMMON HELPERS
    # -------------------------------------------

    @staticmethod
    def normalize(value):

        if not value:
            return None

        value = value.strip()

        return value if value else None

    @staticmethod
    def extract_dates(text):

        patterns = [

            r"\d{2}[/-]\d{2}[/-]\d{4}",
            r"\d{2}[.-]\d{2}[.-]\d{4}",
            r"\d{4}[/-]\d{2}[/-]\d{2}"
        ]

        dates = []

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text
            )

            dates.extend(matches)

        return list(set(dates))

    # -------------------------------------------
    # VEHICLE NUMBER
    # -------------------------------------------

    @staticmethod
    def extract_vehicle_number(text):

        patterns = [

            r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}",
            r"[A-Z]{2}-\d{1,2}-[A-Z]{1,3}-\d{4}"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:

                return (
                    match.group()
                    .replace(" ", "")
                    .replace("-", "")
                )

        return None

    # -------------------------------------------
    # CHASSIS NUMBER
    # -------------------------------------------

    @staticmethod
    def extract_chassis_number(text):

        patterns = [

            r"CHASSIS\s*(?:NO|NUMBER)?\s*[:\-]?\s*([A-Z0-9]{8,25})",

            r"CHASSIS\s*NUMBER\s*[:\-]?\s*([A-Z0-9]{8,25})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:
                return match.group(1)

        return None

    # -------------------------------------------
    # ENGINE NUMBER
    # -------------------------------------------

    @staticmethod
    def extract_engine_number(text):

        patterns = [

            r"ENGINE\s*(?:NO|NUMBER)?\s*[:\-]?\s*([A-Z0-9]{5,25})",

            r"ENGINE\s*NUMBER\s*[:\-]?\s*([A-Z0-9]{5,25})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:
                return match.group(1)

        return None

    # -------------------------------------------
    # OWNER NAME
    # -------------------------------------------

    @staticmethod
    def extract_owner_name(text):

        patterns = [

            r"OWNER\s*NAME\s*[:\-]?\s*([A-Z ]{3,60})",

            r"REGISTERED\s*OWNER\s*[:\-]?\s*([A-Z ]{3,60})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:

                return (
                    match.group(1)
                    .strip()
                )

        return None

    # -------------------------------------------
    # VEHICLE CLASS
    # -------------------------------------------

    @staticmethod
    def extract_vehicle_class(text):

        classes = [

            "MOTOR CAR",
            "LMV",
            "MCWG",
            "MCWOG",
            "TRANSPORT",
            "PRIVATE"
        ]

        for item in classes:

            if item in text:
                return item

        return None

    # -------------------------------------------
    # FUEL TYPE
    # -------------------------------------------

    @staticmethod
    def extract_fuel_type(text):

        fuels = [

            "PETROL",
            "DIESEL",
            "CNG",
            "LPG",
            "ELECTRIC",
            "HYBRID"
        ]

        for fuel in fuels:

            if fuel in text:
                return fuel

        return None

    # -------------------------------------------
    # MANUFACTURER
    # -------------------------------------------

    @staticmethod
    def extract_manufacturer(text):

        brands = [

            "MARUTI",
            "HYUNDAI",
            "HONDA",
            "TATA",
            "MAHINDRA",
            "TOYOTA",
            "KIA",
            "RENAULT",
            "FORD",
            "BMW",
            "AUDI",
            "MERCEDES"
        ]

        for brand in brands:

            if brand in text:
                return brand

        return None

    # -------------------------------------------
    # MODEL
    # -------------------------------------------

    @staticmethod
    def extract_model(text):

        patterns = [

            r"MODEL\s*[:\-]?\s*([A-Z0-9 ]{2,40})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:
                return match.group(1).strip()

        return None

    # -------------------------------------------
    # LICENSE NUMBER
    # -------------------------------------------

    @staticmethod
    def extract_license_number(text):

        patterns = [

            r"[A-Z]{2}\d{13}",
            r"[A-Z]{2}-\d{13}",
            r"DL\s*NO\s*[:\-]?\s*([A-Z0-9\-]+)"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:

                if match.groups():
                    return match.group(1)

                return match.group()

        return None

    # -------------------------------------------
    # NAME
    # -------------------------------------------

    @staticmethod
    def extract_name(text):

        patterns = [

            r"NAME\s*[:\-]?\s*([A-Z ]{3,60})",

            r"HOLDER\s*NAME\s*[:\-]?\s*([A-Z ]{3,60})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:
                return match.group(1).strip()

        return None

    # -------------------------------------------
    # DOB
    # -------------------------------------------

    @staticmethod
    def extract_dob(text):

        patterns = [

            r"DOB\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",

            r"DATE OF BIRTH\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:
                return match.group(1)

        return None

    # -------------------------------------------
    # POLICY NUMBER
    # -------------------------------------------

    @staticmethod
    def extract_policy_number(text):

        patterns = [

            r"POLICY\s*(?:NO|NUMBER)?\s*[:\-]?\s*([A-Z0-9\-]{6,40})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text
            )

            if match:
                return match.group(1)

        return None

    # -------------------------------------------
    # INSURER
    # -------------------------------------------

    @staticmethod
    def extract_insurer(text):

        insurers = [

            "ICICI",
            "HDFC",
            "BAJAJ",
            "IFFCO",
            "NEW INDIA",
            "ORIENTAL",
            "NATIONAL",
            "RELIANCE",
            "TATA AIG",
            "ACKO"
        ]

        for insurer in insurers:

            if insurer in text:
                return insurer

        return None

    # -------------------------------------------
    # NCB %
    # -------------------------------------------

    @staticmethod
    def extract_ncb(text):

        match = re.search(
            r"(\d{1,2})\s*%",
            text
        )

        if match:

            value = int(
                match.group(1)
            )

            if value <= 100:
                return value

        return None

    # -------------------------------------------
    # CLAIM HISTORY
    # -------------------------------------------

    @staticmethod
    def detect_claim_history(text):

        indicators = [

            "CLAIM",
            "CLAIMED",
            "ACCIDENT",
            "SETTLED CLAIM"
        ]

        for indicator in indicators:

            if indicator in text:
                return True

        return False

    # -------------------------------------------
    # AADHAAR
    # -------------------------------------------

    @staticmethod
    def extract_aadhaar(text):

        match = re.search(
            r"\d{4}\s?\d{4}\s?\d{4}",
            text
        )

        if match:
            return match.group()

        return None

    # -------------------------------------------
    # PAN
    # -------------------------------------------

    @staticmethod
    def extract_pan(text):

        match = re.search(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            text
        )

        if match:
            return match.group()

        return None

    # -------------------------------------------
    # RC DOCUMENT
    # -------------------------------------------

    @classmethod
    def extract_rc_fields(
        cls,
        text
    ):

        return {

            "vehicle_number":
                cls.extract_vehicle_number(text),

            "chassis_number":
                cls.extract_chassis_number(text),

            "engine_number":
                cls.extract_engine_number(text),

            "owner_name":
                cls.extract_owner_name(text),

            "registration_date":
                cls.extract_dates(text),

            "vehicle_class":
                cls.extract_vehicle_class(text),

            "fuel_type":
                cls.extract_fuel_type(text),

            "manufacturer":
                cls.extract_manufacturer(text),

            "model":
                cls.extract_model(text)
        }

    # -------------------------------------------
    # DL DOCUMENT
    # -------------------------------------------

    @classmethod
    def extract_dl_fields(
        cls,
        text
    ):

        dates = cls.extract_dates(text)

        return {

            "license_number":
                cls.extract_license_number(text),

            "name":
                cls.extract_name(text),

            "dob":
                cls.extract_dob(text),

            "expiry_date":
                dates[-1] if dates else None,

            "issuing_authority":
                "TRANSPORT AUTHORITY"
                if "TRANSPORT" in text
                else None
        }

    # -------------------------------------------
    # POLICY DOCUMENT
    # -------------------------------------------

    @classmethod
    def extract_policy_fields(
        cls,
        text
    ):

        dates = cls.extract_dates(text)

        return {

            "policy_number":
                cls.extract_policy_number(text),

            "vehicle_number":
                cls.extract_vehicle_number(text),

            "insurer_name":
                cls.extract_insurer(text),

            "expiry_date":
                dates[-1] if dates else None,

            "ncb_percentage":
                cls.extract_ncb(text),

            "claim_history":
                cls.detect_claim_history(text)
        }

    # -------------------------------------------
    # AADHAAR
    # -------------------------------------------

    @classmethod
    def extract_aadhaar_fields(
        cls,
        text
    ):

        return {

            "aadhaar_number":
                cls.extract_aadhaar(text),

            "name":
                cls.extract_name(text),

            "dob":
                cls.extract_dob(text)
        }

    # -------------------------------------------
    # PAN
    # -------------------------------------------

    @classmethod
    def extract_pan_fields(
        cls,
        text
    ):

        return {

            "pan_number":
                cls.extract_pan(text),

            "name":
                cls.extract_name(text)
        }

    # -------------------------------------------
    # PUC
    # -------------------------------------------

    @classmethod
    def extract_puc_fields(
        cls,
        text
    ):

        dates = cls.extract_dates(text)

        return {

            "vehicle_number":
                cls.extract_vehicle_number(text),

            "valid_till":
                dates[-1] if dates else None
        }

    # -------------------------------------------
    # MASTER ROUTER
    # -------------------------------------------

    @classmethod
    def extract_fields(
        cls,
        doc_type,
        text
    ):

        if doc_type == "RC":
            return cls.extract_rc_fields(text)

        if doc_type == "DRIVING_LICENSE":
            return cls.extract_dl_fields(text)

        if doc_type == "POLICY":
            return cls.extract_policy_fields(text)

        if doc_type == "AADHAAR":
            return cls.extract_aadhaar_fields(text)

        if doc_type == "PAN":
            return cls.extract_pan_fields(text)

        if doc_type == "PUC":
            return cls.extract_puc_fields(text)

        return {}
# =====================================================
# METADATA ANALYZER
# =====================================================

class MetadataAnalyzer:

    @staticmethod
    def extract_metadata(file_path):

        metadata = {
            "creation_time": None,
            "modified_time": None,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "suspicious": False
        }

        try:

            stat = os.stat(file_path)

            metadata["creation_time"] = (
                datetime.fromtimestamp(
                    stat.st_ctime
                ).isoformat()
            )

            metadata["modified_time"] = (
                datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat()
            )

        except:
            pass

        try:

            image = Image.open(file_path)

            exif = image.getexif()

            if exif:

                exif_data = {}

                for tag_id, value in exif.items():

                    tag = ExifTags.TAGS.get(
                        tag_id,
                        tag_id
                    )

                    exif_data[tag] = value

                metadata["camera_make"] = (
                    exif_data.get("Make")
                )

                metadata["camera_model"] = (
                    exif_data.get("Model")
                )

                metadata["software"] = (
                    exif_data.get("Software")
                )

                suspicious_tools = [

                    "PHOTOSHOP",
                    "GIMP",
                    "CANVA",
                    "PIXLR",
                    "SNAPSEED"
                ]

                software = str(
                    metadata["software"]
                ).upper()

                for tool in suspicious_tools:

                    if tool in software:

                        metadata["suspicious"] = True

                        break

        except:
            pass

        return metadata


# =====================================================
# QR DETECTOR
# =====================================================

class QRDetector:

    @staticmethod
    def detect_qr(file_path):

        try:

            image = cv2.imread(file_path)

            detector = cv2.QRCodeDetector()

            value, points, _ = detector.detectAndDecode(
                image
            )

            return {

                "has_qr": bool(points is not None),

                "decoded_value": value
            }

        except:

            return {

                "has_qr": False,

                "decoded_value": None
            }


# =====================================================
# TEMPLATE VALIDATOR
# =====================================================

class TemplateValidator:

    REQUIRED_KEYWORDS = {

        "RC": [

            "REGISTRATION",
            "CHASSIS",
            "ENGINE"
        ],

        "DRIVING_LICENSE": [

            "LICENCE",
            "DOB"
        ],

        "POLICY": [

            "POLICY",
            "PREMIUM"
        ],

        "AADHAAR": [

            "AADHAAR",
            "UIDAI"
        ],

        "PAN": [

            "INCOME TAX",
            "PERMANENT ACCOUNT"
        ],

        "PUC": [

            "POLLUTION",
            "VALID"
        ]
    }

    @classmethod
    def validate(
        cls,
        doc_type,
        text
    ):

        keywords = cls.REQUIRED_KEYWORDS.get(
            doc_type,
            []
        )

        found = []

        missing = []

        for keyword in keywords:

            if keyword in text:

                found.append(keyword)

            else:

                missing.append(keyword)

        score = 0

        if keywords:

            score = int(
                (
                    len(found)
                    /
                    len(keywords)
                )
                * 100
            )

        return {

            "score": score,

            "found_keywords": found,

            "missing_keywords": missing
        }


# =====================================================
# MANDATORY FIELD VALIDATOR
# =====================================================

class MandatoryFieldValidator:

    REQUIRED_FIELDS = {

        "RC": [

            "vehicle_number",
            "chassis_number",
            "engine_number"
        ],

        "DRIVING_LICENSE": [

            "license_number",
            "name"
        ],

        "POLICY": [

            "policy_number"
        ],

        "AADHAAR": [

            "aadhaar_number"
        ],

        "PAN": [

            "pan_number"
        ]
    }

    @classmethod
    def validate(
        cls,
        doc_type,
        fields
    ):

        required = cls.REQUIRED_FIELDS.get(
            doc_type,
            []
        )

        present = []

        missing = []

        for field in required:

            value = fields.get(field)

            if value:

                present.append(field)

            else:

                missing.append(field)

        score = 0

        if required:

            score = int(
                (
                    len(present)
                    /
                    len(required)
                )
                * 100
            )

        return {

            "score": score,

            "present": present,

            "missing": missing
        }


# =====================================================
# TAMPERING ANALYZER
# =====================================================

class TamperingAnalyzer:

    @staticmethod
    def detect_image_splicing(file_path):

        try:

            image = cv2.imread(file_path)

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            edges = cv2.Canny(
                gray,
                100,
                200
            )

            edge_density = np.mean(edges)

            suspicious = (
                edge_density > 80
            )

            return {

                "edge_density":
                    float(edge_density),

                "suspicious":
                    suspicious
            }

        except:

            return {

                "edge_density": 0,

                "suspicious": False
            }

    @staticmethod
    def detect_cropping(file_path):

        try:

            image = cv2.imread(file_path)

            h, w = image.shape[:2]

            ratio = w / h

            suspicious = (

                ratio < 0.5
                or
                ratio > 3
            )

            return {

                "aspect_ratio": ratio,

                "suspicious": suspicious
            }

        except:

            return {

                "aspect_ratio": 0,

                "suspicious": False
            }

    @staticmethod
    def detect_text_anomalies(text):

        anomalies = []

        if text.count("XXXX") > 0:

            anomalies.append(
                "MASKED_TEXT"
            )

        if len(text) < 50:

            anomalies.append(
                "VERY_LOW_TEXT"
            )

        return anomalies


# =====================================================
# LOGO HEURISTICS
# =====================================================

class LogoHeuristics:

    GOVT_KEYWORDS = [

        "GOVERNMENT OF INDIA",
        "TRANSPORT DEPARTMENT",
        "UIDAI",
        "MINISTRY",
        "ROAD TRANSPORT"
    ]

    @classmethod
    def detect(
        cls,
        text
    ):

        hits = []

        for keyword in cls.GOVT_KEYWORDS:

            if keyword in text:

                hits.append(keyword)

        return {

            "detected": len(hits) > 0,

            "keywords": hits
        }


# =====================================================
# AUTHENTICITY ENGINE
# =====================================================

class AuthenticityEngine:

    @classmethod
    def analyze(
        cls,
        file_path,
        doc_type,
        text,
        fields
    ):

        template = (
            TemplateValidator.validate(
                doc_type,
                text
            )
        )

        mandatory = (
            MandatoryFieldValidator.validate(
                doc_type,
                fields
            )
        )

        metadata = (
            MetadataAnalyzer.extract_metadata(
                file_path
            )
        )

        qr = (
            QRDetector.detect_qr(
                file_path
            )
        )

        logo = (
            LogoHeuristics.detect(
                text
            )
        )

        splice = (
            TamperingAnalyzer.detect_image_splicing(
                file_path
            )
        )

        crop = (
            TamperingAnalyzer.detect_cropping(
                file_path
            )
        )

        text_anomalies = (
            TamperingAnalyzer.detect_text_anomalies(
                text
            )
        )

        score = 0

        score += (
            template["score"] * 0.35
        )

        score += (
            mandatory["score"] * 0.35
        )

        if qr["has_qr"]:
            score += 10

        if logo["detected"]:
            score += 10

        if not metadata["suspicious"]:
            score += 10

        if splice["suspicious"]:
            score -= 15

        if crop["suspicious"]:
            score -= 15

        if text_anomalies:
            score -= 10

        score = max(
            0,
            min(
                100,
                int(score)
            )
        )

        return {

            "authenticity_score":
                score,

            "template_validation":
                template,

            "mandatory_fields":
                mandatory,

            "metadata":
                metadata,

            "qr_analysis":
                qr,

            "logo_analysis":
                logo,

            "tampering": {

                "splicing":
                    splice,

                "cropping":
                    crop,

                "text_anomalies":
                    text_anomalies
            }
        }
# =====================================================
# DUPLICATE DETECTION
# =====================================================

class DuplicateDetector:

    def __init__(self):

        self.known_hashes = set()

    def check_duplicate(
        self,
        file_hash
    ):

        duplicate = (
            file_hash
            in self.known_hashes
        )

        self.known_hashes.add(
            file_hash
        )

        return {

            "is_duplicate":
                duplicate,

            "file_hash":
                file_hash
        }


duplicate_detector = (
    DuplicateDetector()
)


# =====================================================
# CROSS DOCUMENT VALIDATION
# =====================================================

class CrossDocumentValidator:

    @staticmethod
    def compare_values(
        left,
        right
    ):

        if not left or not right:

            return {

                "match": False,

                "reason":
                    "MISSING_VALUE"
            }

        match = (

            str(left).strip().upper()
            ==
            str(right).strip().upper()

        )

        return {

            "match": match,

            "left": left,

            "right": right
        }

    # ----------------------------------
    # RC VS POLICY
    # ----------------------------------

    @classmethod
    def validate_rc_policy(
        cls,
        rc_fields,
        policy_fields
    ):

        checks = {

            "vehicle_number":
                cls.compare_values(
                    rc_fields.get(
                        "vehicle_number"
                    ),
                    policy_fields.get(
                        "vehicle_number"
                    )
                ),

            "owner_name":
                cls.compare_values(
                    rc_fields.get(
                        "owner_name"
                    ),
                    policy_fields.get(
                        "owner_name"
                    )
                )
        }

        mismatches = [

            k
            for k, v
            in checks.items()
            if not v["match"]
        ]

        score = int(

            (
                len(checks)
                -
                len(mismatches)
            )

            /

            max(
                len(checks),
                1
            )

            * 100
        )

        return {

            "score": score,

            "checks": checks,

            "mismatches":
                mismatches
        }

    # ----------------------------------
    # RC VS INSPECTION
    # ----------------------------------

    @classmethod
    def validate_rc_inspection(
        cls,
        rc_fields,
        inspection_fields
    ):

        checks = {

            "vehicle_number":
                cls.compare_values(
                    rc_fields.get(
                        "vehicle_number"
                    ),
                    inspection_fields.get(
                        "vehicle_number"
                    )
                )
        }

        mismatches = [

            k
            for k, v
            in checks.items()
            if not v["match"]
        ]

        score = int(

            (
                len(checks)
                -
                len(mismatches)
            )

            /

            max(
                len(checks),
                1
            )

            * 100
        )

        return {

            "score": score,

            "checks": checks,

            "mismatches":
                mismatches
        }

    # ----------------------------------
    # VEHICLE AGE CHECK
    # ----------------------------------

    @staticmethod
    def validate_vehicle_age(
        registration_dates
    ):

        if not registration_dates:

            return {

                "valid": False,

                "reason":
                    "NO_DATE"
            }

        date_str = (
            registration_dates[0]
        )

        parsed = None

        formats = [

            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y/%m/%d"
        ]

        for fmt in formats:

            try:

                parsed = datetime.strptime(
                    date_str,
                    fmt
                )

                break

            except:
                pass

        if not parsed:

            return {

                "valid": False,

                "reason":
                    "INVALID_DATE"
            }

        years = (

            datetime.now()
            -
            parsed

        ).days / 365

        return {

            "valid": True,

            "vehicle_age":
                round(years, 1)
        }


# =====================================================
# FRAUD ENGINE
# =====================================================

class FraudEngine:

    @staticmethod
    def risk_level(score):

        if score >= 70:
            return "HIGH"

        if score >= 40:
            return "MEDIUM"

        return "LOW"

    @classmethod
    def calculate(
        cls,
        doc_type,
        ocr_confidence,
        classification_confidence,
        authenticity_score,
        quality_status,
        fields,
        duplicate_info
    ):

        score = 0

        reasons = []

        # --------------------------
        # OCR CONFIDENCE
        # --------------------------

        if ocr_confidence < 40:

            score += 25

            reasons.append(
                "LOW_OCR_CONFIDENCE"
            )

        elif ocr_confidence < 60:

            score += 10

            reasons.append(
                "MODERATE_OCR_CONFIDENCE"
            )

        # --------------------------
        # CLASSIFICATION
        # --------------------------

        if classification_confidence < 50:

            score += 15

            reasons.append(
                "LOW_CLASSIFICATION_CONFIDENCE"
            )

        # --------------------------
        # AUTHENTICITY
        # --------------------------

        if authenticity_score < 50:

            score += 25

            reasons.append(
                "LOW_AUTHENTICITY_SCORE"
            )

        elif authenticity_score < 70:

            score += 10

        # --------------------------
        # IMAGE QUALITY
        # --------------------------

        if "VERY_BLURRY" in quality_status:

            score += 20

            reasons.append(
                "VERY_BLURRY_IMAGE"
            )

        if "BLURRY" in quality_status:

            score += 10

            reasons.append(
                "BLURRY_IMAGE"
            )

        if "TOO_DARK" in quality_status:

            score += 10

            reasons.append(
                "DARK_IMAGE"
            )

        if "OVER_EXPOSED" in quality_status:

            score += 10

            reasons.append(
                "OVER_EXPOSED_IMAGE"
            )

        # --------------------------
        # DUPLICATE
        # --------------------------

        if duplicate_info[
            "is_duplicate"
        ]:

            score += 30

            reasons.append(
                "DUPLICATE_UPLOAD"
            )

        # --------------------------
        # MISSING FIELDS
        # --------------------------

        missing = 0

        for value in fields.values():

            if not value:

                missing += 1

        if missing >= 3:

            score += 20

            reasons.append(
                "MULTIPLE_MISSING_FIELDS"
            )

        elif missing >= 1:

            score += 5

        score = min(
            score,
            100
        )

        return {

            "fraud_score":
                score,

            "risk_level":
                cls.risk_level(
                    score
                ),

            "reasons":
                reasons
        }


# =====================================================
# VEHICLE PHOTO ANALYZER
# =====================================================

class VehiclePhotoAnalyzer:

    @staticmethod
    def analyze(
        file_path
    ):

        result = {

            "is_vehicle_photo":
                True,

            "front_view":
                False,

            "rear_view":
                False,

            "dashboard":
                False,

            "odometer":
                False
        }

        # Placeholder for future CV models

        return result
# =====================================================
# DECISION ENGINE
# =====================================================

class DecisionEngine:

    @staticmethod
    def decide(
        fraud_score,
        authenticity_score,
        ocr_confidence,
        classification_confidence
    ):

        reasons = []

        # --------------------------
        # REJECT
        # --------------------------

        if fraud_score >= 80:

            reasons.append(
                "HIGH_FRAUD_RISK"
            )

            return {
                "decision": "REJECT",
                "reasons": reasons
            }

        # --------------------------
        # MANUAL REVIEW
        # --------------------------

        if authenticity_score < 60:

            reasons.append(
                "LOW_AUTHENTICITY"
            )

            return {
                "decision": "MANUAL_REVIEW",
                "reasons": reasons
            }

        if ocr_confidence < 60:

            reasons.append(
                "LOW_OCR_CONFIDENCE"
            )

            return {
                "decision": "MANUAL_REVIEW",
                "reasons": reasons
            }

        if classification_confidence < 50:

            reasons.append(
                "LOW_CLASSIFICATION_CONFIDENCE"
            )

            return {
                "decision": "MANUAL_REVIEW",
                "reasons": reasons
            }

        # --------------------------
        # AUTO APPROVE
        # --------------------------

        reasons.append(
            "ALL_CHECKS_PASSED"
        )

        return {
            "decision": "AUTO_APPROVE",
            "reasons": reasons
        }


# =====================================================
# MAIN DOCUMENT VERIFIER
# =====================================================

class DocumentVerifier:

    def __init__(self):

        self.classifier = (
            get_classifier()
        )

    # ------------------------------------------
    # OCR ROUTER
    # ------------------------------------------

    def extract_text(
        self,
        file_path
    ):

        ext = (
            os.path.splitext(
                file_path
            )[1]
            .lower()
        )

        if ext == ".pdf":

            text = (
                OCREngine.extract_text_from_pdf(
                    file_path
                )
            )

            return {

                "text": text,

                "confidence": 95
            }

        return (
            OCREngine.extract_text_from_image(
                file_path
            )
        )

    # ------------------------------------------
    # DOCUMENT ANALYSIS
    # ------------------------------------------

    def analyze_document(
        self,
        file_path
    ):

        # ----------------------------------
        # HASH
        # ----------------------------------

        file_hash = (
            calculate_file_hash(
                file_path
            )
        )

        duplicate_info = (
            duplicate_detector
            .check_duplicate(
                file_hash
            )
        )

        # ----------------------------------
        # OCR
        # ----------------------------------

        ocr = self.extract_text(
            file_path
        )

        text = ocr["text"]

        ocr_confidence = (
            ocr["confidence"]
        )

        # ----------------------------------
        # CLASSIFICATION
        # ----------------------------------

        classification = (
            self.classifier.classify(
                text
            )
        )

        doc_type = (
            classification[
                "document_type"
            ]
        )

        classification_confidence = (
            classification[
                "confidence"
            ]
        )

        # ----------------------------------
        # FIELD EXTRACTION
        # ----------------------------------

        fields = (
            FieldExtractor
            .extract_fields(
                doc_type,
                text
            )
        )

        # ----------------------------------
        # IMAGE QUALITY
        # ----------------------------------

        ext = (
            os.path.splitext(
                file_path
            )[1]
            .lower()
        )

        image_quality = {

            "blur_score": None,
            "brightness_score": None,
            "status": []
        }

        if ext != ".pdf":

            blur_score = (
                ImageQualityAnalyzer
                .blur_score(
                    file_path
                )
            )

            brightness_score = (
                ImageQualityAnalyzer
                .brightness_score(
                    file_path
                )
            )

            quality_status = (
                ImageQualityAnalyzer
                .quality_status(
                    blur_score,
                    brightness_score
                )
            )

            image_quality = {

                "blur_score":
                    blur_score,

                "brightness_score":
                    brightness_score,

                "status":
                    quality_status
            }

        else:

            quality_status = [
                "PDF"
            ]

        # ----------------------------------
        # AUTHENTICITY
        # ----------------------------------

        authenticity = (
            AuthenticityEngine
            .analyze(
                file_path,
                doc_type,
                text,
                fields
            )
        )

        authenticity_score = (
            authenticity[
                "authenticity_score"
            ]
        )

        # ----------------------------------
        # FRAUD
        # ----------------------------------

        fraud = (
            FraudEngine.calculate(
                doc_type,
                ocr_confidence,
                classification_confidence,
                authenticity_score,
                quality_status,
                fields,
                duplicate_info
            )
        )

        # ----------------------------------
        # DECISION
        # ----------------------------------

        decision = (
            DecisionEngine.decide(
                fraud[
                    "fraud_score"
                ],
                authenticity_score,
                ocr_confidence,
                classification_confidence
            )
        )

        # ----------------------------------
        # RESPONSE
        # ----------------------------------

        return {

            "document_type":
                doc_type,

            "classification_confidence":
                round(
                    classification_confidence,
                    2
                ),

            "ocr_confidence":
                round(
                    ocr_confidence,
                    2
                ),

            "file_hash":
                file_hash,

            "duplicate":
                duplicate_info,

            "image_quality":
                image_quality,

            "fields":
                fields,

            "authenticity":
                authenticity,

            "fraud":
                fraud,

            "decision":
                decision,

            "ocr_text":
                text[:5000]
        }


# =====================================================
# GLOBAL VERIFIER
# =====================================================

verifier = (
    DocumentVerifier()
)


# =====================================================
# PUBLIC API
# =====================================================

def verify_document(
    file_path
):

    return (
        verifier.analyze_document(
            file_path
        )
    )
# =====================================================
# EXTERNAL VERIFICATION PROVIDERS
# =====================================================

class ExternalVerificationProvider:

    """
    Replace these methods with:

    - Government Registry APIs
    - VAHAN APIs
    - Insurance Core APIs
    - KYC APIs
    - Fraud Watchlists

    """

    @staticmethod
    def verify_vehicle_registration(
        registration_number
    ):

        return {

            "verified": False,

            "source":
                "NOT_CONFIGURED",

            "message":
                "Vehicle registry API not connected"
        }

    @staticmethod
    def verify_driving_license(
        license_number
    ):

        return {

            "verified": False,

            "source":
                "NOT_CONFIGURED",

            "message":
                "DL verification API not connected"
        }

    @staticmethod
    def verify_policy(
        policy_number
    ):

        return {

            "verified": False,

            "source":
                "NOT_CONFIGURED",

            "message":
                "Policy verification API not connected"
        }

    @staticmethod
    def verify_kyc(
        identifier
    ):

        return {

            "verified": False,

            "source":
                "NOT_CONFIGURED",

            "message":
                "KYC API not connected"
        }


# =====================================================
# APPLICATION VALIDATOR
# =====================================================

class ApplicationValidator:

    @staticmethod
    def validate_rc(
        result
    ):

        fields = result.get(
            "fields",
            {}
        )

        return {

            "registration_number":
                fields.get(
                    "vehicle_number"
                ),

            "owner_name":
                fields.get(
                    "owner_name"
                ),

            "chassis_number":
                fields.get(
                    "chassis_number"
                ),

            "engine_number":
                fields.get(
                    "engine_number"
                )
        }

    @staticmethod
    def validate_policy(
        result
    ):

        fields = result.get(
            "fields",
            {}
        )

        return {

            "policy_number":
                fields.get(
                    "policy_number"
                ),

            "vehicle_number":
                fields.get(
                    "vehicle_number"
                ),

            "insurer":
                fields.get(
                    "insurer_name"
                )
        }


# =====================================================
# MULTI DOCUMENT APPLICATION
# =====================================================

class InsuranceApplicationVerifier:

    def __init__(self):

        self.document_results = []

    # ------------------------------------------
    # ADD DOCUMENT
    # ------------------------------------------

    def add_document(
        self,
        file_path
    ):

        result = verify_document(
            file_path
        )

        self.document_results.append(
            result
        )

        return result

    # ------------------------------------------
    # GET DOCUMENT
    # ------------------------------------------

    def get_document(
        self,
        doc_type
    ):

        for doc in self.document_results:

            if doc[
                "document_type"
            ] == doc_type:

                return doc

        return None

    # ------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------

    def cross_validate(self):

        validation = {}

        rc = self.get_document(
            "RC"
        )

        policy = self.get_document(
            "POLICY"
        )

        if rc and policy:

            validation[
                "rc_policy"
            ] = (

                CrossDocumentValidator
                .validate_rc_policy(
                    rc["fields"],
                    policy["fields"]
                )
            )

        return validation

    # ------------------------------------------
    # EXTERNAL CHECKS
    # ------------------------------------------

    def external_verification(self):

        checks = {}

        rc = self.get_document(
            "RC"
        )

        dl = self.get_document(
            "DRIVING_LICENSE"
        )

        policy = self.get_document(
            "POLICY"
        )

        aadhaar = self.get_document(
            "AADHAAR"
        )

        # --------------------------
        # VEHICLE
        # --------------------------

        if rc:

            reg_no = (

                rc["fields"]
                .get(
                    "vehicle_number"
                )
            )

            if reg_no:

                checks[
                    "vehicle_registry"
                ] = (

                    ExternalVerificationProvider
                    .verify_vehicle_registration(
                        reg_no
                    )
                )

        # --------------------------
        # DL
        # --------------------------

        if dl:

            license_no = (

                dl["fields"]
                .get(
                    "license_number"
                )
            )

            if license_no:

                checks[
                    "driving_license"
                ] = (

                    ExternalVerificationProvider
                    .verify_driving_license(
                        license_no
                    )
                )

        # --------------------------
        # POLICY
        # --------------------------

        if policy:

            policy_no = (

                policy["fields"]
                .get(
                    "policy_number"
                )
            )

            if policy_no:

                checks[
                    "policy"
                ] = (

                    ExternalVerificationProvider
                    .verify_policy(
                        policy_no
                    )
                )

        # --------------------------
        # KYC
        # --------------------------

        if aadhaar:

            aadhaar_no = (

                aadhaar["fields"]
                .get(
                    "aadhaar_number"
                )
            )

            if aadhaar_no:

                checks[
                    "kyc"
                ] = (

                    ExternalVerificationProvider
                    .verify_kyc(
                        aadhaar_no
                    )
                )

        return checks

    # ------------------------------------------
    # APPLICATION SCORE
    # ------------------------------------------

    def application_score(self):

        scores = []

        for doc in self.document_results:

            fraud_score = (

                doc["fraud"]
                ["fraud_score"]
            )

            scores.append(
                100 - fraud_score
            )

        if not scores:

            return 0

        return round(
            sum(scores)
            /
            len(scores),
            2
        )

    # ------------------------------------------
    # FINAL DECISION
    # ------------------------------------------

    def final_decision(self):

        if not self.document_results:

            return {

                "decision":
                    "INCOMPLETE",

                "reason":
                    "NO_DOCUMENTS"
            }

        required = {

            "RC",
            "POLICY"
        }

        uploaded = {

            d["document_type"]
            for d
            in self.document_results
        }

        missing = (
            required
            -
            uploaded
        )

        if missing:

            return {

                "decision":
                    "INCOMPLETE",

                "missing":
                    list(missing)
            }

        app_score = (
            self.application_score()
        )

        cross_validation = (
            self.cross_validate()
        )

        mismatches = []

        for section in (
            cross_validation.values()
        ):

            mismatches.extend(
                section.get(
                    "mismatches",
                    []
                )
            )

        # ----------------------
        # AUTO APPROVE
        # ----------------------

        if (

            app_score >= 80

            and

            len(mismatches) == 0

        ):

            return {

                "decision":
                    "AUTO_APPROVE",

                "application_score":
                    app_score
            }

        # ----------------------
        # MANUAL REVIEW
        # ----------------------

        if app_score >= 60:

            return {

                "decision":
                    "MANUAL_REVIEW",

                "application_score":
                    app_score,

                "mismatches":
                    mismatches
            }

        # ----------------------
        # REJECT
        # ----------------------

        return {

            "decision":
                "REJECT",

            "application_score":
                app_score,

            "mismatches":
                mismatches
        }

    # ------------------------------------------
    # COMPLETE REPORT
    # ------------------------------------------

    def generate_report(self):

        return {

            "documents":
                self.document_results,

            "cross_validation":
                self.cross_validate(),

            "external_verification":
                self.external_verification(),

            "application_score":
                self.application_score(),

            "final_decision":
                self.final_decision()
        }