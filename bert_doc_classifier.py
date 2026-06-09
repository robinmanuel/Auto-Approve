from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BERTDocumentClassifier:

    def __init__(self):

        # lightweight BERT model (fast + good enough)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # prototype descriptions for each document type
        self.labels = {
            "RC": "vehicle registration certificate rc book registration number chassis engine details transport vehicle",
            "DRIVING_LICENSE": "driving license driving licence dl number license number transport department date of birth",
            "POLICY": "insurance policy policy number premium insured vehicle coverage sum insured insurer",
            "INVOICE": "repair invoice bill garage estimate total amount gst service charge repair cost"
        }

        self.label_embeddings = {
            label: self.model.encode(text)
            for label, text in self.labels.items()
        }

    def classify(self, text: str):

        if not text:
            return {"document_type": "UNKNOWN", "confidence": 0}

        text_embedding = self.model.encode(text)

        scores = {}

        for label, emb in self.label_embeddings.items():

            sim = cosine_similarity(
                [text_embedding],
                [emb]
            )[0][0]

            scores[label] = float(sim)

        best_label = max(scores, key=scores.get)
        confidence = scores[best_label] * 100

        if confidence < 40:
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


classifier = BERTDocumentClassifier()