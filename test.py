from pipeline import AutoApprovePipeline
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = str(BASE_DIR / "temp_71c79fe60d094235bdb899ddb5ea97f6.jpg")  # change if needed
YOLO_WEIGHTS = str(BASE_DIR / "parts_segmentation.pt")

pipeline = AutoApprovePipeline(YOLO_WEIGHTS)

result = pipeline.analyze(IMAGE_PATH)

print("\n================ PIPELINE OUTPUT ================\n")

print("TOTAL COST:", result["total_estimated_cost"])
print("STATUS:", result["status"])
print("PARTS FOUND:", len(result["parts"]))

print("\n---------------- PART DETAILS ----------------\n")

for p in result["parts"]:
    print(json.dumps(p, indent=2))

print("\n================ END =================\n")