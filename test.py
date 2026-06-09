from pipeline import AutoApprovePipeline
import json

IMAGE_PATH = "temp_71c79fe60d094235bdb899ddb5ea97f6.jpg"  # change if needed
YOLO_WEIGHTS = "parts_segmentation.pt"

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