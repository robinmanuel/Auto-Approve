from ocr import verify_document

files = [
    "samples/rc.jpg",
    "samples/dl.jpg",
    "samples/policy.pdf",
    "samples/aadhaar.jpg"
]

for file in files:

    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    result = verify_document(file)

    print(
        "TYPE:",
        result["document_type"]
    )

    print(
        "DECISION:",
        result["decision"]["decision"]
    )

    print(
        "FRAUD:",
        result["fraud"]["fraud_score"]
    )

    print(
        "FIELDS:",
        result["fields"]
    )