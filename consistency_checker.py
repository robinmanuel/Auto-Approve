from difflib import SequenceMatcher


class ConsistencyChecker:

    def similarity(
        self,
        value1,
        value2
    ):

        if not value1 or not value2:
            return 0

        return SequenceMatcher(
            None,
            str(value1).lower(),
            str(value2).lower()
        ).ratio()

    def check_vehicle_match(
        self,
        policy_data,
        rc_data
    ):

        policy_vehicle = (
            policy_data
            .get("fields", {})
            .get("vehicle_number")
        )

        rc_vehicle = (
            rc_data
            .get("fields", {})
            .get("vehicle_number")
        )

        if (
            not policy_vehicle
            or
            not rc_vehicle
        ):
            return {
                "passed": False,
                "reason":
                "Vehicle number missing"
            }

        if (
            policy_vehicle
            ==
            rc_vehicle
        ):
            return {
                "passed": True,
                "reason":
                "Vehicle numbers match"
            }

        return {
            "passed": False,
            "reason":
            f"Mismatch: {policy_vehicle} vs {rc_vehicle}"
        }

    def check_name_match(
        self,
        policy_data,
        license_data
    ):

        policy_name = (
            policy_data
            .get("fields", {})
            .get("customer_name")
        )

        license_name = (
            license_data
            .get("fields", {})
            .get("customer_name")
        )

        if (
            not policy_name
            or
            not license_name
        ):
            return {
                "passed": False,
                "reason":
                "Customer name missing"
            }

        score = self.similarity(
            policy_name,
            license_name
        )

        if score > 0.85:

            return {
                "passed": True,
                "reason":
                "Names match"
            }

        return {
            "passed": False,
            "reason":
            "Customer name mismatch"
        }

    def check_invoice_amount(
        self,
        invoice_data,
        vehicle_value
    ):

        invoice_amount = (
            invoice_data
            .get("fields", {})
            .get("estimated_amount")
        )

        if not invoice_amount:

            return {
                "passed": False,
                "reason":
                "Invoice amount unavailable"
            }

        if (
            invoice_amount
            >
            vehicle_value
        ):

            return {
                "passed": False,
                "reason":
                "Repair exceeds vehicle value"
            }

        return {
            "passed": True,
            "reason":
            "Repair amount reasonable"
        }

    def evaluate(
        self,
        documents,
        vehicle_value
    ):

        checks = []

        if (
            "POLICY" in documents
            and
            "RC" in documents
        ):

            checks.append(
                self.check_vehicle_match(
                    documents["POLICY"],
                    documents["RC"]
                )
            )

        if (
            "POLICY" in documents
            and
            "DRIVING_LICENSE"
            in documents
        ):

            checks.append(
                self.check_name_match(
                    documents["POLICY"],
                    documents[
                        "DRIVING_LICENSE"
                    ]
                )
            )

        if (
            "INVOICE"
            in documents
        ):

            checks.append(
                self.check_invoice_amount(
                    documents["INVOICE"],
                    vehicle_value
                )
            )

        passed = sum(
            check["passed"]
            for check in checks
        )

        total = len(checks)

        score = (
            passed / total * 100
            if total > 0
            else 0
        )

        issues = [

            check["reason"]

            for check in checks

            if not check["passed"]
        ]

        return {

            "score": round(score, 2),

            "checks": checks,

            "issues": issues
        }


checker = ConsistencyChecker()