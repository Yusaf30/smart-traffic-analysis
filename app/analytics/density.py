class DensityClassifier:
    """Classify traffic density using the number of vehicles."""

    def __init__(self, low_max: int = 5, medium_max: int = 8) -> None:
        if low_max < 0 or medium_max <= low_max:
            raise ValueError("Thresholds must satisfy 0 <= low_max < medium_max.")

        self.low_max = low_max
        self.medium_max = medium_max

    def classify(self, vehicle_count: int) -> str:
        """Return LOW, MEDIUM, or HIGH for a vehicle count."""
        if isinstance(vehicle_count, bool) or not isinstance(vehicle_count, int):
            raise TypeError("vehicle_count must be a non-negative integer.")

        if vehicle_count < 0:
            raise ValueError("vehicle_count cannot be negative.")

        if vehicle_count <= self.low_max:
            return "LOW"
        if vehicle_count <= self.medium_max:
            return "MEDIUM"
        return "HIGH"
