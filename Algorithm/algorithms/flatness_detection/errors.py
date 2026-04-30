"""Domain errors for flatness detection."""


class FlatnessDetectionError(RuntimeError):
    """Base class for user-facing flatness detection failures."""


class ProjectionDiffError(FlatnessDetectionError):
    """Projection reflection difference failed."""


class CornerDetectionError(FlatnessDetectionError):
    """Chessboard corner detection failed."""


class CornerMatchingError(FlatnessDetectionError):
    """Left/right corner matching failed."""
