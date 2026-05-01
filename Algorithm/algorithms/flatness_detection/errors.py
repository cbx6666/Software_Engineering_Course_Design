"""平整度检测流程的领域异常定义。"""


class FlatnessDetectionError(RuntimeError):
    """平整度检测对外暴露的基础异常。"""


class ProjectionDiffError(FlatnessDetectionError):
    """投影反射差分阶段异常。"""


class CornerDetectionError(FlatnessDetectionError):
    """棋盘格角点检测阶段异常。"""


class CornerMatchingError(FlatnessDetectionError):
    """左右角点匹配阶段异常。"""
