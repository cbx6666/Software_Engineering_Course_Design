import base64
from typing import Union
from algorithms.crack_detection.breakage_algorithm import GlassBreakageAlgorithm

class GlassCrackService:

    def __init__(self):
        # 初始化破裂检测算法实例
        self.crack_algo = GlassBreakageAlgorithm()

    def detect_crack(self, image_input: Union[str, bytes]) -> dict:
        # 如果是 bytes（例如上传文件的 content），先转 Base64
        if isinstance(image_input, bytes):
            image_input = base64.b64encode(image_input).decode('utf-8')
            image_input = f"data:image/jpeg;base64,{image_input}"
        
        return self.crack_algo.run(image_input)