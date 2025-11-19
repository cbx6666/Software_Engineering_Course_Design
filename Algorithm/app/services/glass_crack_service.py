import base64
import imghdr
from typing import Union
from algorithms.crack_detection.breakage_algorithm import GlassBreakageAlgorithm

class GlassCrackService:

    def __init__(self):
        # 初始化破裂检测算法实例
        self.crack_algo = GlassBreakageAlgorithm()

    def detect_crack(self, image_input: Union[str, bytes]) -> dict:
        # 如果是 bytes（例如上传文件的 content），先转 Base64
        if isinstance(image_input, bytes):
            # 动态检测图片类型
            ext = imghdr.what(None, h = image_input)  # 返回 'png', 'jpeg', 'gif' 等

            image_base64 = base64.b64encode(image_input).decode('utf-8')
            image_input = f"data:image/{ext};base64,{image_base64}"
        
        return self.crack_algo.run(image_input)