from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('clip')      
# 把下面这个类，丢给前面的函数去加工一下,CLIPVisionTower = register_vision_tower('clip')(CLIPVisionTower)
# 先调用 register_vision_tower('clip')
# 它返回一个装饰器
# 这个装饰器再去装饰下面的类
class CLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = CLIPVisionModel(cfg)
        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path) # image_processor图像预处理工具
        # 自动下载并创建一个图像预处理工具
        # 把图片缩放到指定尺寸（224 / 336 / 384...）
        # 归一化（mean, std）
        # 转成 PyTorch tensor
        # 做 CLIP 要求的所有图像预处理
        

