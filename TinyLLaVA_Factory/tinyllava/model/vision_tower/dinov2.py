from transformers import Dinov2Model, AutoImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('dinov2')      
class DINOv2VisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = Dinov2Model(cfg)
        self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path)

        # cfg_dinov2 = AutoConfig.from_pretrained(cfg.model_name_or_path2)
        # self.dinov2 = Dinov2Model(cfg_dinov2)
        # 先从路径加载 config，再创建模型
