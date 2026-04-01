import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, Dinov2Model, AutoConfig
# 因为 SigLIP 的 API 结构 = 和 CLIP 完全一样！！ SigLIP 是 CLIP 的升级版

from . import register_vision_tower
from .base import VisionTower





class MoF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip = CLIPVisionModel(cfg)

        cfg_dinov2 = AutoConfig.from_pretrained(cfg.model_name_or_path2)
        self.dinov2 = Dinov2Model(cfg_dinov2)
        # 先从路径加载 config，再创建模型

#     def enable_input_require_grads(self):
#         def make_inputs_require_grad(module, input, output):
#             output.requires_grads()

#         if hasattr(self.clip, 'enable_input_require_grads'):
#             self.clip.enable_input_require_grads()
#         else:
#             self.clip.get_input_embeddings(make_inputs_require_grad)

#         if hasattr(self.dinov2, 'enable_input_require_grads'):
#             self.dinov2.enable_input_require_grads()
#         else:
#             self.dinov2.get_input_embeddings(make_inputs_require_grad)


    def forward(self, x, **kwargs):
 
        image_features_clip = self.clip(x, output_hidden_states=True)
        image_features_clip = image_features_clip.hidden_states[kwargs.get('vision_feature_layer', -2)]

        image_features_dinov2 = self.dinov2(x, output_hidden_states=True)
        image_features_dinov2 = image_features_dinov2.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features_clip = image_features_clip[:, 1:]
            image_features_dinov2 = image_features_dinov2[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features_clip = image_features_clip
            image_features_dinov2 = image_features_dinov2
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")


        image_features = image_features_clip, image_features_dinov2
        # 「自动元组打包」变量 = 值1, 值2 ———>> 变量 = (值1, 值2)
        return image_features






@register_vision_tower('mof')      # 注册一个叫 mof 的视觉塔
class MoFVisionTower(VisionTower): # 继承框架的 VisionTower
    def __init__(self, cfg):
        super().__init__(cfg)

        self._vision_tower = MoF(cfg)  # 初始化 MoF 模型（CLIP+DINOv2）

        # 图片处理器（CLIP 标准） image_processor不属于神经网络本体 MoF里面已经是处理好的tensor了
        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)

    def _load_model(self, vision_tower_name, **kwargs):
        # 拿配置里的路径
        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)

        if pretrained_vision_tower_path is None:
            # 情况1：分别加载 CLIP + DINOv2 预训练权重
            model_name_or_path_dinov2 = kwargs.pop('model_name_or_path2')
            self._vision_tower.clip = self._vision_tower.clip.from_pretrained(vision_tower_name, **kwargs)
            self._vision_tower.dinov2 = self._vision_tower.dinov2.from_pretrained(model_name_or_path_dinov2, **kwargs)
        else:
            # 情况2：加载你自己训练好的 MoF 整体权重
            vision_tower_weights = torch.load(...)
            self._vision_tower.load_state_dict(vision_tower_weights)

    def forward(self, x,** kwargs):
        device = x.data.device
        self.to(device)  # 把模型搬到图片所在设备 
        # img_processor不是在这个 forward 里用，而是给框架别的地方用
        return self._vision_tower(x, **kwargs)

