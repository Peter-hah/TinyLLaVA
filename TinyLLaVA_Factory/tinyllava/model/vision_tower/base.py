import os

import torch
import torch.nn as nn

from transformers import PreTrainedModel
# from tinyllava.utils.data_utils import get_value_from_kwargs

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None
        self._image_processor = None #图像预处理工具
        self.config = cfg #cfg = 外面传进来的配置参数
    

    def load_model(self, vision_tower_name, **kwargs): # _load_model 是 “真正干活” 的内部函数
        # load_model 是 “对外给别人调用” 的接口 下划线 _ 代表 “内部私有”
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False)
        # requires_grad → 是否需要计算梯度（是否要学习）



        
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_vision_tower_path')
        if isinstance(self._vision_tower, PreTrainedModel): # hf model
            if pretrained_vision_tower_path is not None: #优先kwargs字典里的pretrained_vision_tower_path
                vision_tower_name = pretrained_vision_tower_path
            self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)      
        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)

        print("Loading vision tower from ", vision_tower_name)
        


    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features
        

    
    @property
    def vision_tower(self):
        return self._vision_tower
        
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower
        
    
