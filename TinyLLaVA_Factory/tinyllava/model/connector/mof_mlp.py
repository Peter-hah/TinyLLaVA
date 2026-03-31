import torch
import torch.nn as nn

from . import register_connector
from .base import Connector


    
       
class MoFMLP(nn.Module): #  Mixture-of-Features MLP（特征混合多层感知机）
    # 不是简单的线性堆叠，而是对输入特征做动态调制（γ・x+β）
    def __init__(self, config):
        super().__init__()
        
        modules_clip = [nn.Linear(config.vision_hidden_size, config.hidden_size), 
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                    ]

        modules_dinov2 = [nn.Linear(config.vision_hidden_size, config.hidden_size), 
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                    ]

        self.clip = nn.Sequential(*modules_clip)
        self.dinov2 = nn.Sequential(*modules_dinov2)



    def forward(self, x):

        image_features_clip = self.clip(x[0])
        image_features_dinov2 = self.dinov2(x[1])
        # 多模态模型里的「双图像编码器」，专门提取两种不同的图像特征  [batch_size, 3, H, W]
        # x[0] 和 x[1]输入的同一张图像
        # 预处理方式不同 CLIP：懂语义（这是什么） DINOv2：懂细节 / 结构（在哪里、长什么样）
        # image_features_clip[batch_size, num_patches, clip_dim]
        # image_features_dinov2 [batch_size, num_patches, dinov2_dim]

        bs = image_features_clip.size(0)
        total_len = image_features_clip.size(1)+image_features_dinov2.size(1) #patch总数量
        dim = image_features_clip.size(-1)

        merged_features = torch.empty(bs, total_len, dim).to(device=x[0].device, dtype=x[0].dtype)
        merged_features[:,0::2] = image_features_clip #从第0列开始，偶数列取值
        merged_features[:,1::2] = image_features_dinov2 #从1开始奇数列取值
        #，image_features_clip 和 image_features_dinov2 的长度，刚好就是 merged_features 长度的一半。

        return merged_features
    
    

# 内核专注网络，外壳专注管理
@register_connector('mof_mlp')    # 1. 给这个壳注册名字
class MoFMLPConnector(Connector): # 2. 壳类
    def __init__(self, config):
        super().__init__()

        self._connector = MoFMLP(config) # 3. 壳里面装真正的模型
