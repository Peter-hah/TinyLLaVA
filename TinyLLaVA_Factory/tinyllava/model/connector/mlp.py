import re #Python 自带的「正则表达式」工具库 Regular Expression

import torch.nn as nn

from . import register_connector
from .base import Connector


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}


    
    
@register_connector('mlp')  #@：Python 装饰器语法 register_connector：一个注册器函数（自己写的工具） ('mlp')：给这个连接器起一个名字叫 mlp
# @装饰器 天生就只作用于紧跟着它的下一个函数或类，这是 Python 解释器的语法规则，谁写代码都必须遵守
class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__()
        
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.connector_type)

        # 正则表达式 判断 config.connector_type 是不是 mlp数字x_gelu 这种格式
        act_type = config.connector_type.split('_')[-1] #把字符串按 下划线 _ 切开，变成一个列表
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)] #这就是个层列表
        #nn.ReLU → 类本身
        # nn.ReLU() → 创建一个真正的层（实例）
        # 你要加到模型里的是层实例，不是类！
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            
        self._connector = nn.Sequential(*modules)
        # *解包/展开， 把这些层串成一条前向传播流水线，它自带 forward，自带参数，自带训练功能
        # nn.Sequential 把它们串成一个大模型

   
        
#     @property
#     def config(self):
#         return {"connector_type": 'mlp',
#                 "in_hidden_size": self.in_hidden_size, 
#                 "out_hidden_size": self.out_hidden_size
#                }
    
