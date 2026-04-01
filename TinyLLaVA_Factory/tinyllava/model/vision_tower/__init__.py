import os

from ...utils import import_modules
# 从当前这个 .py 文件往上数 3 级包目录
# 不是从你打开终端的位置数！不是从你工作目录数！只看文件在项目里的层级！


VISION_TOWER_FACTORY = {}

def VisionTowerFactory(vision_tower_name): # 实现按模型名取模型
    vision_tower_name = vision_tower_name.split(':')[0]
    model = None
    for name in VISION_TOWER_FACTORY.keys():
        if name.lower() in vision_tower_name.lower(): # 把 name 这个字符串全部变成小写
            model = VISION_TOWER_FACTORY[name]
    assert model, f"{vision_tower_name} is not registered"
    # assert 条件, 错误信息
    return model


def register_vision_tower(name):
    def register_vision_tower_cls(cls):
        if name in VISION_TOWER_FACTORY:
            return VISION_TOWER_FACTORY[name]
        VISION_TOWER_FACTORY[name] = cls
        return cls
    return register_vision_tower_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__) # os.path.dirname(...) = 取所在文件夹 __file__：当前 文件本身路径
import_modules(models_dir, "tinyllava.model.vision_tower")
