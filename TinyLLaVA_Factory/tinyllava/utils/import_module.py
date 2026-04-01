import importlib
import os

def import_modules(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and file.endswith(".py")
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            # 找到file.find(".py")文件名中".py"的索引值
            importlib.import_module(namespace + "." + model_name)
            # importlib.import_module("包名.模块名")
    #         它做了 2 件事：

    # 把那个 .py 文件从头到尾跑一遍
    # 让里面的 @注册 装饰器执行
    # 把类注册到 VISION_TOWER_FACTORY
