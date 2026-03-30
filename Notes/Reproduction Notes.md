get_value_from_kwargs:
    从kwargs里取出指定参数，并且取出来后从字典里删掉

TinyLlavaPreTrainedModel(PreTrainedModel):
    TinyLlavaPreTrainedModel 继承自 Hugging Face 的 PreTrainedModel
        _init_weights(self, module):
            1 工具函数 get_value_from_kwargs
            2 模型基类声明:配置类是谁 支不支持 gradient checkpointing 支不支持 flash attention / sdpa
            3 参数初始化规则 Linear Conv2d Embedding

TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel)：
    

