from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *
# from tinyllava.utils.data_utils import get_value_from_kwargs

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
    


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig #模型配置类
    base_model_prefix = "model"
    supports_gradient_checkpointing = True #梯度检查点（为梯度做检查点），节省显存，不全保留中间结果
    _no_split_modules = ["LlavaVisionAttention"] #在做模型分块或设备切分时，不要把 LlavaVisionAttention 这种模块拆开
    _skip_keys_device_placement = "past_key_values" #告诉模型「past_key_values」这个张量不自动分配到指定设备（如 GPU），需要手动处理它的设备位置。
    _supports_flash_attn_2 = True #这个模型支持 Flash Attention 2，可以使用更高效的 attention 后端

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        ) #确定初始化时正态分布的标准差std

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std) #某个模块里有 class_embedding 这个属性，就单独初始化它。

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_() #如果模块是全连接层 Linear卷积层 Conv2d那就：权重用正态分布初始化，偏置初始化为0
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_() # 整个 embedding weight 先正态初始化
                # 如果Embedding 层设置了「填充索引（padding_idx）」，就把该索引对应的权重向量强制置为全 0
    @property #@property是 Python 的装饰器（decorator），核心作用是：把一个类的方法「伪装」成属性
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa #TinyLLaVA 是否支持某种高效 attention，取决于内部 language model。
    # Scaled Dot-Product Attention缩放点积注意力 
    # 调用当前类实例的 language_model的_supports_sdpa属性/方法（如果是方法，这里省略了 ()， —— 通常是一个子模块，返回布尔值


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel): #继承自刚才那个基类 TinyLlavaPreTrainedModel ForConditionalGeneration用于条件生成
    def __init__(self, config: TinyLlavaConfig): #类型注解的分隔符，左边是参数名，右边是参数的「期望类型」
        
        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config) # (...)(config.text_config) 
        # 然后 [0] 取出返回结果中的第一个元素，通常就是：语言模型类本身，按照配置，创建一个具体的语言模型
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)# 按照配置，创建图像编码器
        self.connector = ConnectorFactory(config.connector_type)(config)

        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        # Tokenizer：具体的 tokenizer 类 post_load：tokenizer 加载完之后的后处理函数（Hugging Face）
        self.tokenizer = post_load(Tokenizer.from_pretrained( # 从预训练路径加载 tokenizer
            config.tokenizer_name_or_path,# tokenizer 的名字或本地路径
            cache_dir = config.cache_dir,# 缓存目录
            model_max_length = config.tokenizer_model_max_length,# 最大token长度
            padding_side = config.tokenizer_padding_side,# 左填充还是右填充
            use_fast = config.tokenizer_use_fast,# 是否使用 fast tokenizer
        ))
        self.post_init()# PyTorch/Hugging Face 模型类中标准化的「初始化后收尾方法」
        #  把初始化时不宜分散写的逻辑（如参数初始化、设备适配、权重校验等）集中到这个方法里
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    # self实例转发/代理其内部语言模型的get_input_embeddings()方法调用，直接返回语言模型的输入嵌入层

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    # self 实例调用自身的set_input_embeddings方法时，会把传入的嵌入层值，
    # 传递给内部语言模型的同名方法，完成语言模型输入嵌入层的设置 / 替换
    def get_output_embeddings(self): # 获取输出 embedding 层
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings): # 替换输出层
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder): # 替换内部 decoder Hugging Face 风格的统一接口
        self.language_model.set_decoder(decoder)

    def get_decoder(self): # 拿到内部 decoder
        return self.language_model.get_decoder()

    def tie_weights(self):# 输入 embedding 和输出 embedding 共用权重
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        # 参数名:new_num_tokens:| Optional[int]：核心参数类型注解| =None：默认值为None，不传该参数时，方法会使用模型当前的词表大小| 可选优化参数：pad_to_multiple_of=None默认值None| 返回值类型注解
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # pad_to_multiple_of 是 resize_token_embeddings 方法的可选参数，核心作用是：将调整后的词嵌入层的 token 数量（词汇表大小）向上补齐到「指定数值的整数倍」
        # —— 本质是为了适配硬件（GPU/TPU）的并行计算优化，提升模型训练 / 推理速度
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds # 把新的词表大小同步记录下来。
    # 调整词表大小，并同步更新相关配置

    
    def forward(
        self,
        input_ids: torch.LongTensor = None, # 把文本转换成的数字序列（比如把 "hello world" 转成 [101, 7592, 2088, 102]）
        attention_mask: Optional[torch.Tensor] = None, # Optional = 这个参数「可以传，也可以不传」
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 缓存的 KV cache。推理生成时常用，用于避免每一步都重复计算历史 token 的注意力。
        inputs_embeds: Optional[torch.FloatTensor] = None, # 如果你已经自己把 token 转成 embedding，就可以直接传这个，而不是传 input_ids
        labels: Optional[torch.LongTensor] = None, # 训练时的监督标签。如果传了 labels，语言模型通常会顺便计算 loss。
        use_cache: Optional[bool] = None, # 是否使用 cache
        output_attentions: Optional[bool] = None, # 注意力图
        output_hidden_states: Optional[bool] = None, # 每层 hidden states
        images: Optional[torch.FloatTensor] = None, # 图像张量
        image_sizes: Optional[List[List[int]]] = None, # 原始图像尺寸信息
        return_dict: Optional[bool] = None, # 返回 dict 风格还是 tuple 风格
    ) -> Union[Tuple, CausalLMOutputWithPast]: # Python 函数返回值类型注解：Union二选一，要么返回一个元组（Tuple），要么返回一个模型输出对象
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None: # 没有embedding，那模型就自己处理原始输入
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds, # 重点！模型自己生成了
                labels
            ) = self.prepare_inputs_labels_for_multimodal( # 多模态预处理工具函数
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return self.language_model.forward( # 交给内层language_model
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad() # 推理时不计算梯度，省显存、跑得快
    def generate(  # 生成回答的核心函数
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,# 文字和图片
        image_sizes: Optional[torch.Tensor] = None,# 图片尺寸
        **kwargs,# 其他生成参数（温度、最大长度等）没列出来的所有其他参数：全部自动装进 kwargs 里kwargs = keyword arguments本质是一个 字典（dict）
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None) # 如果找不到这个 key，就返回默认值 default（这里是 None）拿出来后，kwargs 里就不再有这个 key了
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # 不允许你自己传向量
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            # 有图片调用图文融合函数，把文字 + 图片打包成模型能看懂的向量 inputs_embeds
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        # 没有图片，只把文字转成向量 inputs_embeds，用语言模型自带的 embedding 层

        
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        # 把配置文件里的图片处理参数装进 kwargs，告诉视觉模型：用第几层特征、怎么选特征
        images = images.to(device=self.device, dtype=self.dtype)
        #把图片搬到GPU（或 CPU）,统一数据类型
        image_features = self.vision_tower(images, **kwargs) # 图片编码器
        image_features = self.connector(image_features)#把图片特征 → 调整成语言模型能看懂的维度
        return image_features 
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs): # 只管准备 “原始输入” 不是强制为 None，而是：有就传，没有就默认 None
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        # 搬运图片 + 准备文本输入
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        #不需要多模态时，直接返回 情况没图片，没视觉模型，或者只生成一个字
        
        image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False): #getattr(配置, 键, 默认值)
            #看模型配置里是否打开了：tune_mm_mlp_adapter（微调多模态 MLP 适配器）直接抛异常：不支持这个功能！
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # 如果参数不存在，我们就随便填个假张量
        # it is a headache to deal with None all the time.
        # 但一直处理 None 真的很烦！
        # But it is not ideal, and if you have a better idea,
        # 虽然这方法不完美
        # please open an issue / submit a PR, thanks.
        # 有更好的方法欢迎PR
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        # 备份
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            # 创建一个 和 input_ids 形状一模一样所有值都是 True 的张量，赋值给 attention_mask
        else:
            attention_mask = attention_mask.bool()
            # 强制转换成布尔类型 非0就是True，0就是False
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            # torch.arange(start, end, step)
            # 生成一连串连续整数，shape [1] = 句子长度（token 个数），input_ids.shape → 输出 (batch_size, seq_len)，torch.long64位整数
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            # 创建一个和 input_ids 形状完全一样的张量
        # 先统一填好默认值，后面安心写逻辑

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        # 布尔索引，保留为True的位置
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        # 同上，cur当前的意思


        #input_ids(batch_size, seq_len) batch_size：一次喂给模型多少句话
        #input_embeds[seq_len, hidden_dim]
        #cur_input_id[seq_len]
        #cur_input_embeds[total_seq_len, hidden_dim]
        #img_features[batch_size, num_patches, llm_hidden_dim]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids): #enumerate加上序号
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)#只是把长度加起来
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] #-1+各个图像位置的序号+序列总长度
            # 列表拼接,都是列表才能拼接
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # indices索引 长度为图像数+1
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # 里面每个元素是 1D 张量，最后是最后一个图像后到结尾
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim] #split_sizes列表中每个元素值代表每一段张量的长度，元素数=张量数
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim)) # 接收一个【张量列表】，直接拼成一个大张量！
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # torch.split(input,# 要切开的张量 split_size_or_sections,  # 怎么切（每段多长）dim=0  # 在哪个维度切（默认第0维）)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i]) #最后是cur_input_embeds_no_im[num_imgs]
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx] #去对应cur_image_idx位置的图片的 num_patches, llm_hidden_dim
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features) #加入在文字embedding中加入图片信息
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            # 形状,填充值,设备,数据类型， torch.full()创建一个指定形状、全部填充同一个值的张量，(cur_image_features.shape[0],)一维长度，长度=patch数量
            #，PyTorch 规定：要拼接的张量，必须在同一个设备上

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            # 必须把里面每一个张量单独移动
            cur_new_input_embeds = torch.cat(cur_new_input_embeds) #列表不能直接输入大模型
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds] # 截取分词器最大长度的token序列
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds) #段落里句子的最大长度
        batch_size = len(new_input_embeds) #句子数量

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    
    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)

        
    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)

            

        
        
