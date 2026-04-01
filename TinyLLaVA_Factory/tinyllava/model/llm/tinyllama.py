from transformers import LlamaForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('tinyllama')
def return_tinyllamaclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        # 也就是把 pad_token 设成 unk_token，避免没 pad token 的报错
        return tokenizer
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
