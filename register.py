import torch

from register_model.model_huggin import LLaMAModel
from register_model.llama_config import LLaMAConfig


config = LLaMAConfig()
config.register_for_auto_class()
llama_model = LLaMAModel(config)


llama_model.register_for_auto_class("AutoModelForCausalLM")
checkpoint = torch.load("llama_implementation/saved_model.pth", map_location='cpu') 
llama_model.load_state_dict(checkpoint, strict=False)
llama_model.push_to_hub("my_implementation_llama")