from transformers import PreTrainedModel
from src.model import LLaMA
from register_model.llama_config import LLaMAConfig


class LLaMAModel(PreTrainedModel):
    config_class = LLaMAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LLaMA(
            n_heads = config.n_heads, 
            emb_dim = config.emb_dim, 
            vocab_size = config.vocab_size, 
            n_layers = config.n_layers, 
            norm_eps = config.norm_eps, 
            max_seq_len = config.max_seq_len, 
            max_batch_size = config.max_batch_size, 
            device = config.device, 
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)

    