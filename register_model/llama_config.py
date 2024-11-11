from transformers import PretrainedConfig


class LLaMAConfig(PretrainedConfig):
    model_type = "my_implementation_llama_new_new7"
    
    def __init__(
        self,
        n_heads=12,
        emb_dim=768,
        vocab_size=32000,
        n_layers=12,
        norm_eps=1e-6,
        max_seq_len=512,
        max_batch_size=16,
        device='cpu',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.device = device
        super().__init__(**kwargs)
    