import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from  transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import torch.nn.functional as F

from src.model import LLaMA


@torch.no_grad()
def _get_grad_norm(model, norm_type=2):
    """
    Calculates the gradient norm for logging.

    Args:
        norm_type (float | str | None): the order of the norm.
    Returns:
        total_norm (float): the calculated norm.
    """
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return total_norm.item()

if __name__ == '__main__':

    emb_dim = 512
    n_layers = 16
    n_heads = 16
    norm_eps = 1e-5
    max_batch_size = 10
    max_seq_len = 1024
    vocab_size = 32000


    dataset = load_dataset("ashaba1in/small_openwebtext")
    # cropped_dataset = dataset['train'].select(range(1))
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', use_fast=False, token='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')

    tokenizer.add_eos_token = True
    tokenizer.add_bos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    def encode(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=1024, return_tensors="pt")

    tokenized_dataset = dataset['train'].map(encode, batched=True, batch_size=10)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.remove_columns(["attention_mask"])


    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = LLaMA(n_heads, emb_dim, vocab_size, n_layers, norm_eps, max_seq_len, max_batch_size, device=device).to(device)


    batch_size = 2
    run = wandb.init(project="LLaMA", entity = "polina-kadeyshvili")
    
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=1024)
    loader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

    def train_epoch(model, loader, optimizer):
        model.train()
        loss =[]
        for data in tqdm(loader):
            data = data['input_ids'].to(device)
            optimizer.zero_grad()
            output = model(data)
            result_out = output[:, :-1]
            target = data[:, 1:].to(device)
            l = F.cross_entropy(result_out.reshape(-1, 32000), target.reshape(-1))
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            loss.append(l.item() * target.shape[0])
            optimizer.step()
            grad_norm = _get_grad_norm(model)
            wandb.log({'train_loss': l.item(), "grad_norm" : grad_norm})

        return np.mean(loss)

    def train(model, optimizer,loader,  n_epochs, scheduler=None):
        train_loss = 0.0
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, loader, optimizer)

            print(f"Epoch {epoch+1}")
            print(f" train loss: {train_loss}")

            if scheduler is not None:
                scheduler.step()


    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)

    train(model, optimizer, loader, scheduler=scheduler, n_epochs=1)
    torch.save(model.state_dict(), "/home/makarovan/llama_implementation/saved_model.pth")
