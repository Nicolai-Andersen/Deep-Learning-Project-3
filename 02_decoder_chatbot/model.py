import torch
import torch.nn as nn
import math

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(
            embed_size, num_heads, dropout, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, padding_mask):
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=True
        )
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        x = x + self.dropout(self.mlp(x))

        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        positional_encoding = torch.zeros(max_len, embed_size)

        position = torch.arange(0, max_len).unsqueeze(1)

        # The division term that shrinks the frequency for higher dimensions
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))

        # Even indices get sin, odd indices get cos
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(0) # Add batch dim: (1, max_len, embed_size)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        return x


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()

if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

