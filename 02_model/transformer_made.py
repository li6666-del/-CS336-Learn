import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5






class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T






class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
    
class RMSNorm(nn.Module):
        def  __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
            super().__init__()

            factory_kwargs = {"device": device, "dtype": dtype}

            self.d_model = d_model
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            original_dtype = x.dtype

            # 为了数值稳定，RMSNorm 通常在 float32 下计算
            x_float = x.float()

            rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
            y = x_float / rms

            return (y * self.weight).to(original_dtype)
        
        
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.gate_proj = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.up_proj = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down_proj = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        silu_gate = gate * torch.sigmoid(gate)

        return self.down_proj(silu_gate * up)
    
    
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0, device=None):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires head_dim to be even.")

        self.head_dim = head_dim
        self.theta = theta

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, heads, seq_len, head_dim)
        """
        batch, heads, seq_len, head_dim = x.shape

        if head_dim != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {head_dim}")

        positions = torch.arange(seq_len, device=x.device).float()

        # freqs: (seq_len, head_dim / 2)
        freqs = torch.outer(positions, self.inv_freq)

        cos = torch.cos(freqs)[None, None, :, :]
        sin = torch.sin(freqs)[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd

        return out
    
    
    
    
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.context_length = context_length

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            theta=rope_theta,
            device=device,
        )

        causal_mask = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool, device=device)
        )

        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        return: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        if seq_len > self.context_length:
            raise ValueError(
                f"seq_len={seq_len} exceeds context_length={self.context_length}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (batch, seq_len, d_model)
        # -> (batch, seq_len, num_heads, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # attention_scores: (batch, heads, seq_len, seq_len)
        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)

        # out: (batch, heads, seq_len, head_dim)
        out = attention_weights @ v

        # (batch, heads, seq_len, head_dim)
        # -> (batch, seq_len, heads, head_dim)
        # -> (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.o_proj(out)
    
    
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(
            d_model,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            context_length=context_length,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )

        self.ffn_norm = RMSNorm(
            d_model,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLUFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
    
    
    
class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig, device=None, dtype=None):
        super().__init__()

        self.config = config

        self.token_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            device=device,
            dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    context_length=config.context_length,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            config.d_model,
            config.vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"seq_len={seq_len} exceeds context_length={self.config.context_length}"
            )

        x = self.token_embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits
    
    
    
    
if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=10000,
        context_length=256,
        d_model=384,
        num_layers=6,
        num_heads=6,
        d_ff=1536,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerLM(config, device=device).to(device)

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(4, 128),
        device=device,
    )

    logits = model(x)

    print("input shape:", x.shape)
    print("logits shape:", logits.shape)