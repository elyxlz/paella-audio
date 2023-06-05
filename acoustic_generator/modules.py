from typing import Optional, Any, Literal
from torchtyping import TensorType

import torch
import torch.nn as nn
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm

from .attend import Attend


def exists(val: Any) -> bool:
    return val is not None

def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
    
    
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)
    

class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.gamma


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        hidden_size,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = hidden_size * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Conv1d(hidden_size, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, hidden_size, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
  
    
class RotaryEmbedder(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: TensorType["n"],         # a 1-D Tensor of N indices, one per batch element.
        dim: int,                   # the dimension of the output.
        max_period: int = 10000,    # controls the minimum frequency of the embeddings.
    ) -> TensorType["n", "d"]:
        """
        Create sinusoidal timestep embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(
        self,
        t: TensorType["n"],
    ) -> TensorType["n", "d"]:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: bool = None,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            norm_layer: Optional[nn.Module] = None,
            bias: bool = True,
            drop: float = 0.,
            use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(
        self,
        x: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "c", "h", "w"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads = 8,
        dim_head = 64,
        qkv_bias = False,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.num_heads, exists(context)
        context = default(context, x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ConformerBlock(nn.Module):
    """
    A Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning, cross attention, and a conformer conv module.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        conv_causal: bool = False,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.context_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv = ConformerConvModule(
            hidden_size=hidden_size,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            )
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )

    def forward(self, x, c, context=None, rotary_emb=None):
        modulation_parts = self.adaLN_modulation(c).chunk(12, dim=1)
        shift_msa, scale_msa, gate_msa = modulation_parts[0:3]
        shift_xa, scale_xa, gate_xa = modulation_parts[3:6]
        shift_conv, scale_conv, gate_conv = modulation_parts[6:9]
        shift_mlp, scale_mlp, gate_mlp = modulation_parts[9:12]
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rotary_emb=rotary_emb)
        if exists(context):
            x = x + gate_xa.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_xa, scale_xa), context=self.context_norm(context))
        x = x + gate_conv.unsqueeze(1) * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm4(x), shift_mlp, scale_mlp))
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of Transformer.
    """
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: TensorType["batch", "tokens", "hidden_size"],
        c: TensorType["batch", "tokens", "hidden_size"],
    ) -> TensorType["batch", "tokens", "hidden_size"]:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x
    

class Conformer(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        embedding_features: Optional[int] = None,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ConformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size)
        
        if exists(embedding_features) and embedding_features != hidden_size:
            self.embedding_projection = nn.Linear(embedding_features, hidden_size)
        else:
            self.embedding_projection = nn.Identity()
            
        self.initialize_weights()
    
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
        
    def forward(
        self,
        x: TensorType["batch", "tokens", "hidden_size"],                                    # token embeddings
        features: TensorType["batch", "hidden_size"],                                       # conditioning features
        context: Optional[TensorType["batch", "channels", "embedding_features"]] = None,  # conditioning embeddings
        rotary_emb: Optional[TensorType["tokens", "head_dim"]] = None,                      # rotary embeddings
        **kwargs
    ) -> TensorType["batch", "tokens", "hidden_size"]:                                      # output embeddings
        
        num_tokens, device = x.shape[1], x.device

        if context is not None:
            context = self.embedding_projection(context)
                
        for block in self.blocks:
            x = block(x, c=features, context=context, rotary_emb=rotary_emb, **kwargs)
            
        out = self.final_layer(x[:, :num_tokens], features)
        return out


class AcousticGenerator(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        hidden_size: int = 512,
        num_heads: int= 8,
        depth: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.conformer = Conformer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            **kwargs,
        )
        
        self.num_quantizers = num_quantizers
        self.vocab_size = 1024 * num_quantizers
        self.token_embedder = nn.Embedding(self.vocab_size, hidden_size)
        self.pos_embedder = RotaryEmbedder(64) # attention dim
        
        self.timestep_embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            frequency_embedding_size=256,
        )
        
        self.quantizer_embedder = nn.Embedding(num_quantizers, hidden_size)
        
        self.to_logits = nn.Linear(hidden_size, self.vocab_size, bias=False)
                
        self.initialize_parameters()
        
        
    def initialize_parameters(
        self,
    ):
        # initialize token_embedder
        nn.init.xavier_uniform_(self.token_embedder.weight)
        
        # initialize timestep embedder
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        
        # initialize quantizer embedder
        nn.init.xavier_uniform_(self.quantizer_embedder.weight)
        
        nn.init.xavier_uniform_(self.to_logits.weight)
        if self.to_logits.bias is not None:
            nn.init.constant_(self.to_logits.bias, 0)
        
    # TODO: make q level batchable for stable training
    def forward(
        self,
        x: TensorType["batch", "num_quantizers", "seq_len"],
        q: Optional[int] = None, # what quantizer level is being processed
        time: Optional[TensorType["batch"]] = None,
        embedding: Optional[TensorType["batch", "channels", "embedding_features"]] = None,
        return_logits: bool = False,
    ):
        b, device = x.shape[0], x.device
        seq_len = x.shape[-1]
                
        if not exists(time):
            time = torch.rand(b, device=device)
            
        if not exists(q):
            # logarithmically select q = 1 from log uniform distribution
            #distribution_p = lambda x: 0.04 * torch.exp(-11. * (x - 1) / (x + 100))

            # Sample the distribution values for x from 1 to q
            #probabilities = distribution_p(torch.arange(1, self.num_quantizers + 1))
            #probabilities[0] = 0.95
            probabilities = torch.Tensor([0.5, 0.175, 0.175, 0.175])
            probabilities = [0.5]
            probabilities.extend([0.175] * (self.num_quantizers - 1))
            probabilities = torch.Tensor(probabilities)

            categorical_dist = torch.distributions.Categorical(logits=probabilities.log())
            sample = categorical_dist.sample()
            q = sample.item() + 1
            
        x_q = x[:, :q, :].clone()
        # expand matrix, improve with broadcasting
        rvq_matrix = torch.cat([torch.zeros(b, 1, seq_len).long() + 1024 * i for i in range(q)], dim=1).to(device)
        x_q += rvq_matrix #(b, q, s)
        
        x_noised = x_q.clone()
        # add noise to q
        if self.training:
            x_noised = torch.cat([
                x_noised[:, :q - 1, :], self.add_noise(x_noised[:, q - 1, :].unsqueeze(1), time)[0],
            ], dim=1)
            
        # make sure max token isn't larger than vocab size
        token_embeds = self.token_embedder(x_noised)
                
        # sum over quantizers to reduce sequence length
        token_embeds = reduce(token_embeds, "b q s h -> b s h", "sum", q=q)
        
        # pos emb
        rotary_embeds = self.pos_embedder(x_noised.shape[-1])
        
        # time emb
        time_embeds = self.timestep_embedder(time)
        
        # quantizer emb
        quantizer_embeds = self.quantizer_embedder(torch.Tensor([q - 1]).long().to(device))
                
        features = time_embeds + quantizer_embeds
        
        # dont use cross attention for highly correlated quantizers
        if q > 2:
            embedding = None 
            
        out = self.conformer(
            token_embeds,
            context=embedding,
            features=features,
            rotary_emb=rotary_embeds
        )
        
        # get logits for current quantizer
        logits = self.to_logits(out)[..., (q-1) * 1024:q * 1024]
                
        if return_logits: 
            return logits
        
        # calculates loss only on current quantizer
        pred = rearrange(logits, "b s v -> (b s) v")
        true = rearrange(x[:, q - 1, :], "b s -> (b s)")
        loss_full = F.cross_entropy(pred, true, reduction="none")
        loss = loss_full.mean()
        return loss
    
    
    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        steps_scheduler: list[int],
        embedding: Optional[TensorType["batch", "channels", "embedding_features"]] = None,
        temperature: tuple = (0.7, 0.3),
        batch_size: int = 1,
        mode: str = Literal["multinomial", "argmax"],
        device: str = "cuda",
        **kwargs
    ):
        self.eval()
        msg = f"steps scheduler must have length {self.num_quantizers}, got {len(steps_scheduler)}"
        assert len(steps_scheduler) == self.num_quantizers, msg
        
        steps = steps_scheduler[0]
    
        x_noisy = torch.randint(0, 1024, (batch_size, self.num_quantizers, seq_len), device=device)
                
        for q, steps in enumerate(steps_scheduler):
            q += 1
            
            timesteps = torch.linspace(1, 0, steps, device=device)
            temperatures = torch.linspace(temperature[0], temperature[1], steps, device=device)
            
            progress_bar = tqdm(timesteps.cpu().numpy())
            for j, t in enumerate(progress_bar):
                progress_bar.set_description(f"step: {j}, time: {t}, quantizer: {q}")
                
                time = torch.ones(batch_size, device=device) * t
                
                q_logits = self.forward(
                    x_noisy,
                    q=q,
                    time=time,
                    embedding=embedding,
                    return_logits=True,
                    **kwargs
                )
                q_scores = q_logits.div(temperatures[j]).softmax(dim=-1)
                
                if mode == 'multinomial':
                    q_scores = rearrange(q_scores, "b s v -> (b s) v")
                    q_sampled = torch.multinomial(q_scores, 1)
                    q_sampled = rearrange(q_sampled, "(b s) 1 -> b s", b=batch_size, s=seq_len)
                elif mode == 'argmax':
                    q_sampled = q_scores.argmax(dim=-1)
                else:
                    raise ValueError(f"mode {mode} not supported")
                
                x_noisy[:, q - 1, :] = q_sampled
                if j == steps - 1:
                    break
                
                t_next = torch.ones(batch_size, device=device) * timesteps[j+1]
                
                # add noise to last quantizer
                x_noisy[:, q - 1, :] = self.add_noise(x_noisy[:, q - 1, :], t_next)[0]
        
        return x_noisy
    
    def add_noise(self, x, t, mask=None, random_x=None):
        if mask is None:
            mask = (torch.rand_like(x.float()) <= t[:, None, None]).long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, 1024)
        x = x * (1 - mask) + random_x * mask
        return x, mask