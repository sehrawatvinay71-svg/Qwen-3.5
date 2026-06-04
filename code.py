import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self,dim: int,eps=1e-6):
        super().__init__()
        self.weight = nn.parameter(torch.zeros(dim))
        self.eps = eps
    
    def _norm(self,x: torch.Tensor):
        x = x * (torch.sqrt(x.pow(2).mean(-1, keepdim=True).float()) + self.eps)
        return x

    def forward(self, x: torch.Tensor):
        out = self._norm(x)
        out = out * (1.0 + self.weight.float())
        return out
    

class RoPE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dim = config.head_dim
        self.pos_rotary_factor = 1.0
        self.theta = config.theta * self.pos_rotary_factor

        self.inv_freq = 1.0 / (
            self.theta ** torch.arange(0, self.dim, 2) / self.dim
        )
    
    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor):

        # x: [batch, num_heads, seq_len, head_dim]
        # pos_ids : [batch, seq_len]

        # [3, batch, seq_len]
        pos_ids = pos_ids[None, ...].expand(
            3,
            pos_ids.shape[0],
            -1
        )

        # [3, batch, dim // 2, 1]
        inv_freq = self.inv_freq[None, None, : ,None].expand(
            3,
            pos_ids.shape[1],
            -1,
            1
        )

        # [3. batch, dim // 2, 1] x [3, batch, 1, seq_len] = [3, batch, dim // 2, seq_len]
        freq = inv_freq @ pos_ids[:, :, None, :]
        freq = freq.transpose(2, 3) # [3, batch, seq_len, dim // 2]

        freq = apply_interleaved_mrope(freq) # [batch, seq_len, dim // 2]
        torch.cat((freq, freq), dim=-1) # [batch, seq_len, dim]

        # [batch, seq_len, dim]
        return embd.cos().to(dtype=x.dtype), embd.sin().to(dtype = x.dtype)
    
def rotate_half(x: torch.Tensor):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2: ]
    return torch.cat((-x2,x1), dim=-1)

def apply_rotary_pos_emd(q, k, cos, sin):
    # q: [batch_size, num_attn_heads, seq_len, head_dim]
    # cos: [batch_size, seq_len, pos_dim]

    cos = cos.unsqueeze(1) # [batch_size, 1, seq_len, pos_dim]
    sin = cos.unsqueeze(1) # [batch_size, 1, seq_len, pos_dim]

    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_emebd = (q_rot*cos) + (rotate_half(q_rot)*sin)
    k_emebd = (k_rot*cos) + (rotate_half(k_rot)*sin)

    q = torch.cat((q_emebd,q_pass),dim=-1)
    k = torch.cat((k_emebd,k_pass),dim=-1)

    return q,k



class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.kv_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias = config.attention_bias
        )

        self.k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias = config.attention_bias)
        self.v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias = config.attention_bias)

        self.proj_out = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.scaling = self.head_dim ** -0.5

    def forward(
            self,
            hidden_states: torch.Tensor,
            pos_embeddings: tuple,
            attention_mask: torch.Tensor |  None = None,
            cache = None    
        ):
            # hidden_states: [batch, seq_len, hidden_size]
            # pos_embeddings: [batch, seq_len, pos_dim]
            # attention_mask/causal: [batch, num_heads, query_length, kv_length]

            batch_size, seq_len, _ = self.q_proj(hidden_states)

            q_proj = self.q_proj(hidden_states)
            q, gate = torch.chunk(q_proj, 2, dim=-1)
            gate = gate.reshape(batch_size, seq_len, -1)

            # [batch_size, num_attn_heads, seq_len, head_dim]
            q = self.q_norm(q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)).transpose(1,2)
            k = self.k_norm(self.k(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)).transpose(1,2)
            v = self.v(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
            
            cos,sin = pos_embeddings
            q,k = apply_rotary_pos_emd(q, k, cos, sin)

            if cache is not None:
                k, v = cache.update(k,v)

            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

            #compute attention
            attn_weights = torch.matmul(q, k.transpose(2,3))
            attn_weights = attn_weights * self.scaling

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_weights, v) # [ batch, num_heads, seq_len dim]
            attn_out = attn_out.transpose(2,3) # [batch, seq_len, num_heads, dim]

            attn_out = attn_out.reshape(batch_size, seq_len, -1) # [batch, seq, hidden_size]
            attn_out = attn_out * F.sigmoid(gate)

            out = self.proj_out(attn_out)
            return out


            

class GatedDeltaNet(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim

        self.kernel_size = config.linear_conv_kernel_size

        self.k_dim = self.num_k_heads * self.head_k_dim
        self.v_dim = self.num_k_heads * self.head_v_dim

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.conv_dim = self.k_dim * 2 + self.v_dim

        self.qkv = nn.Linear(
            self.hidden_size,
            self.conv_dim
        )

        self.conv1d = nn.Conv1d(
            in_channels = self.conv_dim,
            out_channels = self.conv_dim,
            kernel_size = self.kernel_size,
            padding = self.kernel_size -1,
            groups = self.conv_dim,
            bias = False
        )

        self.out_proj = nn.Linear(
            self.v_dim,
            self.hidden_size,
            bias=False
        )

class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()

class DynamicCache(nn.Module):
    def __init__(self,config):
        super().__init__()

class TextModel(nn.Module):
    def __init__(self,config):
        super().__init__()

if __name__=="__main__":
    norm  = RMSNorm(64)
    out = norm(torch.randn(4, 20, 64))
    print(out.shape)
