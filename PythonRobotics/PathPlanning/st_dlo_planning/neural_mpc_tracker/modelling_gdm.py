import torch.nn as nn
import torch
from torch.nn import functional as F
import math
from einops import reduce, rearrange
from einops.layers.torch import Rearrange
from st_dlo_planning.neural_mpc_tracker.configuration_gdm import GDM_CFG, RNN_GDM_CFG, CONV_GDM_CFG


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. 
        PyTorch doesn't support simply bias=False 
    """

    def __init__(self, ndim, bias: bool=True):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
    
class MLP(nn.Module):
    def __init__(self, embed_dim:int):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 2 * embed_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(2 * embed_dim, embed_dim)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class MultiheadAttention(nn.Module):
    '''
        multi-head attention module
    '''
    def __init__(self, embed_dim, nhead, dropout: float=0.0, bias: bool = True):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % nhead == 0
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        
        # q, k, v projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        
        # regularization
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        
        # optimize attention computation
        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, q, k, v, attn_mask=None):
        '''
            attn_mask: position with 0 will be masked out in attention weight matrix
        '''
        batch_size, q_seq_len, embed_dim = q.size()
        k_seq_len = k.shape[1]
        v_seq_len = v.shape[1]        
        # project query, key, and value
        Q = self.q_proj(q)  # (batch_size, q_seq_length, embed_dim)
        K = self.k_proj(k)  # (batch_size, k_seq_length, embed_dim)
        V = self.v_proj(v)  # (batch_size, v_seq_length, embed_dim)
        
        # Split into heads and reshape for scaled_dot_product_attention
        Q = Q.view(batch_size, q_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, v_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        # (batch_size, num_heads, seq_length, head_dim)
        
        if self.use_flash:
            # using Flash Attention (FA) implementation
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(attn_mask == 0., float('-inf'))
            y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, 
                                                                 attn_mask=attn_mask,
                                                                 dropout_p=self.dropout if self.training else 0)
            
        else:
            # using Native Attention (NA) implementation
            att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ V # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y = y.transpose(1, 2).contiguous().view(batch_size, q_seq_len, embed_dim)
        output = self.out_proj(y)
        return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim:int, nhead:int, dropout: float=0., bias: bool=True):
        super().__init__()
        assert embed_dim % nhead == 0
        self.attn_layer = MultiheadAttention(embed_dim, nhead, dropout, bias)
        
    def forward(self, x):
        y = self.attn_layer(x, x, x)
        return y
    

class SABlock(nn.Module):
    '''
        self-attention (SA) block with pre-norm style
    '''
    def __init__(self, embed_dim, nhead):
        super(SABlock, self).__init__()
        self.ln_1 = LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(embed_dim, nhead)
        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class CABlock(nn.Module):
    '''
        cross-attention block
    '''
    def __init__(self, embed_dim, nhead):
        super(CABlock, self).__init__()
        self.ln_1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, nhead)
        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        
    def forward(self, q, k, v):
        q = self.ln_1(q)
        k = self.ln_1(k)
        v = self.ln_1(v)
        
        x = q + self.attn(q, k, v)
        x = x + self.mlp(self.ln_2(x))
        return x

class KeypointEmbedding(nn.Module):
    '''
        Keypoint embedding with conv1d + group_norm
    '''
    def __init__(self, 
                 in_channels:int = 2, 
                 embed_dim:int = 64,
                 kernel_size:int = 1,
                 ngroup:int = 4):
        super(KeypointEmbedding, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.conv1d = nn.Conv1d(in_channels=in_channels, 
                                out_channels=embed_dim, 
                                kernel_size=kernel_size, 
                                padding='same')
        self.gn = nn.GroupNorm(num_groups=ngroup, num_channels=embed_dim)

    def forward(self, keypoints: torch.Tensor):
        '''
            input:
                keypoints: [batch_size, num_keypoints, kp_dim]
        '''
        keypoints = keypoints.permute(0, 2, 1)
        conv_output = self.conv1d(keypoints)
        conv_output = self.gn(conv_output)
        kp_embedings = conv_output.permute(0, 2, 1)
        return kp_embedings


class DLOEncoder(nn.Module):
    def __init__(self, model_cfg: GDM_CFG):
        super(DLOEncoder, self).__init__()
        self.transformer = nn.ModuleDict(
            dict(
                kp_embed_layer = KeypointEmbedding(in_channels=model_cfg.kp_dim, 
                                                   embed_dim=model_cfg.embed_dim, 
                                                   kernel_size=model_cfg.conv1d_kernel_size,
                                                   ngroup=model_cfg.conv1d_ngroup),
                pe = nn.Embedding(model_cfg.max_kp, model_cfg.embed_dim),
                h = nn.ModuleList([SABlock(model_cfg.embed_dim, model_cfg.nhead) for _ in range(model_cfg.num_layers)]),
                ln_f = LayerNorm(model_cfg.embed_dim)
            )
        )
        
    def forward(self, x: torch.Tensor):
        '''
            x: in shape of [batch_size, seq_len, x_dim]
        '''
        seq_len = x.shape[1]
        x = self.transformer.kp_embed_layer(x)
        # add positional embedding
        pos = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        pos_emb = self.transformer.pe(pos)
        x = x + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x
    
    
class GDM(nn.Module):
    def __init__(self, model_cfg: GDM_CFG):
        '''
            global deformation model
        '''
        super(GDM, self).__init__()
        self.dlo_encoder = DLOEncoder(model_cfg)
        
        self.eef_proj = nn.Linear(model_cfg.eef_dim, model_cfg.embed_dim)
        self.delta_eef_proj = nn.Linear(model_cfg.delta_eef_dim, model_cfg.embed_dim)

        self.eef_pe = nn.Embedding(model_cfg.num_eef, model_cfg.embed_dim)
        
        self.ca_decoder = CABlock(model_cfg.embed_dim, model_cfg.nhead)
        
        self.delta_kp_head = nn.Sequential(nn.Linear(model_cfg.embed_dim, model_cfg.embed_dim // 2),
                                          nn.GELU(approximate='tanh'),
                                          nn.Linear(model_cfg.embed_dim // 2, model_cfg.kp_dim))
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000000.
        print(f"Total number of learnable parameters: {total_params}M")


    def local_transform(self, dlo_kp, eef_states):
        batch_size = dlo_kp.shape[0]
        
        dlo_kp_center = torch.mean(dlo_kp, dim=1, keepdim=True)
        local_dlo_kp = dlo_kp - dlo_kp_center

        tmp = torch.concatenate([dlo_kp_center, torch.zeros(batch_size, 1, 1, device=dlo_kp.device)], dim=2)
        local_eef_states = eef_states - tmp

        return local_dlo_kp, local_eef_states                                                  


    def forward(self, dlo_kp, eef_states, delta_eef_states):
        '''
            dlo_kp: in shape of (batch_size, kp_num, kp_dim)
            eef_states: in shape of (batch_size, eef_num, eef_dim)
            delta_eef_states: in shape of (batch_size, eef_num, delta_eef_dim)
        '''
        dlo_kp, eef_states = self.local_transform(dlo_kp, eef_states)
        q = self.dlo_encoder(dlo_kp)
        k = self.eef_proj(eef_states)
        # add positional embedding for eefs
        pos = torch.arange(0, k.shape[1], device=dlo_kp.device, dtype=torch.long).unsqueeze(0)
        eef_pos_emb = self.eef_pe(pos)
        k = k + 0.01 * eef_pos_emb

        v = self.delta_eef_proj(delta_eef_states)

        z = self.ca_decoder(q, k, v)
        predicts = self.delta_kp_head(z)
        return predicts
    

# ===================================== CNN1D ===========================
class Conv1dBlock(nn.Module):
    
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        '''
            Conv1d --> GroupNorm --> Mish (only change channels)
        '''
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    

class ConditionalResidualBlock1D(nn.Module):
    '''
        conditional residual block (the channel_dim could be changed)
    '''
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True):
        
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure channels compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out



class Conv_GDM(nn.Module):
    '''
        Conditional Conv1D based GDM
    '''
    def __init__(self, model_cfg: CONV_GDM_CFG):
        super(Conv_GDM, self).__init__()

        self.dlo_proj = nn.Linear(model_cfg.kp_dim, model_cfg.embed_dim)
        self.eef_proj = nn.Linear(model_cfg.eef_dim, model_cfg.embed_dim)
        self.delta_eef_proj = nn.Linear(model_cfg.delta_eef_dim, model_cfg.embed_dim)

        self.conv1d_1 = ConditionalResidualBlock1D(model_cfg.embed_dim, model_cfg.embed_dim, model_cfg.embed_dim*model_cfg.num_eef)
        self.conv1d_2 = ConditionalResidualBlock1D(model_cfg.embed_dim, model_cfg.embed_dim, model_cfg.embed_dim*model_cfg.num_eef)
        
        self.delta_kp_head = nn.Sequential(nn.Linear(model_cfg.embed_dim, model_cfg.embed_dim // 2),
                                           nn.GELU(approximate='tanh'),
                                           nn.Linear(model_cfg.embed_dim // 2, model_cfg.kp_dim))


        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000000.
        print(f"Total number of learnable parameters: {total_params}M")


    def local_transform(self, dlo_kp, eef_states):
        batch_size = dlo_kp.shape[0]
        
        dlo_kp_center = torch.mean(dlo_kp, dim=1, keepdim=True)
        local_dlo_kp = dlo_kp - dlo_kp_center

        tmp = torch.concatenate([dlo_kp_center, torch.zeros(batch_size, 1, 1, device=dlo_kp.device)], dim=2)
        local_eef_states = eef_states - tmp

        return local_dlo_kp, local_eef_states                                                  


    def forward(self, dlo_kp, eef_states, delta_eef_states):
        '''
            dlo_kp: in shape of (batch_size, kp_num, kp_dim)
            eef_states: in shape of (batch_size, eef_num, eef_dim)
            delta_eef_states: in shape of (batch_size, eef_num, delta_eef_dim)
        '''
        dlo_kp, eef_states = self.local_transform(dlo_kp, eef_states)

        kp_proj = self.dlo_proj(dlo_kp)
        eef_proj = self.eef_proj(eef_states)
        delta_eef_proj = self.delta_eef_proj(delta_eef_states)
        kp_proj = kp_proj.permute(0, 2, 1)

        eef_cond = eef_proj.reshape(eef_proj.shape[0], -1)
        delta_eef_cond = delta_eef_proj.reshape(delta_eef_proj.shape[0], -1)

        x = self.conv1d_1(kp_proj, eef_cond)
        x = self.conv1d_2(x, delta_eef_cond)
        x = x.permute(0, 2, 1)

        predicts = self.delta_kp_head(x)
        return predicts



class RNN_GDM(nn.Module):
    '''
        RNN based GDM
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def local_transform(self, dlo_kp, eef_states):
        batch_size = dlo_kp.shape[0]
        
        dlo_kp_center = torch.mean(dlo_kp, dim=1, keepdim=True)
        local_dlo_kp = dlo_kp - dlo_kp_center

        tmp = torch.concatenate([dlo_kp_center, torch.zeros(batch_size, 1, 1, device=dlo_kp.device)], dim=2)
        local_eef_states = eef_states - tmp

        return local_dlo_kp, local_eef_states                                                  


    def forward(self, dlo_kp, eef_states, delta_eef_states):
        '''
            dlo_kp: in shape of (batch_size, kp_num, kp_dim)
            eef_states: in shape of (batch_size, eef_num, eef_dim)
            delta_eef_states: in shape of (batch_size, eef_num, delta_eef_dim)
        '''
        dlo_kp, eef_states = self.local_transform(dlo_kp, eef_states)
        q = self.dlo_encoder(dlo_kp)
        k = self.eef_proj(eef_states)