import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_configs import *

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

#Create Mamba block
def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
        return block

#1D-Conv for sequence
class Conv1d4nonverbal(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, dropout):
        super(Conv1d4nonverbal, self).__init__()
        
        self.conv1dLayer = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BatchNorm = nn.BatchNorm1d(out_dim)
        self.LayerNorm = nn.LayerNorm(out_dim)
        self.ffn = PositionWiseFeedForward(out_dim, dropout=dropout)
    def forward(self, x):
        y = x
        shapeOne = x.shape[0]
        shapeTwo = x.shape[1]
        shapeThree = x.shape[2]
        x = x.permute(0, 2, 1)
        x = self.conv1dLayer(x)
        x = x.permute(0, 2, 1)
        # x = x.reshape(-1, shapeThree)
        x = self.LayerNorm(x)
        # x = x.reshape(shapeOne, shapeTwo, shapeThree)
        # x = self.ffn(x)
        return x

#Implementation for the MSG in paper “Dynamically Shifting Multimodal Representations via Hybrid-Modal Attention for Multimodal Sentiment Analysis”
class MultiModalShiftGate(nn.Module):
    def __init__(self, dim, mu = 0.5, ep = 1e-7):
        super(MultiModalShiftGate, self).__init__()
        self.proj = nn.Linear(2 * dim, dim)
        self.mu = nn.Parameter(torch.tensor([mu]))
        self.ep = ep
    def forward(self, t, a, v):
        fused = torch.cat((a, v), dim=-1)
        fused = self.proj(fused)
        norm_Ft = torch.norm(t, p=2, dim=2, keepdim=True)
        norm_hs = torch.norm(fused, p=2, dim=2, keepdim=True)
        eta = norm_Ft * self.mu / (norm_hs + self.ep)
        eta = torch.min(eta, torch.ones_like(eta))
        adjuested_rep = eta * fused
        return t + adjuested_rep
    
#Variation for the MSG in paper “Dynamically Shifting Multimodal Representations via Hybrid-Modal Attention for Multimodal Sentiment Analysis”
class MultiModalShiftGateBi(nn.Module):
    def __init__(self, dim, mu = 0.5, ep = 1e-7):
        super(MultiModalShiftGateBi, self).__init__()
        self.mu = nn.Parameter(torch.tensor([mu]))
        self.ep = ep
    def forward(self, t, a):
        fused = a
        norm_Ft = torch.norm(t, p=2, dim=2, keepdim=True)
        norm_hs = torch.norm(fused, p=2, dim=2, keepdim=True)
        eta = norm_Ft * self.mu / (norm_hs + self.ep)
        eta = torch.min(eta, torch.ones_like(eta))
        adjuested_rep = eta * fused
        return t + adjuested_rep
    

#BiGRU
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiGRUModel, self).__init__()
        # 定义双向 GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        # x 的形状为 (batch_size, sequence_length, input_size)
        output, _ = self.gru(x)
        return output

#BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BiLSTM, self).__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.bilstm = nn.LSTM(in_dim, out_dim, batch_first=True, bidirectional=True, bias=False)

    def forward(self, x):
        # out1 = self.layernorm(x)
        out, _ = self.bilstm(x)
        return out

#Create BiMambaBlocks
class NewMambaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_layer=1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm=True,
        residual_in_fp32=True,
        bidirectional=True,
        device=DEVICE,
        dtype=torch.float32,
        dropout=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NewMambaBlock, self).__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        
        self.forward_blocks = nn.ModuleList(
            [
                create_block(
                    in_channels,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    device=device,
                    dtype=dtype
                )
                for i in range(n_layer)
            ]
        )
        if bidirectional:
            self.backward_blocks = nn.ModuleList(
                [
                    create_block(
                        in_channels,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=1e-5,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                        device=device,
                        dtype=dtype
                    )
                    for i in range(n_layer)
                ]
            )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            in_channels, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(partial(_init_weights, n_layer=n_layer))
        self.ffn = PositionWiseFeedForward(in_channels, dropout)
    
    def forward(self, input, inference_params=None):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=inference_params)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

        if not self.fused_add_norm:
            f_residual = (forward_f + for_residual) if for_residual is not None else forward_f
            b_residual = (backward_f + back_residual) if back_residual is not None else backward_f
            for_hidden_states = self.norm_f(f_residual.to(dtype=self.norm_f.weight.dtype))
            back_hidden_states = self.norm_f(b_residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            for_hidden_states = fused_add_norm_fn(
                forward_f,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=for_residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            back_hidden_states = fused_add_norm_fn(
                backward_f,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=back_residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        backback = torch.flip(back_hidden_states, [1])
        hidden_states = for_hidden_states + backback
        residual = self.ffn(residual)

        return hidden_states
    
#Class of Mamba block
class MambaBlock(nn.Module):
    def __init__(self, in_channels, out_dim, n_layer=1, bidirectional=True):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                        Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )
        else: self.backward_blocks = None

        self.apply(partial(_init_weights, n_layer=n_layer))
        self.proj_0 = nn.Linear(in_channels * 2, out_dim)

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
            residual = self.proj_0(residual)
        
        return residual

#Multi-head Self Attention
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=8):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, embedding):
        Q = self.Wq(embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

#Multi-head Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, hidden_size, head_num=8):
        super(CrossAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(TEXT_DIM, 1)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

#Feedforward module
class PositionWiseFeedForward(nn.Module):

    """
    w2(relu(w1(layer_norm(x))+b1))+b2
    """

    def __init__(self, TEXT_DIM, dropout=None):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(TEXT_DIM, 64)
        self.w_2 = nn.Linear(64, TEXT_DIM)
        self.layer_norm = nn.LayerNorm(TEXT_DIM, eps=1e-5)
        self.dropout_1 = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output

#Feedforward model with assigned dimension
class PositionWiseFeedForwardModified(nn.Module):

    """
    w2(relu(w1(layer_norm(x))+b1))+b2
    """

    def __init__(self, in_dim, inter_dim, out_dim, dropout=None):
        super(PositionWiseFeedForwardModified, self).__init__()
        self.w_1 = nn.Linear(in_dim, inter_dim)
        self.w_2 = nn.Linear(inter_dim, out_dim)
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output
    
    
#Fusion gate with feedforward
class GatedMultimodalLayerWithFFN(nn.Module):
    
    def __init__(self, size_in1, size_in2, dropout, size_out=32):
        super(GatedMultimodalLayerWithFFN, self).__init__()
        # self.hidden1 = PositionWiseFeedForwardModified(size_in1, 64, size_out, dropout)
        # self.hidden2 = PositionWiseFeedForwardModified(size_in2, 64, size_out, dropout)
        # self.hidden_sigmoid = nn.Linear(size_out*2, 1)
        self.hidden_sigmoid = nn.Linear(size_in1*2, 1)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        # h1 = self.tanh_f(self.hidden1(x1))
        # h2 = self.tanh_f(self.hidden2(x2))
        h1 = self.tanh_f(x1)
        h2 = self.tanh_f(x2)
        x = torch.cat((x1, x2), dim=-1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        z = z.expand_as(x1)
        h = z * h1 + (1 - z) * h2
        return h

#Fusion gate
class GatedMultimodalLayer(nn.Module):
    
    def __init__(self, size_in1, size_in2, size_out=32):
        super(GatedMultimodalLayer, self).__init__()
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out*2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        x = torch.cat((h1, h2), dim=-1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        z = z.expand_as(x1)
        h = z * x1 + (1 - z) * x2
        return h

#Posiembedding
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # def forward(self, x, mask):
    def forward(self, x):
        """
        Args:
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        # assert mask is not None
        mask = torch.ones(x.shape[0], 50, dtype=torch.bool).to(torch.device("cuda:0"))
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # import pdb; pdb.set_trace()
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2).int() / self.num_pos_feats)


        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # (bsz, L, num_pos_feats*2)
        # import ipdb; ipdb.set_trace()
        return pos_x  # .permute(0, 2, 1)  # (bsz, num_pos_feats*2, L)

#CFM
class CrossFusionModule(nn.Module):
    """
    crossmodal fusion module: to get crossmodal attention and to return the fused feature.
    The cross attention is calculated by the output embedding from each encoder layers of audio and visual modalities.
    """
    def __init__(self, dim=256):
        super(CrossFusionModule, self).__init__()

        # linear project + norm + corr + concat + conv_layer + tanh
        self.project_audio = nn.Linear(768, dim)  # linear projection
        self.project_vision = nn.Linear(768, dim)
        self.corr_weights = torch.nn.Parameter(torch.empty(
            dim, dim, requires_grad=True).type(torch.cuda.FloatTensor))
        nn.init.xavier_normal_(self.corr_weights)
        self.project_bottleneck = nn.Sequential(nn.Linear(dim * 2, 768),
                                                nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
                                                nn.ReLU())
    def forward(self, audio_feat, visual_feat):
        """

        :param audio_feat: [batchsize 64 768]
        :param visual_feat:[batchsize 64 768]
        :return: fused feature
        """
        audio_feat = self.project_audio(audio_feat)
        visual_feat = self.project_vision(visual_feat)

        visual_feat = visual_feat.transpose(1, 2)  # 768, 64

        a1 = torch.matmul(audio_feat, self.corr_weights)  # 768, 768
        cc_mat = torch.bmm(a1, visual_feat)  # 64*64

        audio_att = F.softmax(cc_mat, dim=1)
        visual_att = F.softmax(cc_mat.transpose(1, 2), dim=1)
        atten_audiofeatures = torch.bmm(audio_feat.transpose(1, 2), audio_att)
        atten_visualfeatures = torch.bmm(visual_feat, visual_att)
        atten_audiofeatures = atten_audiofeatures + audio_feat.transpose(1, 2)
        atten_visualfeatures = atten_visualfeatures + visual_feat  # 256, 64

        fused_features = self.project_bottleneck(torch.cat((atten_audiofeatures,
                                                            atten_visualfeatures), dim=1).transpose(1, 2))

        return fused_features

#Grouped gated fusion
class GGF(nn.Module):
    # 分组门控融合机制
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super(GGF, self).__init__()
        self.Wa = nn.Linear(input_dim, intermediate_dim)
        self.Wv = nn.Linear(input_dim, intermediate_dim)
        self.Wav = nn.Linear(input_dim, intermediate_dim)
        self.Wva = nn.Linear(input_dim, intermediate_dim)

        self.Wz_s = nn.Linear(input_dim * 2, intermediate_dim)
        self.Wz_m = nn.Linear(input_dim * 2, intermediate_dim)
        self.proj_o = nn.Linear(intermediate_dim, output_dim)

    def forward(self, xa, xv, xav, xva):
        h_a = torch.tanh(self.Wa(xa))
        h_v = torch.tanh(self.Wv(xv))
        h_av = torch.tanh(self.Wav(xav))
        h_va = torch.tanh(self.Wva(xva))

        s_concat = torch.cat([xa, xv], dim=2)
        m_concat = torch.cat([xav, xva], dim=2)

        p_s = torch.sigmoid(self.Wz_s(s_concat))
        p_m = torch.sigmoid(self.Wz_m(m_concat))

        h = p_s * h_a + (1 - p_s) * h_v + p_m * h_av + (1 - p_m) * h_va
        result = self.proj_o(h)
        return result
