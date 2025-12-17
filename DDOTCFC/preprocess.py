import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
# (不再需要 Transformer_EncDec)
import sys


class Model(nn.Module):
    """
    双流正交-动态网络 (DOCN)

    1. (长期流): OLinear  (无 Transformer)，使用 OrthoTrans + 线性映射 (模仿 DLinear)。
    2. (短期流): PRNN (Patching-based RNN)，使用 GRU 编解码器捕捉序列动态。
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # --- 核心配置 ---
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 输入通道数 (N)
        self.c_out = configs.c_out  # 输出通道数
        self.d_model = configs.d_model
        self.dropout_val = configs.dropout
        self.hidden_size = configs.hidden_size  # 用于融合层

        # (确保 configs 中有 Q_chan_indep, embed_size 等)
        self.Q_chan_indep = getattr(configs, 'Q_chan_indep', False)

        # --- 共享层 ---
        # RevIN  期望 (B, T, N)，它在最后一个维度(N)上归一化
        # (FESTRNN 原版 RevIN: affine=False, subtract_last=False)
        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)

        # --- 流 1: OLinear (长期/全局流) ---

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载输入正交矩阵 Q
        q_mat_dir = getattr(configs, 'Q_MAT_file', None) if self.Q_chan_indep else getattr(configs, 'q_mat_file',
                                                                                           'q_mat.npy')
        if not os.path.isfile(q_mat_dir):
            q_mat_dir = os.path.join(configs.root_path, q_mat_dir)
        self.Q_mat = torch.from_numpy(np.load(q_mat_dir)).to(torch.float32).to(device)

        # 加载输出正交矩阵 Q_out
        q_out_mat_dir = getattr(configs, 'Q_OUT_MAT_file', None) if self.Q_chan_indep else getattr(configs,
                                                                                                   'q_out_mat_file',
                                                                                                   'q_out_mat.npy')
        if not os.path.isfile(q_out_mat_dir):
            q_out_mat_dir = os.path.join(configs.root_path, q_out_mat_dir)
        self.Q_out_mat = torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32).to(device)

        # OLinear  的序列映射层 (DLinear 风格)
        # 直接在正交域中将 seq_len 映射到 pred_len
        self.ortho_linear_map = nn.Linear(self.seq_len, self.pred_len)

        # (移除 OLinear  的 embedding 和 ol_fc)
        self.delta1 = nn.Parameter(torch.zeros(1, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, 1, self.pred_len))

        # --- 流 2: PRNN (短期/动态流) ---
        # (来自您的 FESTRNN)
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict_rnn = nn.Sequential(
            nn.Dropout(self.dropout_val),
            nn.Linear(self.d_model, self.seg_len)
        )

        # --- 最终融合层 ---
        # (来自您的 FESTRNN, 适配 config)
        self.endlinear = nn.Linear(self.enc_in * 2, self.hidden_size)
        self.finallinear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.c_out)
        )

    # OLinear  的正交变换函数 (无 Transformer, 无 Embedding)
    def Fre_Trans(self, x):
        # x: [B, N, T_seq]
        B, N, T = x.shape
        assert T == self.seq_len

        # 1. 正交变换
        if self.Q_chan_indep:
            # (注意: Q_mat [N, T, T])
            x_trans = torch.einsum('bnt,ntv->bnv', x, self.Q_mat.transpose(-1, -2))
        else:
            # (注意: Q_mat [T, T])
            x_trans = torch.einsum('bnt,tv->bnv', x, self.Q_mat.transpose(-1, -2)) + self.delta1

        # 2. 线性映射 (DLinear 风格)
        # (B, N, T_seq) -> (B, N, T_pred)
        x_trans = self.ortho_linear_map(x_trans)

        # 3. 逆正交变换
        if self.Q_chan_indep:
            # (注意: Q_out_mat [N, T_pred, T_pred])
            x_out = torch.einsum('bnt,ntv->bnv', x_trans, self.Q_out_mat)
        else:
            # (注意: Q_out_mat [T_pred, T_pred])
            x_out = torch.einsum('bnt,tv->bnv', x_trans, self.Q_out_mat) + self.delta2

        return x_out  # [B, N, T_pred]

    def forward(self, x_enc):

        # ======================= 维度修复 =======================
        # 您的 main.py 传入的 x_enc 形状不一致

        # 1. 处理 4D 输入 (来自 evaluate)
        if x_enc.dim() == 4:
            x_enc = x_enc.squeeze(1)  # [B, 1, N, T] -> [B, N, T]

        # 2. 统一内部处理维度为 (B, T, N)
        # 您的框架提供 (B, N, T) = [B, 12, 168]
        # 模型的 RevIN 和 PRNN 都期望 (B, T, N) = [B, 168, 12]
        try:
            B, N, T = x_enc.shape
            x_enc = x_enc.transpose(1, 2)  # [B, N, T] -> [B, T, N]
        except ValueError as e:
            # 捕获 ptflops 第一次解包失败 (B,T,N = x_enc.shape)
            # 假设 ptflops 的 (1, 12, 168) 意为 (B, N, T)
            x_enc = x_enc.transpose(1, 2)
            B, T, N = x_enc.shape

        assert T == self.seq_len, f"模型T={self.seq_len}, 得到T={T}"
        assert N == self.enc_in, f"模型N={self.enc_in}, 得到N={N}"
        # ======================= 维度修复结束 =======================

        # x_enc 现在固定为 [B, T, N]

        # 1. 归一化 (共享)
        x_norm = self.revin_layer(x_enc, 'norm')  # [B, T, N]

        # --- 流 1: OLinear (长期/全局流) ---
        # Fre_Trans 期望 [B, N, T]
        z = self.Fre_Trans(x_norm.transpose(1, 2))  # [B, N, T_pred]
        z = z.transpose(1, 2)  # [B, T_pred, N]

        # --- 流 2: PRNN (短期/动态流) ---
        # PRNN 期望 [B, T, N]
        seq_last = x_norm[:, -1:, :].detach()  # [B, 1, N]
        x_rnn = (x_norm - seq_last)  # [B, T, N]

        # RNN 编码器
        x_rnn = x_rnn.permute(0, 2, 1)  # [B, N, T]
        x_rnn = self.valueEmbedding(x_rnn.reshape(-1, self.seg_num_x, self.seg_len))
        _, hn = self.rnn(x_rnn)

        # RNN 解码器
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(B, 1, 1)

        hn_dec = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        _, hy = self.rnn(pos_emb, hn_dec)

        y = self.predict_rnn(hy).view(-1, self.enc_in, self.pred_len)  # [B, N, T_pred]
        y = y.permute(0, 2, 1) + seq_last  # [B, T_pred, N]

        # --- 3. 融合与输出 ---
        fused_out = torch.cat((z, y), dim=-1)  # [B, T_pred, 2*N]

        out = self.endlinear(fused_out)  # [B, T_pred, hidden_size]
        out = self.finallinear(out)  # [B, T_pred, c_out]

        # 4. 反归一化
        # revin_layer denorm 期望 [B, T_pred, C_out]
        out = self.revin_layer(out, 'denorm')

        return out