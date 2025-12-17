import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN  # 请确保该文件在您的路径中

class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = False
        if "no_gate" in self.hparams: self._no_gate = self.hparams["no_gate"]
        self._minimal = False
        if "minimal" in self.hparams: self._minimal = self.hparams["minimal"]

        # 激活函数选择
        if self.hparams["backbone_activation"] == "silu":
            backbone_activation = nn.SiLU
        elif self.hparams["backbone_activation"] == "relu":
            backbone_activation = nn.ReLU
        elif self.hparams["backbone_activation"] == "tanh":
            backbone_activation = nn.Tanh
        elif self.hparams["backbone_activation"] == "gelu":
            backbone_activation = nn.GELU
        elif self.hparams["backbone_activation"] == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError("Unknown activation")

        layer_list = [nn.Linear(input_size + hidden_size, self.hparams["backbone_units"]), backbone_activation()]
        for i in range(1, self.hparams["backbone_layers"]):
            layer_list.append(nn.Linear(self.hparams["backbone_units"], self.hparams["backbone_units"]))
            layer_list.append(backbone_activation())
            if "backbone_dr" in self.hparams.keys(): layer_list.append(torch.nn.Dropout(self.hparams["backbone_dr"]))
        self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams["backbone_units"], hidden_size)

        if self._minimal:
            self.w_tau = torch.nn.Parameter(data=torch.zeros(1, self.hidden_size), requires_grad=True)
            self.A = torch.nn.Parameter(data=torch.ones(1, self.hidden_size), requires_grad=True)
        else:
            self.ff2 = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_a = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_b = nn.Linear(self.hparams["backbone_units"], hidden_size)
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.get("init")
        if init_gain is not None:
            for w in self.parameters():
                if w.dim() == 2:
                    torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):
        batch_size = input.size(0)
        ts = ts.view(batch_size, 1)
        w_ts = torch.exp(ts * (1 - 2 * torch.log(ts)))

        x = torch.cat([input, hx], -1)
        x = self.backbone(x)

        if self._minimal:
            ff1 = self.ff1(x)
            new_hidden = (-self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1))) * ff1 + self.A)
        else:
            ff1 = self.tanh(self.ff1(x))
            ff2 = self.tanh(self.ff2(x))
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * w_ts + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden

class MultiScaleSeasonalTrendDecomposition(nn.Module):


    def __init__(self, kernel_size, enc_in):
        super(MultiScaleSeasonalTrendDecomposition, self).__init__()

        k1 = kernel_size
        k2 = max(3, kernel_size // 2)

        if k2 % 2 == 0: k2 += 1

        self.kernels = [k1, k2] if k1 > 4 else [k1]

        self.avg_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=0)
            for k in self.kernels
        ])

    def forward(self, x):

        trend_sum = 0

        for i, avg_pool in enumerate(self.avg_pools):
            k = self.kernels[i]

            pad_left = (k - 1) // 2
            pad_right = k - 1 - pad_left

            if pad_left + pad_right < x.shape[-1]:
                x_padded = F.pad(x, (pad_left, pad_right), mode='reflect')
            else:
                x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')

            trend_sum += avg_pool(x_padded)

        final_trend = trend_sum / len(self.kernels)

        return final_trend


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.dropout_val = configs.dropout
        self.hidden_size = configs.hidden_size

        self.Q_chan_indep = getattr(configs, 'Q_chan_indep', False)

        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)

        self.moving_avg = configs.moving_avg
        self.decomposition = MultiScaleSeasonalTrendDecomposition(
            kernel_size=self.moving_avg,
            enc_in=self.enc_in
        )

        # --- 流 1: OTL (长期/趋势流) ---
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        q_mat_dir = getattr(configs, 'Q_MAT_file', None) if self.Q_chan_indep else getattr(configs, 'q_mat_file',
                                                                                           'q_mat.npy')
        if not os.path.isfile(q_mat_dir): q_mat_dir = os.path.join(configs.root_path, q_mat_dir)
        self.Q_mat = torch.from_numpy(np.load(q_mat_dir)).to(torch.float32).to(device)

        q_out_mat_dir = getattr(configs, 'Q_OUT_MAT_file', None) if self.Q_chan_indep else getattr(configs,
                                                                                                   'q_out_mat_file',
                                                                                                   'q_out_mat.npy')
        if not os.path.isfile(q_out_mat_dir): q_out_mat_dir = os.path.join(configs.root_path, q_out_mat_dir)
        self.Q_out_mat = torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32).to(device)

        self.ortho_linear_map = nn.Linear(self.seq_len, self.pred_len)
        self.delta1 = nn.Parameter(torch.zeros(1, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, 1, self.pred_len))

        # --- 流 2: Lite-CFC (短期/季节流) ---
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.predict_rnn = nn.Sequential(
            nn.Dropout(self.dropout_val),
            nn.Linear(self.d_model, self.seg_len)
        )
        default_cfc_hparams = {
            "backbone_activation": "gelu", "backbone_units": 128,
            "backbone_layers": 1, "backbone_dr": 0.1, "init": 1.0,
            "no_gate": False, "minimal": False
        }
        self.cfc_hparams = getattr(configs, 'cfc_hparams', default_cfc_hparams)
        self.rnn_cell = CfcCell(
            input_size=self.d_model,
            hidden_size=self.d_model,
            hparams=self.cfc_hparams
        )

        self.projection = nn.Linear(self.enc_in, self.c_out)

    def Ortho_Trans(self, x):
        B, N, T = x.shape
        assert T == self.seq_len
        if self.Q_chan_indep:
            x_trans = torch.einsum('bnt,ntv->bnv', x, self.Q_mat.transpose(-1, -2))
        else:
            x_trans = torch.einsum('bnt,tv->bnv', x, self.Q_mat.transpose(-1, -2)) + self.delta1
        x_trans = self.ortho_linear_map(x_trans)
        if self.Q_chan_indep:
            x_out = torch.einsum('bnt,ntv->bnv', x_trans, self.Q_out_mat)
        else:
            x_out = torch.einsum('bnt,tv->bnv', x_trans, self.Q_out_mat) + self.delta2
        return x_out

    def forward(self, x_enc):

        if x_enc.dim() == 4: x_enc = x_enc.squeeze(1)
        try:
            B, N, T = x_enc.shape
            x_enc = x_enc.transpose(1, 2)
        except ValueError as e:
            x_enc = x_enc.transpose(1, 2)
            B, T, N = x_enc.shape
        assert T == self.seq_len
        assert N == self.enc_in

        x_norm = self.revin_layer(x_enc, 'norm')  # [B, T, N]

        x_norm_transposed = x_norm.transpose(1, 2)

        x_trend_transposed = self.decomposition(x_norm_transposed)

        x_trend = x_trend_transposed.transpose(1, 2)
        x_seasonal = x_norm - x_trend

        z = self.Ortho_Trans(x_trend.transpose(1, 2))  # [B, N, T_pred]
        z = z.transpose(1, 2)  # [B, T_pred, N]

        seq_last = x_seasonal[:, -1:, :].detach()
        x_rnn = (x_seasonal - seq_last)  # [B, T, N]

        x_rnn = x_rnn.permute(0, 2, 1)  # [B, N, T]
        x_rnn = self.valueEmbedding(x_rnn.reshape(-1, self.seg_num_x, self.seg_len))

        B_N, T_x, D = x_rnn.shape
        hx = torch.zeros(B_N, self.d_model, device=x_rnn.device)
        ts_dummy_enc = torch.tensor(1.0, device=x_rnn.device).expand(B_N)
        for t in range(T_x):
            input_t = x_rnn[:, t, :]
            hx = self.rnn_cell(input_t, hx, ts_dummy_enc)
        hn = hx.unsqueeze(0)

        # RNN 解码器
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(B, 1, 1)
        hx_dec = hn.squeeze(0)
        dec_input = pos_emb.reshape(B, N, self.seg_num_y, self.d_model)
        ts_dummy_dec = torch.tensor(1.0, device=x_rnn.device).expand(B * N)
        output_patches = []
        for t in range(self.seg_num_y):
            input_t = dec_input[:, :, t, :].reshape(B_N, D)
            hx_dec = self.rnn_cell(input_t, hx_dec, ts_dummy_dec)
            output_patches.append(hx_dec)
        hy_seq = torch.stack(output_patches, dim=0)
        hy_flat = hy_seq.permute(1, 0, 2).reshape(-1, self.d_model)
        y = self.predict_rnn(hy_flat)
        y = y.view(-1, self.enc_in, self.pred_len)

        y = y.permute(0, 2, 1) + seq_last  # [B, T_pred, N]

        out = z + y
        out = self.projection(out)  # [B, T_pred, c_out]
        out = self.revin_layer(out, 'denorm')

        return out