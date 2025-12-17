import argparse


class BaseConfig:

    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'
        self.seq_len = 168
        self.label_len = 12
        self.pred_len = 24
        self.embed = 'timeF'
        self.activation = 'LeakyRelu'
        self.dropout = 0.3
        self.use_norm = 1
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.moving_avg = 24
        self.factor = 1


class DDOTCFCConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # === 基本任务参数 ===
        self.task_name = "long_term_forecast"
        self.seq_len = 168
        self.pred_len = 24
        self.enc_in = 12
        self.c_out = 3
        self.dropout = 0.3

        # === 共享模型参数 ===
        self.d_model = 16
        self.root_path = './data/'

        self.Q_chan_indep = False
        self.Q_MAT_file = None
        self.q_mat_file = 'q_mat.npy'
        self.Q_OUT_MAT_file = None
        self.q_out_mat_file = 'q_out_mat.npy'

        self.seg_len = 24

        self.hidden_size = 128

        self.cfc_hparams = {
            "backbone_activation": "gelu",
            "backbone_units": 128,
            "backbone_layers": 1,
            "backbone_dr": 0.2,
            "init": 1,
            "no_gate": False,
            "minimal": False
        }

        self.lr = 0.0011
        self.batchsize = 64
        self.epochs = 29
        self.weightdecay = 0.5
        self.decaypatience = 2

        self.individual = False


def get_config(model_name: str):
    if model_name == 'DDOTCFC':
        return DDOTCFCConfig()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

