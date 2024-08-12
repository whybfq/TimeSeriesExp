import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)
        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        if args.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x

class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        else:
            return x
        return x

class TSLANet(nn.Module):
    def __init__(self, args, task_data_config_list):
        super(TSLANet, self).__init__()
        self.args = args
        self.task_data_config_list = task_data_config_list
        self.seq_len = args.seq_len
        self.patch_len = args.patch_len
        self.stride = self.patch_len // 2
        num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)

        self.input_layer = nn.Linear(self.patch_len, configs.d_model)
        dpr = [x.item() for x in torch.linspace(0, configs.dropout, configs.e_layers)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)
        ])
        self.out_layer = nn.Linear(configs.d_model * num_patches, self.pred_len)

    def forward(self, x, x_mark=None):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = outputs * stdev
        outputs = outputs + means
        return outputs

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # For compatibility with the UNITS interface
        return self.forward(x_enc)

    def imputation(self, x_enc, x_mark_enc, mask):
        # Implement imputation if needed
        pass

    def anomaly_detection(self, x_enc, x_mark_enc):
        # Implement anomaly detection if needed
        pass

    def classification(self, x_enc, x_mark_enc):
        # Implement classification if needed
        pass

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, task_name='forecast'):
        if task_name == 'long_term_forecast' or task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, mask)
        elif task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, x_mark_enc)
        elif task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError(f"Unknown task name: {task_name}")