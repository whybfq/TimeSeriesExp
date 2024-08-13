import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from models.UniTS import (CLSHead, PatchEmbedding, LearnablePositionalEmbedding,
                          DynamicLinear, ForecastHead, BasicBlock, F)


def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


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
        adaptive_mask = ((
                                     normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
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


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(nn.Module):
    def __init__(self, args, configs_list, pretrain=False):
        super(TSLANet, self).__init__()

        # Tokens settings
        self.num_task = len(configs_list)
        self.prompt_tokens = nn.ParameterDict({})
        self.mask_tokens = nn.ParameterDict({})
        self.cls_tokens = nn.ParameterDict({})
        self.category_tokens = nn.ParameterDict({})

        for i in range(self.num_task):
            dataset_name = configs_list[i][1]['dataset']
            task_data_name = configs_list[i][0]
            if dataset_name not in self.prompt_tokens:
                self.prompt_tokens[dataset_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], args.prompt_num, args.d_model)
                torch.nn.init.normal_(
                    self.prompt_tokens[dataset_name], std=.02)
                self.mask_tokens[dataset_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)

            if configs_list[i][1]['task_name'] == 'classification':
                self.category_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], configs_list[i][1]['num_class'], args.d_model)
                torch.nn.init.normal_(
                    self.category_tokens[task_data_name], std=.02)
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)
            if pretrain:
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)

        self.cls_nums = {}
        for i in range(self.num_task):
            task_data_name = configs_list[i][0]
            if configs_list[i][1]['task_name'] == 'classification':
                self.cls_nums[task_data_name] = configs_list[i][1]['num_class']
            elif configs_list[i][1]['task_name'] == 'long_term_forecast':
                remainder = configs_list[i][1]['seq_len'] % args.patch_len
                if remainder == 0:
                    padding = 0
                else:
                    padding = args.patch_len - remainder
                input_token_len = calculate_unfold_output_length(
                    configs_list[i][1]['seq_len'] + padding, args.stride, args.patch_len)
                input_pad = args.stride * \
                            (input_token_len - 1) + args.patch_len - \
                            configs_list[i][1]['seq_len']
                pred_token_len = calculate_unfold_output_length(
                    configs_list[i][1]['pred_len'] - input_pad, args.stride, args.patch_len)
                real_len = configs_list[i][1]['seq_len'] + \
                           configs_list[i][1]['pred_len']
                self.cls_nums[task_data_name] = [pred_token_len,
                                                 configs_list[i][1]['pred_len'], real_len]

        self.configs_list = configs_list

        ### model settings ###
        self.prompt_num = args.prompt_num
        self.stride = args.stride  # self.patch_len // 2
        self.pad = args.stride
        self.patch_len = args.patch_len

        # input processing
        self.patch_embeddings = PatchEmbedding(
            args.d_model, args.patch_len, args.stride, args.stride, args.dropout)
        self.position_embedding = LearnablePositionalEmbedding(args.d_model)
        self.prompt2forecat = DynamicLinear(128, 128, fixed_in=args.prompt_num)

        # basic blocks
        self.block_num = args.e_layers
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=args.dropout, attn_drop=0., drop_path=0.,
                        init_values=None, prefix_token_length=args.prompt_num) for l in range(args.e_layers)]
        )

        # output processing
        self.cls_head = CLSHead(args.d_model, head_dropout=args.dropout)
        self.forecast_head = ForecastHead(
            args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=args.prompt_num,
            head_dropout=args.dropout)
        if pretrain:
            self.pretrain_head = ForecastHead(
                args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=1,
                head_dropout=args.dropout)

    def tokenize(self, x, mask=None):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(dim=1)
        else:
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x.permute(0, 2, 1)
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        return x, means, stdev, n_vars, padding

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

