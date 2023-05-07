import math

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# from s4d import S4D


def model_cfg_str(options):
    r"""Config to string."""
    return (
        f"sinconv0{options.get('sinc_conv0')}_"
        f"cfg{'x'.join(str(x) for x in options.get('cfg'))}_"
        f"kr{'x'.join(str(x) for x in options.get('kernel'))}_"
        f"krgr{'x'.join(str(x) for x in options.get(''))}"
    )


def padding_same(input,  kernel, stride=1, dilation=1):
    """
        Calculates padding for applied dilation.
    """
    return int(0.5 * (stride * (input - 1) - input + kernel + (dilation - 1) * (kernel - 1)))


def padding_same2(kernel_size, dilation=1):
    return (kernel_size-1) * dilation


def flip(x, dim):
    r"""Flip function."""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    print(f"[flip] x:{x.size()}, {x}")
    x = x.view(x.size(0), x.size(1), -1)[
        : getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)


def sinc(band, t_right):
    r"""sinc function."""
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    print(f"[sinc] y_right({y_right.shape}): {y_right}")
    y_left = flip(y_right, 0)
    # y_left = torch.flip(y_right, dims=[1])
    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])
    return y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BranchConvResNet(nn.Module):
    def __init__(self, options, shortcut=True, log=print) -> None:
        super().__init__()

        self.iter = 0
        self.log = log

        self.branches = nn.ModuleList([])
        self.gap_layers = nn.ModuleList([])
        self.post_gap_layers = nn.ModuleList([])
        for i_br in range(len(options['kernels'])):
            self.gap_layers.append(nn.AdaptiveAvgPool1d(1))
            layers_brkr = []
            ch_in = 1
            residual_block = []
            last_blk_out_channel = ch_in
            br_ch_out = -1
            for i_brkr, brkr in enumerate(options['kernels'][i_br]):
                if brkr == 'M':
                    layers_brkr += [nn.MaxPool1d(kernel_size=2, stride=2)]
                elif brkr == 'R':
                    # Include upto last conv (exclude bn, relu, do), since identify will be added with conv-out.
                    layers_brkr += [
                        ShortcutBlock(
                            nn.Sequential(*residual_block[:-2]), 
                            in_channels=last_blk_out_channel, out_channels=ch_out)
                    ]
                    # Init conv weights
                    for layer in residual_block:
                        if isinstance(layer, nn.Conv1d):
                            layer.weight.data.normal_(0, 0.01)
                            # print(f"{layer} initialised.")
                    residual_block = []  # reset block
                    last_blk_out_channel = ch_out
                else:
                    # kr, ch_out = brkr
                    kr, stride, dilation, ch_out = brkr

                    padding_ = padding_same2(kr, dilation) if stride==1 else 0

                    conv = weight_norm(nn.Conv1d(
                        in_channels=ch_in, out_channels=ch_out,
                        kernel_size=kr, stride=stride, dilation=dilation,
                        padding=padding_same2(kr, dilation) if stride==1 else 0
                    ))
                    residual_block += [
                        conv, 
                        Chomp1d(padding_),
                        # nn.BatchNorm1d(ch_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(options['dropout'])
                    ]
                    ch_in = ch_out
                    br_ch_out = ch_out
            self.branches.append(
                nn.Sequential(*layers_brkr)
            )
            # Reduce channels to fit same number of channels per branch
            self.post_gap_layers.append(
                nn.Sequential(
                    nn.Conv1d(br_ch_out, options['feat_per_branch'],
                          kernel_size=1, bias=False, groups=1),
                    nn.BatchNorm1d(options['feat_per_branch'])
                )
            )

            # Init conv weights
            for layer in layers_brkr:
                if isinstance(layer, nn.Conv1d):
                    layer.weight.data.normal_(0, 0.01)
                    # print(f"{layer} initialised.")

        self.classifier = nn.Sequential(
            # nn.Linear(options['h'][0], options['h'][1]),
            # nn.BatchNorm1d(options['h'][1]),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            # nn.Linear(options['h'][1], options['h'][-1]),
            # nn.BatchNorm1d(options['h'][-1]),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(options['h'], options['n_classes'])
        )
        
    def forward(self, x):
        self.debug(f'input: {x.shape}')

        out = None
        for i_br, (l_br, l_gap, l_post_gap) in enumerate(zip(self.branches, self.gap_layers, self.post_gap_layers)):
            z = l_br(x)
            self.debug(f"br[{i_br}]-conv out: {z.shape}")
            z = l_gap(z)
            self.debug(f"br[{i_br}]-gap out: {z.shape}")
            z = l_post_gap(z)
            self.debug(f"br[{i_br}]-post-gap out: {z.shape}")
            z = z.view(z.size(0), -1)
            self.debug(f"br[{i_br}]-reshape out: {z.shape}")
            if out is None:
                out = z
            else:
                out = torch.cat([out, z], 1)
            self.debug(f"br[{i_br}]-cat out: {out.shape}")
        out = self.classifier(out)
        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvBlock(nn.Module):
    def __init__(self, io_channels, hidden_channels, kernel_size, padding, dilation, no_residual) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            nn.Conv1d(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                dilation=dilation, 
                groups=hidden_channels),
            Chomp1d(padding),    
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08)
        )
        self.res_out = (
            None
            if no_residual
            else nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        )
        self.skip_out = nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        
    def forward(self, x):
        feature = self.conv_layers(x)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class AutoEncTcn(nn.Module):
    def __init__(self, options=None, log=print) -> None:
        super().__init__()

        self.iter = 0
        self.log = log
        self.options = options

        self.input_norm = nn.GroupNorm(num_groups=1, num_channels=1, eps=1e-8)
        ch_in = options['kernels'][0][-1]
        self.input_conv = nn.Conv1d(in_channels=1, out_channels=ch_in, kernel_size=1)
        
        layers_brkr = []
        # ch_in = 1
        residual_block = []
        last_blk_out_channel = ch_in
        for i_brkr, brkr in enumerate(options['kernels']):
            self.debug(f"kernel-{i_brkr}: {brkr}")
            if brkr == 'M':
                layers_brkr += [nn.MaxPool1d(kernel_size=2, stride=2)]
            elif brkr == 'R':
                # Include upto last conv (exclude bn, relu, do), since identify will be added with conv-out.
                layers_brkr += [
                    ShortcutBlock(
                        nn.Sequential(*residual_block[:-2]), 
                        in_channels=last_blk_out_channel, out_channels=ch_out),
                    SELayer(channel=ch_out, reduction=8)
                ]
                # Init conv weights
                for layer in residual_block:
                    if isinstance(layer, nn.Conv1d):
                        layer.weight.data.normal_(0, 0.01)
                        # print(f"{layer} initialised.")
                residual_block = []  # reset block
                last_blk_out_channel = ch_out
            else:
                # kr, ch_out = brkr
                kr, stride, dilation, ch_out = brkr

                padding_ = padding_same2(kr, dilation) if stride==1 else 0

                conv = weight_norm(nn.Conv1d(
                    in_channels=ch_in, out_channels=ch_out,
                    kernel_size=kr, stride=stride, dilation=dilation,
                    padding=padding_same2(kr, dilation) if stride==1 else 0
                ))
                residual_block += [
                    conv, 
                    Chomp1d(padding_),
                    # nn.BatchNorm1d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout(options['dropout'])
                ]
                ch_in = ch_out
        self.encoder = nn.Sequential(*layers_brkr)
        # self.pre_classifier = nn.Conv1d(in_channels=ch_in, out_channels=1, kernel_size=1)
        self.classifier_branchs = nn.ModuleList([
            nn.Conv1d(in_channels=ch_in, out_channels=1, kernel_size=1),
            nn.Sequential(
                nn.Linear(options['h'][0], 5),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
        ])

        # self.decoder_layers = nn.ModuleList()
        layers = []
        # prepare for opposit channel order
        out_channels_ = [1, 16, 32]
        in_chan = out_channels_[-1]
        for i_layer in range(len(out_channels_)-1):
            out_chan = out_channels_[-i_layer-2]
            layers += [
                nn.ConvTranspose1d(
                    in_channels=in_chan, 
                    out_channels=out_chan, 
                    kernel_size=2, 
                    stride=2),
                nn.ReLU(inplace=True) if (i_layer == len(out_channels_) - 1) else nn.Sigmoid()
            ]
            in_chan = out_chan
        self.decoder = nn.Sequential(*layers)

        self.classifier = nn.Linear(options['h'][1], 2)

    def forward(self, x):
        self.debug(f'input: {x.shape}')

        out = self.input_norm(x)
        out = self.input_conv(out)
        self.debug(f"in_conv out: {out.shape}")

        # encoder
        out = out_encoder = self.encoder(out)
        self.debug(f"enc out: {out_encoder.shape}")

        # decoder 
        out_decoder = self.decoder(out)
        self.debug(f"dec out: {out_decoder.shape}")

        # encoder output -> classifier
        out_classif = None
        for i_br_classif in range(len(self.classifier_branchs)):
            z = self.classifier_branchs[i_br_classif](out_encoder)
            self.debug(f"classif_br-{i_br_classif} layer0: {z.shape}")
            z = z.view(z.size(0), -1)
            if out_classif is None:
                out_classif = z
            else:
                out_classif = torch.cat([out_classif, z], 1)
        self.debug(f"classif cat: {out_classif.shape}")
        out_classif = self.classifier(out_classif)

        # decoder output (refined) -> encoder -> classifier
        out_classif_refined = None
        # out_encoder_refined = self.encoder(self.input_conv(out_decoder))
        # for i_br_classif in range(len(self.classifier_branchs)):
        #     z = self.classifier_branchs[i_br_classif](out_encoder_refined)
        #     self.debug(f"classif_br-{i_br_classif} layer0: {z.shape}")
        #     z = z.view(z.size(0), -1)
        #     if out_classif_refined is None:
        #         out_classif_refined = z
        #     else:
        #         out_classif_refined = torch.cat([out_classif_refined, z], 1)
        # self.debug(f"classif_refined cat: {out_classif_refined.shape}")
        # out_classif_refined = self.classifier(out_classif_refined)        

        # self.debug(f"pre_classif flatten: {out_classif.shape}")
        self.iter += 1
        return out_classif, out_classif_refined, out_decoder
        

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class SincConv1d_fast(nn.Module):
    r"""Sinc convolution layer."""

    def __init__(
        self, in_channels=1, out_channels=None, kernel_dim=None, hz=None, stride=1,
        padding=0, dilation=1, bias=False, groups=1, min_low_hz=2, min_band_hz=1,
        init_band_hz=2.0, fq_band_lists=None, log=print,
    ):
        r"""Instantiate class."""
        super(SincConv1d_fast, self).__init__()

        if in_channels != 1:
            raise ValueError(
                f"SincConv1d supports one input channel, found {in_channels}."
            )
        if groups != 1:
            raise ValueError(f"SincConv1d supports single group, found {groups}.")
        if stride > 1 and padding > 0:
            raise ValueError(f"Padding should be 0 for >1 stride, found {padding}.")

        self.iter = 0
        self.log = log
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = kernel_dim

        r"Force filters to be odd."
        if kernel_dim % 2 == 0:
            self.kernel_dim += 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if bias:
            raise ValueError("SincConv1d does not support bias.")
        if groups > 1:
            raise ValueError("SincConv1d does not support groups.")

        self.fs = hz
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        # self.max_band_hz = max_band_hz

        r"Initialise filter-banks equally spaced in ECG frequency range."
        low_hz = self.min_low_hz

        if fq_band_lists is not None:
            # fq_band_lists format ([low_hz list], [band_hz list])
            assert len(fq_band_lists[0]) == len(fq_band_lists[-1])
            self.low_hz_ = nn.Parameter(
                torch.Tensor(np.array(fq_band_lists[0])).view(-1, 1)
            )
            self.band_hz_ = nn.Parameter(
                torch.Tensor(np.array(fq_band_lists[-1])).view(-1, 1)
            )
            r"Adjust out channels based on supplied fq lists."
            self.out_channels = len(fq_band_lists[0])
        else:
            # high_hz = low_hz + self.hz - self.fs//4
            # high_hz = low_hz + self.fs - min_band_hz
            high_hz = low_hz + self.fs / 2.0

            hz = np.linspace(low_hz, high_hz, self.out_channels + 1)
            # hz = np.tile(min_low_hz, self.out_channels)
            # hz = np.random.uniform(low_hz, high_hz, self.out_channels+1)

            # hz = np.random.uniform(
            #     low_hz, high_hz, self.out_channels + 1)
            # hz = np.tile(np.array([self.min_low_hz]), self.out_channels + 1)

            r"Filter low frequency (out_channels + 1)"
            self.low_hz_ = nn.Parameter(torch.Tensor(np.sort(hz[:-1])).view(-1, 1))
            # self.low_hz_ = nn.Parameter(torch.Tensor(hz).view(-1, 1))
            # self.low_hz_ = nn.Parameter(torch.Tensor(hz).view(-1, 1))

            r"Filter frequency bank (out_channels, 1)"
            self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
            # self.band_hz_ = nn.Parameter(
            #     torch.Tensor(np.tile(high_hz, self.out_channels)).view(-1, 1))
            # self.band_hz_ = nn.Parameter(
            #     torch.Tensor(np.tile(init_band_hz, self.out_channels)).view(-1, 1))

        r"Hamming window"
        # self.window_ = torch.hamming_window(self.kernel_dim)
        n_lin = torch.linspace(
            0, (self.kernel_dim / 2) - 1, steps=self.kernel_dim // 2
        )  # half-window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_dim)

        # (1, kernel_dim/2)
        n = (self.kernel_dim - 1) / 2.0
        r"Due to symmetry, I only need half of the time axes"
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.fs

        # self.debug(
        #     f"n_:{self.n_.data.detach().cpu().view(-1)}, "
        #     f"low_hz_:{self.low_hz_.data.detach().cpu().view(-1)}, "
        #     f"band_hz_:{self.band_hz_.data.detach().cpu().view(-1)}")

    def name(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size=({self.kernel_dim},), stride=({self.stride},), "
            f"padding=({self.padding},))"
        )

    def forward(self, waveforms):
        r"""
        Parameters
        ----------
        waveforms : 'torch.Tensor' (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : 'torch.Tensor' (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.debug(f"in: {waveforms.shape}")

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        # self.debug(f"low_hz_:{self.low_hz_},\nband_hz_:{self.band_hz_}")

        low = self.min_low_hz + torch.abs(self.low_hz_)
        # low = torch.clamp(
        #     low, self.min_low_hz, self.fs)

        # high = torch.clamp(
        #     low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.fs/2.0)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.fs
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        r"Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM \
        RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified \
        the terms. This way I avoid several useless computations."
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )
        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_dim)

        self.iter += 1

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=self.groups,
        )

    def debug(self, *args):
        if self.iter == 0 and self.log:
            self.log(f"[{self.name()}] {args}")



class SincNet(nn.Module):
    def __init__(self, options, log=print):
        super(SincNet, self).__init__()

        self.iter = 0
        self.log = log
        self.options = options

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])

        # self.feature = self.create_encoder()
        self.create_encoder()
        if options["gap_layer"]:
            self.gap = nn.AdaptiveAvgPool1d(1)
        # optional GRU layer
        if options.get("gru_module") is None:
            options["gru_module"] = False
        if options.get("gru_module"):
            self.gru_module = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.3,
                bidirectional=True)
        
        if options.get("flatten"):
            r"Linear decision layer."
            self.classifier = nn.Linear(self.hidden_dim, options["n_class"])

        # if options.get("flatten"):   # Test
        #     r"Linear decision layer."
        #     # self.hidden_dim = self.hidden_dim * options["input_sz"]//(2*5)
        #     self.hidden_dim = options["hidden_dim"]
        #     self.debug(f"Hidden-dim:{self.hidden_dim}")
        #     self.classifier = nn.Sequential(
        #         nn.Linear(self.hidden_dim, self.hidden_dim//3),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.2),
        #         nn.Linear(self.hidden_dim//3, self.hidden_dim//(3*3)),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.2),
        #         nn.Linear(self.hidden_dim//(3*3), options["n_class"]),
        #     )
        else:
            r"Point conv-layer as decision layer."
            self.classifier = nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=options["n_class"],
                kernel_size=1,
                padding=0,
                dilation=1,
            )
        self.input_bn = nn.BatchNorm1d(1)

    def name(self):
        r"Format cfg meaningfully."
        return f"{self.__class__.__name__}_" f"{model_cfg_str(self.options)}"

    def forward(self, x, i_zero_kernels=None):
        self.debug(f"input: {x.shape}")

        # out = self.feature(x)
        # self.debug(f"features: {out.shape}")

        layer_out = {
            "conv": [],
            "bn": [],
            "act": [],
            "gap": [],
            "flatten": [],
        }

        # out = x
        out = self.input_bn(x)

        i_conv, i_pool, i_act = 0, 0, 0

        z = out
        for x in self.options["cfg"]:
            if x == "M":
                out = self.pool[i_pool](out)
                i_pool += 1
            elif x == "A":
                out = self.act[i_act](out)                
                i_act += 1
            else:
                out = self.conv[i_conv](out)
                self.debug(f"conv{i_conv} out: {out.shape}")

                if i_conv == 0 and i_zero_kernels:
                    r"zero kernel if index is non-negative."
                    i_zero_kernels = i_zero_kernels if i_zero_kernels else []
                    for i_zero_kernel in i_zero_kernels:
                        if i_zero_kernel < 0:
                            continue
                        _b, _c, _d = out.shape
                        out[0, i_zero_kernel, :] = torch.zeros(_d)

                r"Layer output debug."

                layer_out.get("conv").append(out.detach().cpu().numpy())

                out = self.bn[i_conv](out)

                # layer_out.get('bn').append(
                #     out.detach().cpu().numpy()
                # )

                # out = self.act[i_conv](out)

                # layer_out.get('act').append(
                #     out.detach().cpu().numpy()
                # )

                i_conv += 1

            self.debug(f"cfg->{x}, out:{out.shape}")

        if self.options.get("gap_layer"):
            out = self.gap(out)
            layer_out.get("gap").append(out.detach().cpu().numpy())
            self.debug(f"  GAP out: {out.shape}")

        if self.options.get("gru_module"):
            # Post GAP reshape (B, Ch, 1) -> (B, 1, Ch)
            out = out.view(out.size(0), out.size(2), -1)
            # self.debug(f"out reshape:{out.shape}")
            weight = next(self.parameters()).data
            n_layers, n_directions = 2, 2
            self.hidden = (weight.new(n_layers*n_directions, out.shape[0], self.hidden_dim)
            ).zero_().to(out.device)
            _, self.hidden = self.gru_module(out, self.hidden)
            self.debug(f"gru hidden:{self.hidden.shape}")
            final_state = self.hidden.view(
                n_layers, n_directions, out.shape[0], self.hidden_dim
            )
            # self.debug(f"pre final_state:{final_state.shape}")
            final_state = final_state[-1]
            # self.debug(f"final-state:{final_state.shape}")
            h_1, h_2 = final_state[0], final_state[1]
            out = h_1 + h_2
            self.debug(f"*final-state reshape:{out.shape}")
            self.hidden = self.hidden.detach()
        
        if self.options.get("flatten"):
            # if self.feature_pool:
            #     out = self.feature_pool(out)  # TEST
            #     self.debug(f"feature_pool out:{out.shape}")
            out = out.view(out.size(0), -1)
            layer_out.get("flatten").append(out.detach().cpu().numpy())
            self.debug(f"  flatten: {out.shape}")

        out = self.classifier(out)
        self.debug(f"classif: {out.shape}")

        self.iter += 1
        return out, layer_out

    def create_encoder(self):
        # layers = []
        # count_pooling = 0
        i_kernel = 0
        in_chan = 1
        input_sz = self.options["input_sz"]
        for x in self.options["cfg"]:
            self.debug(f"cfg:{x}, i_kernel:{i_kernel}")
            if x == "M":
                # count_pooling += 1
                # layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                pool_kr_stride = 2
                self.pool.append(nn.MaxPool1d(kernel_size=pool_kr_stride, stride=pool_kr_stride))
                input_sz /= pool_kr_stride
            elif x == "A":
                self.act.append(
                    # nn.ELU(inplace=True)
                    nn.ReLU(inplace=True)
                )
            else:
                if i_kernel == 0 and self.options["sinc_conv0"]:
                    conv = SincConv1d_fast(
                        in_channels=in_chan,
                        out_channels=x,
                        kernel_dim=self.options["kernel"][i_kernel],
                        hz=self.options["hz"],
                        log=self.log,
                        fq_band_lists=self.options.get("fq_band_lists"),
                        stride=self.options["stride"][i_kernel],
                        padding=padding_same(
                            input=input_sz,
                            kernel=self.options["kernel"][i_kernel],
                            stride=1,
                            dilation=1,
                        )
                        if self.options["stride"][i_kernel] <= 1
                        else 0,
                    )
                else:
                    conv = nn.Conv1d(
                        in_channels=in_chan,
                        out_channels=x,
                        kernel_size=self.options["kernel"][i_kernel],
                        groups=self.options["kernel_group"][i_kernel],
                        stride=self.options["stride"][i_kernel],
                        padding=padding_same(
                            input=input_sz,
                            kernel=self.options["kernel"][i_kernel],
                            stride=1,
                            dilation=1,
                        )
                        if self.options["stride"][i_kernel] <= 1
                        else 0,
                    )
                i_kernel += 1
                self.conv.append(conv)
                self.bn.append(nn.BatchNorm1d(x))
                in_chan = self.hidden_dim = x
            pass  # for

    def debug(self, *args):
        if self.iter == 0 and self.log:
            # self.log(args)
            self.log(f"[{self.__class__.__name__}] {args}")



class SincResNet(nn.Module):
    r"""Sinc convolution with optional residual connections."""

    def __init__(
        self,
        segment_sz,
        kernels=None,
        dilations=None,
        in_channels=None,
        out_channels=None,
        conv_groups=None,
        n_conv_layers_per_block=2,
        n_blocks=4,
        n_classes=None,
        low_conv_options=None,
        shortcut_conn=False,
        log=print,
    ):
        r"""Instance of convnet."""
        super(SincResNet, self).__init__()

        log(
            f"segment_sz:{segment_sz}, kernels:{kernels}, in-chan:{in_channels}, "
            f"out-chan:{out_channels}, conv-gr:{conv_groups}, "
            f"n-conv-layer-per-block:{n_conv_layers_per_block}, "
            f"n_block:{n_blocks}, n_class:{n_classes}, "
            f"low-conv:{low_conv_options}, shortcut:{shortcut_conn}"
        )

        self.iter = 0
        self.log = log
        self.input_sz = self.segment_sz = segment_sz
        self.kernels = kernels
        self.dilations = dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_groups = conv_groups
        self.n_conv_layers_per_block = n_conv_layers_per_block
        self.n_blocks = n_blocks
        self.low_conv_options = low_conv_options
        self.shortcut_conn = shortcut_conn

        self.input_bn = nn.BatchNorm1d(self.in_channels)
        self.low_conv = self.make_low_conv()

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.shortcut = nn.ModuleList([])
        self.make_layers_deep()
        self.gap = nn.AdaptiveAvgPool1d(1)
        # optional GRU layer
        self.debug(f"self.input_sz for GRU:{self.input_sz}")
        self.gru_module = nn.GRU(
            input_size=int(self.hidden_dim),
            hidden_size=int(self.hidden_dim),
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Linear(self._out_channels, n_classes)

    def name(self):
        return (
            f"{self.__class__.__name__}_"
            f"segsz{self.segment_sz}_scut{self.shortcut_conn}_"
            f"sinc{self.low_conv_options.get('sinc_kernel')}_"
            f"sinckr{self.low_conv_options.get('kernel')[0] if self.low_conv_options.get('sinc_kernel') else 0}_"
            f"lccfg{'x'.join(str(x) for x in self.low_conv_options['cfg'])}_"
            f"lckr{'x'.join(str(x) for x in self.low_conv_options['kernel'])}_"
            f"lcst{'x'.join(str(x) for x in self.low_conv_options['stride'])}_"
            f"lccg{'x'.join(str(x) for x in self.low_conv_options['conv_groups'])}_"
            f"blk{self.n_blocks}_cpblk{self.n_conv_layers_per_block}_"
            f"kr{'x'.join(str(x) for x in self.kernels)}_"
            f"och{'x'.join(str(x) for x in self.out_channels)}_"
            f"cg{'x'.join(str(x) for x in self.conv_groups)}"
        )
        # return f'{self.__class__.__name__}'

    def forward(self, x):
        self.debug(f"  input: {x.shape}")

        x = self.input_bn(x)

        out = self.low_conv(x)
        self.debug(f"  low_conv out: {out.shape}")

        # out = self.features(x)
        # self.debug(f'features out: {out.shape}')

        for i_blk in range(self.n_blocks):
            if self.shortcut_conn:
                out = self.shortcut[i_blk](out)
                self.debug(f"[block:{i_blk}] shortcut out: {out.shape}")
            else:
                for i_conv_blk in range(self.n_conv_layers_per_block):
                    idx_flat = 2 * i_blk + i_conv_blk
                    self.debug(
                        f"  block({i_blk}) conv({i_conv_blk}) {self.conv[idx_flat]}"
                    )
                    out = self.conv[idx_flat](out)
                    out = self.bn[idx_flat](out)
                    out = self.act[idx_flat](out)
                self.debug(
                    f"  block({i_blk}) out:{out.shape}, data:{out.detach().cpu()[0, 0, :10]}"
                )
            r"One less pooling layer."
            if i_blk < self.n_blocks - 1:
                out = self.pool[i_blk](out)
                self.debug(f"  block({i_blk}) pool-out:{out.shape}")

        out = self.gap(out)
        self.debug(f"  GAP out: {out.shape}")


        if self.gru_module is not None:
            out = out.view(out.size(0), out.size(2), -1)
            self.debug(f"out reshape:{out.shape}")
            weight = next(self.parameters()).data
            n_layers, n_directions = 2, 2
            self.hidden = (weight.new(n_layers*n_directions, out.shape[0], self.hidden_dim)
            ).zero_().to(out.device)
            _, self.hidden = self.gru_module(out, self.hidden)
            self.debug(f"gru hidden:{self.hidden.shape}")
            final_state = self.hidden.view(
                n_layers, n_directions, out.shape[0], self.hidden_dim
            )
            self.debug(f"pre final_state:{final_state.shape}")
            final_state = final_state[-1]
            self.debug(f"final-state:{final_state.shape}")
            h_1, h_2 = final_state[0], final_state[1]
            out = h_1 + h_2
            self.debug(f"*final-state reshape:{out.shape}")
            self.hidden = self.hidden.detach()

        out = out.view(out.size(0), -1)
        self.debug(f"  flatten: {out.shape}")
        out = self.classifier(out)
        self.debug(f"  out: {out.shape}")
        self.iter += 1
        return out

    def calculate_hidden(self):
        return self.input_sz * self.out_channels[-1]

    def make_layers_deep(self):
        # layers = []
        # in_channels = self.in_channels
        in_channels = self.low_conv_hidden_dim
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            layers_for_shortcut = []
            in_channel_for_shortcut = in_channels
            for _ in range(self.n_conv_layers_per_block):
                self.conv.append(
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i],
                        dilation=self.dilations[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        padding=padding_same(
                            input=self.input_sz,
                            kernel=self.kernels[i],
                            stride=1,
                            dilation=self.dilations[i],
                        ),
                    )
                )
                self.bn.append(nn.BatchNorm1d(self._out_channels))
                self.act.append(nn.ReLU(inplace=True))
                if self.shortcut_conn:
                    layers_for_shortcut.extend(
                        [
                            self.conv[-1],
                            self.bn[-1],
                        ]
                    )
                in_channels = self.hidden_dim = self._out_channels

            if self.shortcut_conn:
                self.shortcut.append(
                    ShortcutBlock(
                        layers=nn.Sequential(*layers_for_shortcut),
                        in_channels=in_channel_for_shortcut,
                        out_channels=self._out_channels,
                        point_conv_group=self.conv_groups[i],
                    )
                )
            if i < self.n_blocks - 1:
                self.pool.append(nn.MaxPool1d(2, stride=2))
                self.input_sz //= 2
        r"If shortcut_conn is true, empty conv, and bn module-list. \
        This may be necessary to not to calculate gradients for the \
        same layer twice."
        if self.shortcut_conn:
            self.conv = nn.ModuleList([])
            self.bn = nn.ModuleList([])
            self.act = nn.ModuleList([])

    def make_low_conv(self):
        layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = self.in_channels
        # input_sz = self.input_sz
        for x in self.low_conv_options["cfg"]:
            if x == "M":
                count_pooling += 1
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.input_sz /= 2
            else:
                stride_ = self.low_conv_options["stride"][i_kernel]
                dilation_ = self.low_conv_options["dilation"][i_kernel]
                padding_ = (
                    0
                    if stride_ > 1
                    else padding_same(
                        input=self.input_sz,
                        kernel=self.low_conv_options["kernel"][i_kernel],
                        stride=stride_,
                        dilation=dilation_,
                    )
                )
                if i_kernel == 0 and self.low_conv_options.get("sinc_kernel"):
                    conv = SincConv1d_fast(
                        in_channels=in_chan,
                        out_channels=x,
                        kernel_dim=self.low_conv_options["kernel"][i_kernel],
                        stride=stride_,
                        hz=self.low_conv_options["hz"],
                        log=self.log,
                        fq_band_lists=self.low_conv_options.get("fq_band_lists"),
                        padding=padding_,
                    )
                else:
                    conv = nn.Conv1d(
                        in_channels=in_chan,
                        out_channels=x,
                        kernel_size=self.low_conv_options["kernel"][i_kernel],
                        groups=self.low_conv_options["conv_groups"][i_kernel],
                        dilation=dilation_,
                        stride=stride_,
                        padding=padding_,
                    )
                layers += [conv, nn.BatchNorm1d(x), nn.ReLU(inplace=True)]
                in_chan = self.low_conv_hidden_dim = x
                self.input_sz /= self.low_conv_options["stride"][i_kernel]
                i_kernel += 1
            pass  # for
        return nn.Sequential(*layers)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")



class BranchRawAndRRConvNet(nn.Module):
    def __init__(self, options, log=print) -> None:
        super().__init__()

        self.iter = 0
        self.log = log

        self.branches = nn.ModuleList([])
        self.gap_layers = nn.ModuleList([])
        for i_br in range(len(options['kernels'])):
            self.gap_layers.append(nn.AdaptiveAvgPool1d(1))
            layers_brkr = []
            ch_in = 1
            input_sz = options['input_sz']
            for brkr in options['kernels'][i_br]:
                if brkr == 'M':
                    layers_brkr += [nn.MaxPool1d(kernel_size=2, stride=2)]
                    input_sz /= 2
                else:
                    kr, stride, dilation, ch_out = brkr
                    conv = nn.Conv1d(
                        in_channels=ch_in, out_channels=ch_out,
                        kernel_size=kr, stride=stride, dilation=dilation,
                        padding=padding_same(input=input_sz, kernel=kr,) if stride==1 else 0)
                    layers_brkr += [
                        conv,
                        nn.BatchNorm1d(ch_out),
                        nn.ReLU(inplace=True)
                    ]
                    ch_in = ch_out
            self.branches.append(
                nn.Sequential(*layers_brkr)
            )

        # R-R signal branch
        self.branches_rr = nn.ModuleList([])
        self.gap_layers_rr = nn.ModuleList([])
        for i_br in range(len(options['kernels_rr'])):
            self.gap_layers_rr.append(nn.AdaptiveAvgPool1d(1))
            layers_brkr = []
            ch_in = 1
            input_sz = options['input_rr_sz']
            for brkr in options['kernels_rr'][i_br]:
                if brkr == 'M':
                    layers_brkr += [nn.MaxPool1d(kernel_size=2, stride=2)]
                    input_sz /= 2
                else:
                    kr, stride, dilation, ch_out = brkr
                    conv = nn.Conv1d(
                        in_channels=ch_in, out_channels=ch_out,
                        kernel_size=kr, stride=stride, dilation=dilation,
                        padding=padding_same(input=input_sz, kernel=kr, stride=stride, dilation=dilation) if stride==1 else 0)
                    layers_brkr += [
                        conv,
                        nn.BatchNorm1d(ch_out),
                        nn.ReLU(inplace=True)
                    ]
                    ch_in = ch_out
            self.branches_rr.append(
                nn.Sequential(*layers_brkr)
            )

        # self.pre_classifier_bn = nn.BatchNorm1d(options['h'][0])

        self.classifier = nn.Sequential(
            # nn.Linear(options['feat_per_branch']*(len(self.branches)+len(self.branches_rr)), options['h1']),
            nn.Linear(options['h'][0], options['h'][1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(options['h'][1], options['h'][-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(options['h'][-1], 2)
        )
        
    def forward(self, x):
        self.debug(f'input: {x.shape}')

        b, c, d = x.size()
        x_orig = x 
        x = x_orig[:, :c//2]
        x2 = x_orig[:, c//2:, :d//100]
        # x2 = x2.view(x2.size(0), 1, -1)
        self.debug(f"x1:{x.shape}, x2:{x2.shape}")

        out = None
        for i_br, (l_br, l_gap) in enumerate(zip(self.branches, self.gap_layers)):
            z = l_br(x)
            self.debug(f"br[{i_br}]-conv out: {z.shape}")
            z = l_gap(z)
            self.debug(f"br[{i_br}]-gap out: {z.shape}")
            z = z.view(z.size(0), -1)
            self.debug(f"br[{i_br}]-reshape out: {z.shape}")
            if out is None:
                out = z
            else:
                out = torch.cat([out, z], 1)
            self.debug(f"br[{i_br}]-cat out: {out.shape}")

        # R-R branch
        out_rr = None
        for i_br, (l_br, l_gap) in enumerate(zip(self.branches_rr, self.gap_layers_rr)):
            z = l_br(x2)
            self.debug(f"rr_br[{i_br}]-conv out: {z.shape}")
            z = l_gap(z)
            self.debug(f"rr_br[{i_br}]-gap out: {z.shape}")
            z = z.view(z.size(0), -1)
            self.debug(f"rr_br[{i_br}]-reshape out: {z.shape}")
            if out_rr is None:
                out_rr = z
            else:
                out_rr = torch.cat([out_rr, z], 1)
            self.debug(f"rr_br[{i_br}]-cat out: {out_rr.shape}")

        out = torch.cat([out, out_rr], 1)
        self.debug(f"cat* out: {out.shape}")

        # out = self.pre_classifier_bn(out)

        out = self.classifier(out)
        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class BranchRawAndRRConvResNet(nn.Module):
    def __init__(self, options, log=print) -> None:
        super().__init__()

        self.iter = 0
        self.log = log
        self.options = options
        
        self.branches = nn.ModuleList([])
        self.gap_layers = nn.ModuleList([])
        self.post_gap_layers = nn.ModuleList([])
        self.tcn_layers = nn.ModuleList([])
        for i_br in range(len(options['kernels_0'])):
            self.gap_layers.append(nn.AdaptiveAvgPool1d(1))
            layers_brkr = []
            ch_in = 1
            residual_block = []
            last_blk_out_channel = ch_in
            br_ch_out = -1
            last_kr_size = -1
            for i_brkr, brkr in enumerate(options['kernels_0'][i_br]):
                if brkr == 'M':
                    layers_brkr += [nn.MaxPool1d(kernel_size=2, stride=2)]
                elif brkr == 'R':
                    # Include upto last conv (exclude bn, relu, do), since identify will be added with conv-out.
                    layers_brkr += [
                        ShortcutBlock(
                            nn.Sequential(*residual_block[:-2]), 
                            in_channels=last_blk_out_channel, out_channels=ch_out)
                    ]
                    # Init conv weights
                    for layer in residual_block:
                        if isinstance(layer, nn.Conv1d):
                            layer.weight.data.normal_(0, 0.01)
                            # print(f"{layer} initialised.")
                    residual_block = []  # reset block
                    last_blk_out_channel = ch_out
                else:
                    # kr, ch_out = brkr
                    kr, stride, dilation, ch_out = brkr
                    last_kr_size = kr
                    padding_ = padding_same2(kr, dilation) if stride==1 else 0
                    conv = nn.Conv1d(
                        in_channels=ch_in, out_channels=ch_out,
                        kernel_size=kr, stride=stride, dilation=dilation,
                        padding=padding_same2(kr, dilation) if stride==1 else 0
                    )
                    # conv = weight_norm(nn.Conv1d(
                    #     in_channels=ch_in, out_channels=ch_out,
                    #     kernel_size=kr, stride=stride, dilation=dilation,
                    #     padding=padding_same2(kr, dilation) if stride==1 else 0
                    # ))
                    residual_block += [
                        conv, 
                        Chomp1d(padding_),
                        # nn.BatchNorm1d(ch_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(options['dropout'])
                    ]
                    ch_in = ch_out
                    br_ch_out = ch_out
            self.branches.append(
                nn.Sequential(*layers_brkr)
            )
            # Temporal convolution
            self.tcn_layers.append(
                TemporalConvNet(br_ch_out, [br_ch_out*2]*4, last_kr_size, dropout=options['dropout']))
            # Reduce channels to fit same number of channels per branch
            self.post_gap_layers.append(
                nn.Sequential(
                    nn.Conv1d(br_ch_out*2, options['feat_per_branch'],
                          kernel_size=1, bias=False, groups=1),
                    nn.BatchNorm1d(options['feat_per_branch'])
                )
            )
            # Init conv weights
            for layer in layers_brkr:
                if isinstance(layer, nn.Conv1d):
                    layer.weight.data.normal_(0, 0.01)
                    # print(f"{layer} initialised.")

        # R-R
        if options.get('kernels_1') is not None:
            self.branches_rr = nn.ModuleList([])
            self.gap_layers_rr = nn.ModuleList([])
            self.post_gap_layers_rr = nn.ModuleList([])
            self.tcn_layers_rr = nn.ModuleList([])
            for i_br in range(len(options['kernels_1'])):
                self.gap_layers_rr.append(nn.AdaptiveAvgPool1d(1))
                layers_brkr = []
                ch_in = 1
                residual_block = []
                last_blk_out_channel = ch_in
                br_ch_out = -1
                last_kr_size = -1
                for i_brkr, brkr in enumerate(options['kernels_1'][i_br]):
                    if brkr == 'M':
                        layers_brkr += [nn.MaxPool1d(kernel_size=2, stride=2)]
                    elif brkr == 'R':
                        # Include upto last conv (exclude bn, relu, do), since identify will be added with conv-out.
                        layers_brkr += [
                            ShortcutBlock(
                                nn.Sequential(*residual_block[:-2]), 
                                in_channels=last_blk_out_channel, out_channels=ch_out)
                        ]
                        # Init conv weights
                        for layer in residual_block:
                            if isinstance(layer, nn.Conv1d):
                                layer.weight.data.normal_(0, 0.01)
                                # print(f"{layer} initialised.")
                        residual_block = []  # reset block
                        last_blk_out_channel = ch_out
                    else:
                        # kr, ch_out = brkr
                        kr, stride, dilation, ch_out = brkr
                        last_kr_size = kr
                        padding_ = padding_same2(kr, dilation) if stride==1 else 0
                        conv = nn.Conv1d(
                            in_channels=ch_in, out_channels=ch_out,
                            kernel_size=kr, stride=stride, dilation=dilation,
                            padding=padding_same2(kr, dilation) if stride==1 else 0
                        )
                        # conv = weight_norm(nn.Conv1d(
                        #     in_channels=ch_in, out_channels=ch_out,
                        #     kernel_size=kr, stride=stride, dilation=dilation,
                        #     padding=padding_same2(kr, dilation) if stride==1 else 0
                        # ))
                        residual_block += [
                            conv, 
                            Chomp1d(padding_),
                            # nn.BatchNorm1d(ch_out),
                            nn.ReLU(inplace=True),
                            nn.Dropout(options['dropout'])
                        ]
                        ch_in = ch_out
                        br_ch_out = ch_out
                self.branches_rr.append(
                    nn.Sequential(*layers_brkr)
                )
                # Temporal convolution
                self.tcn_layers_rr.append(
                    TemporalConvNet(br_ch_out, [br_ch_out*2]*3, last_kr_size, dropout=options['dropout']))
                # Reduce channels to fit same number of channels per branch
                self.post_gap_layers_rr.append(
                    nn.Sequential(
                        nn.Conv1d(br_ch_out*2, options['feat_per_branch'],
                            kernel_size=1, bias=False, groups=1),
                        nn.BatchNorm1d(options['feat_per_branch'])
                    )
                )
                # Init conv weights
                for layer in layers_brkr:
                    if isinstance(layer, nn.Conv1d):
                        layer.weight.data.normal_(0, 0.01)
                        # print(f"{layer} initialised.")

        self.pre_classif_bn = nn.BatchNorm1d(options['h'])
        self.classifier = nn.Sequential(
            # nn.Linear(options['h'][0], options['h'][1]),
            # nn.BatchNorm1d(options['h'][1]),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            # nn.Linear(options['h'][1], options['h'][-1]),
            # nn.BatchNorm1d(options['h'][-1]),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(options['h'], options['n_classes'])
        )
        
    def forward(self, x):
        self.debug(f'input: {x.shape}')

        _, c, _ = x.size()
        x_orig = x 
        x = x_orig[:, :c//2]
        x2 = x_orig[:, c//2:]
        self.debug(f"x1:{x.shape}, x2:{x2.shape},")
        self.debug(f"rr_branch:{self.options.get('kernels_1') is not None}")

        out = None
        for i_br, (l_br, l_tcn, l_gap, l_post_gap) in enumerate(zip(self.branches, self.tcn_layers, self.gap_layers, self.post_gap_layers)):
            z = l_br(x)
            self.debug(f"br[{i_br}]-conv out: {z.shape}")
            z = l_tcn(z)
            self.debug(f"br[{i_br}]-tcn out: {z.shape}")
            z = l_gap(z)
            self.debug(f"br[{i_br}]-gap out: {z.shape}")
            z = l_post_gap(z)
            self.debug(f"br[{i_br}]-post-gap out: {z.shape}")
            z = z.view(z.size(0), -1)
            self.debug(f"br[{i_br}]-reshape out: {z.shape}")
            if out is None:
                out = z
            else:
                out = torch.cat([out, z], 1)
            self.debug(f"br[{i_br}]-cat out: {out.shape}")

        if self.options.get('kernels_1') is not None:
            out_rr = None
            for i_br, (l_br, l_tcn, l_gap, l_post_gap) in enumerate(zip(self.branches_rr, self.tcn_layers_rr, self.gap_layers_rr, self.post_gap_layers_rr)):
                z = l_br(x2)
                self.debug(f"rr_br[{i_br}]-conv out: {z.shape}")
                z = l_tcn(z)
                self.debug(f"rr_br[{i_br}]-tcn out: {z.shape}")
                z = l_gap(z)
                self.debug(f"rr_br[{i_br}]-gap out: {z.shape}")
                z = l_post_gap(z)
                self.debug(f"rr_br[{i_br}]-post-gap out: {z.shape}")
                z = z.view(z.size(0), -1)
                self.debug(f"rr_br[{i_br}]-reshape out: {z.shape}")
                if out_rr is None:
                    out_rr = z
                else:
                    out_rr = torch.cat([out_rr, z], 1)
                self.debug(f"rr_br[{i_br}]-cat out: {out.shape}")            

        if self.options.get('kernels_1') is not None:
            out = torch.cat([out, out_rr], 1)
        self.debug(f"cat* out: {out.shape}")

        out = self.pre_classif_bn(out)
        out = self.classifier(out)
        self.debug(f"out: {out.shape}")
        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class ConvBlockTemplate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, groups) -> None:
        super().__init__()

        layers = [nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-08),]
        if in_channels != out_channels:
            layers += [
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=groups),
                nn.PReLU(),
                nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-08),
            ]
        layers += [            
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups),
            Chomp1d(padding),
            nn.PReLU(),
        ]
        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv_layers(x)



class PyramidNet(nn.Module):
    def __init__(self, cfg, log=print) -> None:
        super().__init__()

        self.iter = True
        self.log = log
        self.cfg = cfg
        ch_out = cfg["sinc_conv0"]
        sinc_layer = SincConv1d_fast(
            in_channels=1, out_channels=ch_out, kernel_dim=17, hz=cfg["hz"],
            fq_band_lists=cfg.get("fq_band_lists"), stride=17//2,)
        self.low_conv_layers = nn.Sequential(
            nn.BatchNorm1d(1),
            sinc_layer,
            nn.PReLU()
        )
        layers = []
        for layer_cfg in cfg['layers']:
            if layer_cfg == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                _kernel, _stride, _dilation, _chan, _group = layer_cfg
                _padding = padding_same2(_kernel, _dilation) if _stride==1 else 0
                layers += [
                    ConvBlockTemplate(
                        ch_out, _chan, _kernel, _padding, _dilation, _group)
                ]
                ch_out = _chan
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(cfg['h'][0], cfg['h'][1]),
            nn.BatchNorm1d(cfg['h'][1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cfg['h'][1], cfg['h'][2]),
            nn.BatchNorm1d(cfg['h'][2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cfg['h'][2], 2)
        )
    
    def forward(self, x):
        self.debug(f'input: {x.shape}')
        
        out = self.low_conv_layers(x)
        self.debug(f'pre_encoder: {out.shape}')
        
        out = self.encoder(out)
        self.debug(f'encoder: {out.shape}')

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        self.debug(f'classifier: {out.shape}')
        self.iter = False
        return out
    
    def debug(self, args):
        if self.iter:
            self.log(f"[{self.__class__.__name__}] {args}")


class BranchConvNet(nn.Module):
    def __init__(self, options, log=print) -> None:
        super().__init__()

        self.iter = 0
        self.log = log

        self.branches = nn.ModuleList([])
        self.gap_layers = nn.ModuleList([])
        for i_br in range(len(options['kernels'])):
            self.gap_layers.append(nn.AdaptiveAvgPool1d(1))
            layers_brkr = []
            ch_in = 1
            input_sz = options['input_sz']
            for brkr in options['kernels'][i_br]:
                if brkr == 'M':
                    layers_brkr += [nn.MaxPool1d(kernel_size=2, stride=2)]
                    input_sz /= 2
                else:
                    kr, ch_out = brkr
                    conv = nn.Conv1d(
                        in_channels=ch_in, out_channels=ch_out,
                        kernel_size=kr,
                        padding=padding_same(input=input_sz, kernel=kr,))
                    layers_brkr += [
                        conv,
                        nn.BatchNorm1d(ch_out),
                        nn.ReLU(inplace=True)
                    ]
                    ch_in = ch_out
            self.branches.append(
                nn.Sequential(*layers_brkr)
            )
        self.classifier = nn.Sequential(
            nn.Linear(options['feat_per_branch']*len(self.branches), options['h1']),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(options['h1'], options['h2']),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(options['h2'], 2)
        )
        
    def forward(self, x):
        self.debug(f'input: {x.shape}')

        out = None
        for i_br, (l_br, l_gap) in enumerate(zip(self.branches, self.gap_layers)):
            z = l_br(x)
            self.debug(f"br[{i_br}]-conv out: {z.shape}")
            z = l_gap(z)
            self.debug(f"br[{i_br}]-gap out: {z.shape}")
            z = z.view(z.size(0), -1)
            self.debug(f"br[{i_br}]-reshape out: {z.shape}")
            if out is None:
                out = z
            else:
                out = torch.cat([out, z], 1)
            self.debug(f"br[{i_br}]-cat out: {out.shape}")
        out = self.classifier(out)
        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class ShortcutBlock(nn.Module):
    '''Pass a Sequence and add identity shortcut, following a ReLU.'''

    def __init__(
        self, layers=None, in_channels=None, out_channels=None,
        point_conv_group=1
    ):
        super(ShortcutBlock, self).__init__()
        self.iter = 0
        self.layers = layers
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, bias=False, groups=point_conv_group),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # self.debug(f'input: {x.shape}')

        out = self.layers(x)
        # self.debug(f'layers out: {out.shape}')

        out += self.shortcut(x)
        # self.debug(f'shortcut out: {out.shape}')

        out = F.relu(out)

        self.iter += 1
        return out

    def debug(self, *args):
        if self.iter == 0:
            print(self.__class__.__name__, args)


class ClassicConv(nn.Module):
    r"""Convolution network."""

    def __init__(
        self, segment_sz, kernels=None, in_channels=None, out_channels=None,
        conv_groups=None, n_conv_layers_per_block=1, n_blocks=2,
        n_classes=None, low_conv_options=None, shortcut_conn=False, log=print,
        gap_layer=True, hidden_neurons_classif=-1
    ):
        r"""Instance of convnet."""
        super(ClassicConv, self).__init__()

        log(
            f"segment_sz:{segment_sz}, kernels:{kernels}, in-chan:{in_channels}, "
            f"out-chan:{out_channels}, conv-gr:{conv_groups}, "
            f"n-conv-layer-per-block:{n_conv_layers_per_block}, "
            f"n_block:{n_blocks}, n_class:{n_classes}, "
            f"low-conv:{low_conv_options}, shortcut:{shortcut_conn}")

        self.iter = 0
        self.log = log
        self.input_sz = self.segment_sz = segment_sz
        self.kernels = kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_groups = conv_groups
        self.n_conv_layers_per_block = n_conv_layers_per_block
        self.n_blocks = n_blocks
        self.low_conv_options = low_conv_options
        self.shortcut_conn = shortcut_conn
        self.gap_layer = gap_layer
        self.hidden_neurons_classif = hidden_neurons_classif

        self.input_bn = nn.BatchNorm1d(self.in_channels)
        self.low_conv = self.make_low_conv()
        # self.features = self.make_layers_deep()

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.shortcut = nn.ModuleList([])
        self.make_layers_deep()

        # n_hidden = self.calculate_hidden()
        # self.classifier = nn.Sequential(
        #     nn.Linear(n_hidden, n_hidden//2),
        #     nn.BatchNorm1d(n_hidden//2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(n_hidden//2, 2)
        # )
        if self.gap_layer:
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(self._out_channels, n_classes)
        else:
            self.classifier = nn.Linear(hidden_neurons_classif, n_classes)

    def name(self):
        return (
            f"{self.__class__.__name__}_"
            f"segsz{self.segment_sz}_scut{self.shortcut_conn}_"
            f"lccfg{'x'.join(str(x) for x in self.low_conv_options['cfg'])}_"
            f"lckr{'x'.join(str(x) for x in self.low_conv_options['kernel'])}_"
            f"lcst{'x'.join(str(x) for x in self.low_conv_options['stride'])}_"
            f"lccg{'x'.join(str(x) for x in self.low_conv_options['conv_groups'])}_"
            f"blk{self.n_blocks}_cpblk{self.n_conv_layers_per_block}_"
            f"kr{'x'.join(str(x) for x in self.kernels)}_"
            f"och{'x'.join(str(x) for x in self.out_channels)}_"
            f"cg{'x'.join(str(x) for x in self.conv_groups)}")
        # return f'{self.__class__.__name__}'

    def forward(self, x):
        self.debug(f'  input: {x.shape}')

        x = self.input_bn(x)

        out = self.low_conv(x)
        self.debug(f'  low_conv out: {out.shape}')

        for i_blk in range(self.n_blocks):
            if self.shortcut_conn:
                out = self.shortcut[i_blk](out)
                self.debug(f"[block:{i_blk}] shortcut out: {out.shape}")
            else:
                for i_conv_blk in range(self.n_conv_layers_per_block):
                    idx_flat = 2*i_blk+i_conv_blk
                    self.debug(
                        f"  block({i_blk}) conv({i_conv_blk}) {self.conv[idx_flat]}")
                    out = self.conv[idx_flat](out)
                    out = self.bn[idx_flat](out)
                    out = self.act[idx_flat](out)
                self.debug(
                    f"  block({i_blk}) out:{out.shape}, data:{out.detach().cpu()[0, 0, :10]}")
            r"One less pooling layer."
            if i_blk < self.n_blocks - 1:
                out = self.pool[i_blk](out)
                self.debug(f"  block({i_blk}) pool-out:{out.shape}")

        if self.gap_layer:
            out = self.gap(out)
            self.debug(f"  GAP out: {out.shape}")

        out = out.view(out.size(0), -1)
        self.debug(f'  flatten: {out.shape}')
        out = self.classifier(out)
        self.debug(f'  out: {out.shape}')
        self.iter += 1
        return out

    def calculate_hidden(self):
        return self.input_sz * self.out_channels[-1]

    def make_layers_deep(self):
        in_channels = self.low_conv_hidden_dim
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            layers_for_shortcut = []
            in_channel_for_shortcut = in_channels
            for _ in range(self.n_conv_layers_per_block):
                self.conv.append(
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        padding=padding_same(
                            input=self.input_sz,
                            kernel=self.kernels[i],
                            stride=1,
                            dilation=1)
                    ))
                self.bn.append(nn.BatchNorm1d(self._out_channels))
                self.act.append(nn.ReLU(inplace=True))
                if self.shortcut_conn:
                    layers_for_shortcut.extend([
                        self.conv[-1], self.bn[-1],
                    ])
                in_channels = self._out_channels

            if self.shortcut_conn:
                self.shortcut.append(
                    ShortcutBlock(
                        layers=nn.Sequential(*layers_for_shortcut),
                        in_channels=in_channel_for_shortcut,
                        out_channels=self._out_channels,
                        point_conv_group=self.conv_groups[i])
                )
            if i < self.n_blocks - 1:
                self.pool.append(nn.MaxPool1d(2, stride=2))
                self.input_sz //= 2
        # return nn.Sequential(*layers)
        r"If shortcut_conn is true, empty conv, and bn module-list. \
        This may be necessary to not to calculate gradients for the \
        same layer twice."
        if self.shortcut_conn:
            self.conv = nn.ModuleList([])
            self.bn = nn.ModuleList([])
            self.act = nn.ModuleList([])

    def make_low_conv(self):
        r"""Make low convolution block."""
        layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = self.in_channels
        for x in self.low_conv_options["cfg"]:
            if x == 'M':
                count_pooling += 1
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.input_sz /= 2
            else:
                stride_ = self.low_conv_options["stride"][i_kernel]
                conv = nn.Conv1d(
                    in_channels=in_chan, out_channels=x,
                    kernel_size=self.low_conv_options["kernel"][i_kernel],
                    groups=self.low_conv_options["conv_groups"][i_kernel],
                    stride=stride_,
                    padding=padding_same(
                        input=self.input_sz,
                        kernel=self.low_conv_options["kernel"][i_kernel],
                        stride=1,
                        dilation=1),
                )
                layers += [
                    conv,
                    nn.BatchNorm1d(x),
                    nn.ReLU(inplace=True)
                    ]
                in_chan = self.low_conv_hidden_dim = x
                self.input_sz /= self.low_conv_options["stride"][i_kernel]
                i_kernel += 1
            pass    # for
        return nn.Sequential(*layers)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class MSTNet(nn.Module):
    def __init__(
            self, d_input, d_output=2, d_model=256, n_layers=4, dropout=0.2, log=print) -> None:
        super().__init__()

        self.iter = 0
        self.log = log
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_input//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_input//2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_chan = 1
        out_chan = 8
        kernel_sz = 51
        dilation_sz = 1

        self.pre_conv = nn.Conv1d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, bias=False)
        for _ in range(n_layers):
            self.conv_layers.append(
                S4D(d_model=d_model, dropout=dropout, transposed=True, lr=0.001)
                # nn.Conv1d(
                #     in_channels=out_chan,
                #     out_channels=out_chan,
                #     kernel_size=kernel_sz,
                #     dilation=dilation_sz,
                #     padding=padding_same(input=d_model, kernel=kernel_sz, dilation=dilation_sz))
            )
            # self.norms.append(nn.BatchNorm1d(d_model))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
            in_chan = out_chan
        self.norms = nn.ModuleList()

        self.decoder = nn.Linear(d_model, d_output)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_output)
        )

    def forward(self, x):
        self.debug(f"input: {x.shape}") 
        
        x = self.encoder(x)
        self.debug(f"encoder: {x.shape}") 

        x = x.reshape(x.size()[0], 1, -1)
        self.debug(f"encoder reshape: {x.shape}") 

        x = F.relu(self.pre_conv(x))
        for layer, norm, dropout in zip(self.conv_layers, self.norms, self.dropouts):
            z = x
            z = norm(z)            
            z = layer(z)
            z = dropout(z)
            x = z + x
            # x = F.relu(x)
        
        # x = x.transpose(-1, -2)
        self.debug(f"conv out: {x.shape}") 
        x = x.mean(dim=1)
        self.debug(f"x mean: {x.shape}") 

        x = self.decoder(x)
        self.debug(f"decoder: {x.shape}") 

        self.iter += 1
        return x

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class RRNet(nn.Module):
    def __init__(self, d_input=10, d_output=2, d_model=10, dropout=0.1, log=print) -> None:
        super().__init__()

        self.iter = 0
        self.log = log
        self.hidden_dim = d_model
        self.input_bn = nn.BatchNorm1d(1)
        # self.encoder = nn.Sequential(
        #     nn.Linear(d_input, d_model),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout)
        # )
        self.gru_module = nn.GRU(
            input_size=d_input,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.gru_bn = nn.BatchNorm1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model//3, d_output)
        )

    def forward(self, x):
        self.debug(f"input: {x.shape}") 
        x = self.input_bn(x)
        out = x

        # out = x.view(x.size(0), -1)
        # out = self.encoder(out)
        # self.debug(f"encoder: {out.shape}") 

        # out = out.view(out.size(0), out.size(2), -1)
        # self.debug(f"out reshape:{out.shape}")
        weight = next(self.parameters()).data
        n_layers, n_directions = 2, 2
        self.hidden = (weight.new(n_layers*n_directions, out.shape[0], self.hidden_dim)
        ).zero_().to(out.device)
        _, self.hidden = self.gru_module(out, self.hidden)
        self.debug(f"gru hidden:{self.hidden.shape}")
        final_state = self.hidden.view(
            n_layers, n_directions, out.shape[0], self.hidden_dim
        )
        self.debug(f"pre final_state:{final_state.shape}")
        final_state = final_state[-1]
        self.debug(f"final-state:{final_state.shape}")
        h_1, h_2 = final_state[0], final_state[1]
        out = h_1 + h_2
        self.debug(f"*final-state reshape:{out.shape}")
        self.hidden = self.hidden.detach()

        out = self.gru_bn(out)

        out = out.view(out.size(0), -1)

        out = self.decoder(out)
        self.debug(f"decoder: {out.shape}") 
        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")    



class MultiScaleTemporalNet(nn.Module):
    def __init__(
            self, segment_sz=None, in_chan=1, n_classes=None, temporal_options=None, log=print) -> None:
        super().__init__()
        self.iter = 0
        self.in_chan = in_chan
        self.segment_sz = segment_sz
        self.temporal_options = temporal_options
        self.log = log

        self.input_bn = nn.BatchNorm1d(in_chan)
        self.temporal_conv_branches = self.make_temporal_conv()
        self.temporal_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.temporal_options["temporal_scale"][0], self.temporal_options["temporal_scale_target"][0]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)),
            nn.Sequential(
                nn.Linear(self.temporal_options["temporal_scale"][1], self.temporal_options["temporal_scale_target"][1]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)),
            nn.Sequential(
                nn.Linear(self.temporal_options["temporal_scale"][2], self.temporal_options["temporal_scale_target"][2]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)),
            nn.Sequential(
                nn.Linear(self.temporal_options["temporal_scale"][3], self.temporal_options["temporal_scale_target"][3]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)),    
        ])
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_options["classifier_neurons"][0][0], self.temporal_options["classifier_neurons"][0][1]),
            nn.BatchNorm1d(self.temporal_options["classifier_neurons"][0][1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.temporal_options["classifier_neurons"][1][0], self.temporal_options["classifier_neurons"][1][1]),
            nn.BatchNorm1d(self.temporal_options["classifier_neurons"][1][1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.temporal_options["classifier_neurons"][1][1], n_classes)
        ) 
    
    def forward(self, x):
        self.debug(f"input:{x.shape}") 
        x = self.input_bn(x)
        out = None
        for i, temporal_branch in enumerate(self.temporal_conv_branches):
            br_out = temporal_branch(x)
            self.debug(f"x_{i} conv_out:{br_out.shape}")
            # br_out = torch.mean(br_out, 1, True)
            br_out = br_out.view(br_out.size(0), -1)
            self.debug(f"x_{i} flatten:{br_out.shape}")
            br_out = self.temporal_scale[i](br_out)
            self.debug(f"x_{i} scale:{br_out.shape}")
            out = br_out if out is None else torch.cat((out, br_out), 1)
        self.debug(f"out-cat:{out.shape}")
        out = self.classifier(out)
        self.debug(f"out classif:{out.shape}")
        self.iter += 1
        return out
            
    def make_temporal_conv(self):
        layers_sm = []
        layers_medium = []
        layers_large = []
        layers_xl = []
        in_chan = self.in_chan
        alt_max_pool = True
        for i in range(len(self.temporal_options["out_chan"])):
            kr, dilation = self.temporal_options["kernel_sz_sm"][i]
            layers_sm += [
                nn.Conv1d(
                    in_channels=in_chan,
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    stride=self.temporal_options["stride"][i],
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=self.temporal_options["out_chan"][i],
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
            ]
            kr, dilation = self.temporal_options["kernel_sz_medium"][i]
            layers_medium += [
                nn.Conv1d(
                    in_channels=in_chan,
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    stride=self.temporal_options["stride"][i],
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=self.temporal_options["out_chan"][i],
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True)
            ]
            kr, dilation = self.temporal_options["kernel_sz_large"][i]
            layers_large += [
                nn.Conv1d(
                    in_channels=in_chan,
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    stride=self.temporal_options["stride"][i],
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=self.temporal_options["out_chan"][i],
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True)
            ]
            kr, dilation = self.temporal_options["kernel_sz_xl"][i]
            layers_xl += [
                nn.Conv1d(
                    in_channels=in_chan,
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    stride=self.temporal_options["stride"][i],
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=self.temporal_options["out_chan"][i],
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    dilation=dilation,
                    padding=padding_same(
                            input=self.segment_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True)
            ]
            # if alt_max_pool:
            layers_sm += [
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            layers_medium += [
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            layers_large += [
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            layers_xl += [
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            # alt_max_pool = not alt_max_pool
            in_chan = self.temporal_options["out_chan"][i]
        return nn.ModuleList([
            nn.Sequential(*layers_sm),
            nn.Sequential(*layers_medium),
            nn.Sequential(*layers_large),
            nn.Sequential(*layers_xl),
        ])

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")



class HybridConv(nn.Module):
    r"""Convolution network."""

    def __init__(
        self, segment_sz, kernels=None, padding=None, dilation=None, in_channels=None, out_channels=None,
        conv_groups=None, n_conv_layers_per_block=1, n_blocks=2,
        n_classes=None, low_conv_options=None, shortcut_conn=False, log=print,
        temporal_options=None, options=None
    ):
        r"""Instance of convnet."""
        super(HybridConv, self).__init__()

        log(
            f"segment_sz:{segment_sz}, kernels:{kernels}, in-chan:{in_channels}, "
            f"out-chan:{out_channels}, conv-gr:{conv_groups}, "
            f"n-conv-layer-per-block:{n_conv_layers_per_block}, "
            f"n_block:{n_blocks}, n_class:{n_classes}, "
            f"low-conv:{low_conv_options}, shortcut:{shortcut_conn}")

        self.iter = 0
        self.log = log
        self.input_sz = self.segment_sz = segment_sz
        self.kernels = kernels
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_groups = conv_groups
        self.n_conv_layers_per_block = n_conv_layers_per_block
        self.n_blocks = n_blocks
        self.low_conv_options = low_conv_options
        self.shortcut_conn = shortcut_conn
        self.temporal_options = temporal_options
        self.options = options

        self.input_bn = nn.BatchNorm1d(self.in_channels)
        self.input2_bn = nn.BatchNorm1d(1)
        # self.low_conv = self.make_low_conv()
        self.low_conv_hidden_dim = 10
        # self.features = self.make_layers_deep()

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.shortcut = nn.ModuleList([])
        # self.make_layers_deep()
        self.features = self.make_layers_deep_compact()
        self.features_nogroup = self.make_layers_deep_compact(group_conv=False)
        self.feature_scale = nn.Sequential(
            nn.Linear(options["feature_scale_h"][0], 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3))
        self.feature_scale_nogroup = nn.Sequential(
            nn.Linear(options["feature_scale_h"][0], 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3))
        
        self.temporal_conv_branches = self.make_temporal_conv()
        self.temporal_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.temporal_options["temporal_scale"][0], 100),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)),
            nn.Sequential(
                nn.Linear(self.temporal_options["temporal_scale"][1], 100),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)),
            nn.Sequential(
                nn.Linear(self.temporal_options["temporal_scale"][2], 100),
                nn.ReLU(inplace=True),    
                nn.Dropout(0.3)),
        ])
        

        # n_hidden = self.calculate_hidden()
        self.classifier = nn.Sequential(
            nn.Linear(self.options["classifier_neuron_h"][0], 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(50, 2)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.classifier = nn.Linear(self._out_channels+1, n_classes)

    def name(self):
        return (
            f"{self.__class__.__name__}_"
            f"segsz{self.segment_sz}_scut{self.shortcut_conn}_"
            f"lccfg{'x'.join(str(x) for x in self.low_conv_options['cfg'])}_"
            f"lckr{'x'.join(str(x) for x in self.low_conv_options['kernel'])}_"
            f"lcst{'x'.join(str(x) for x in self.low_conv_options['stride'])}_"
            f"lccg{'x'.join(str(x) for x in self.low_conv_options['conv_groups'])}_"
            f"blk{self.n_blocks}_cpblk{self.n_conv_layers_per_block}_"
            f"kr{'x'.join(str(x) for x in self.kernels)}_"
            f"och{'x'.join(str(x) for x in self.out_channels)}_"
            f"cg{'x'.join(str(x) for x in self.conv_groups)}")
        # return f'{self.__class__.__name__}'

    def forward(self, x):
        r""" x is stacked 1 sec signals, x2 is 10 sec signal """
        self.debug(f'  input:{x.shape}')

        "(ch, sig) = (20, sig), flat last 20-ch to form a long signal."
        b, c, _ = x.size()
        x_orig = x 
        x = x_orig[:, :c//2]
        x2 = x_orig[:, c//2:]
        x2 = x2.view(x2.size(0), 1, -1)
        self.debug(f"x1:{x.shape}, x2:{x2.shape}")

        x = self.input_bn(x)
        x2 = self.input2_bn(x2)

        # out = self.low_conv(x)
        # out = x
        self.debug(f'X: {x.shape}')

        "Computation tree for X"
        # for i_blk in range(self.n_blocks):
        #     if self.shortcut_conn:
        #         out = self.shortcut[i_blk](out)
        #         self.debug(f"[block:{i_blk}] shortcut out: {out.shape}")
        #     else:
        #         for i_conv_blk in range(self.n_conv_layers_per_block):
        #             idx_flat = 2*i_blk+i_conv_blk
        #             out = self.conv[idx_flat](out)
        #             out = self.bn[idx_flat](out)
        #             out = self.act[idx_flat](out)
        #             self.debug(
        #                 f"  block({i_blk}), conv({i_conv_blk}):{self.conv[idx_flat]}, out:{out.shape}")
        #         # self.debug(
        #         #     f"  block({i_blk}) out:{out.shape}, data:{out.detach().cpu()[0, 0, :10]}")
        #     r"One less pooling layer."
        #     if i_blk < self.n_blocks - 1:
        #         out = self.pool[i_blk](out)
        #         self.debug(f"  block({i_blk}) pool-out:{out.shape}")
        out_feat = self.features(x)
        self.debug(f"Features out:{out_feat.shape}")
        out_feat = self.gap(out_feat)
        self.debug(f"  GAP out: {out_feat.shape}")
        out_feat = out_feat.view(out_feat.size(0), -1)
        self.debug(f'  flatten: {out_feat.shape}')
        out_feat = self.feature_scale(out_feat)
        self.debug(f'  feature_scale: {out_feat.shape}')

        self.debug(f'X: {x.shape}')
        out_feat_ng = self.features_nogroup(x)
        self.debug(f"Features no-group out:{out_feat_ng.shape}")
        out_feat_ng = self.gap(out_feat_ng)
        self.debug(f"  GAP out: {out_feat_ng.shape}")
        out_feat_ng = out_feat_ng.view(out_feat_ng.size(0), -1)
        self.debug(f'  flatten: {out_feat_ng.shape}')
        out_feat_ng = self.feature_scale_nogroup(out_feat_ng)
        self.debug(f'  feature_scale: {out_feat_ng.shape}')

        "Computation block for X2"
        out2 = None
        for i, temporal_branch in enumerate(self.temporal_conv_branches):
            br_out = temporal_branch(x2)
            self.debug(f'x2_{i} conv_out: {br_out.shape}')
            # br_out = torch.mean(br_out, 1, True)  # Channel wise average
            # self.debug(f'x2_{i} channel-mean: {br_out.shape}')
            br_out = br_out.view(br_out.size(0), -1)
            self.debug(f'x2_{i} flatten: {br_out.shape}')

            br_out = self.temporal_scale[i](br_out)
            self.debug(f'x2_{i} scale: {br_out.shape}')

            if out2 is None:
                out2 = br_out 
            else:
                out2 = torch.cat((out2, br_out), 1)
        
        out = torch.cat((out_feat, out_feat_ng, out2), 1)
        self.debug(f'out-cat: {out.shape}')


        out = self.classifier(out)
        self.debug(f'  out: {out.shape}')
        self.iter += 1
        return out
    
    def make_temporal_conv(self):
        layers_large = []
        layers_medium = []
        layers_small = []
        in_chan = self.temporal_options["in_chan"][0]
        for i in range(len(self.temporal_options["out_chan"])):
            kr, dilation = self.temporal_options["kernel_sz_large"][i]
            layers_large += [
                nn.Conv1d(
                    in_channels=in_chan,
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    stride=self.temporal_options["stride"][i],
                    dilation=dilation,
                    padding=padding_same(
                            input=self.input_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=self.temporal_options["out_chan"][i],
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    stride=1,
                    dilation=1,
                    padding=padding_same(
                            input=self.input_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),                
            ]
            kr, dilation = self.temporal_options["kernel_sz_medium"][i]
            layers_medium += [
                nn.Conv1d(
                    in_channels=in_chan,
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    stride=self.temporal_options["stride"][i],
                    dilation=dilation,
                    padding=padding_same(
                            input=self.input_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=self.temporal_options["out_chan"][i],
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    stride=1,                    
                    dilation=1,
                    padding=padding_same(
                            input=self.input_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
            ]
            kr, dilation = self.temporal_options["kernel_sz_small"][i]
            layers_small += [
                nn.Conv1d(
                    in_channels=in_chan,
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    stride=self.temporal_options["stride"][i],                    
                    dilation=dilation,
                    padding=padding_same(
                            input=self.input_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=self.temporal_options["out_chan"][i],
                    out_channels=self.temporal_options["out_chan"][i],
                    kernel_size=kr,
                    stride=1,
                    dilation=1,
                    padding=padding_same(
                            input=self.input_sz,
                            kernel=kr,
                            stride=1,
                            dilation=dilation) if self.temporal_options["padding"][i]==-1 else self.temporal_options["padding"][i]),
                nn.ReLU(inplace=True),
            ]
            if i < len(self.temporal_options["out_chan"]) -1:
                layers_large += [
                    nn.MaxPool1d(kernel_size=2, stride=2)
                ]
                layers_medium += [
                    nn.MaxPool1d(kernel_size=2, stride=2)
                ]
                layers_small += [
                    nn.MaxPool1d(kernel_size=2, stride=2)
                ]
                in_chan = self.temporal_options["out_chan"][i]
        return nn.ModuleList([
            nn.Sequential(*layers_small), 
            nn.Sequential(*layers_medium), 
            nn.Sequential(*layers_large)
            ])
        

    def calculate_hidden(self):
        return self.input_sz * self.out_channels[-1]

    def make_layers_deep_compact(self, group_conv=True):
        in_channels = self.low_conv_hidden_dim
        layers = []
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            for _ in range(self.n_conv_layers_per_block):
                _padd = self.padding[i]
                layers += [
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i] if group_conv else 1,
                        dilation=self.dilation[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        padding=padding_same(
                            input=self.input_sz,
                            kernel=self.kernels[i],
                            stride=1,
                            dilation=1) if _padd == -1 else _padd),
                    nn.BatchNorm1d(self._out_channels),
                    nn.ReLU(inplace=True)
                    ]
                in_channels = self._out_channels
            if i < self.n_blocks - 1:
                layers += [nn.MaxPool1d(2, stride=2)]
                self.input_sz //= 2
        return nn.Sequential(*layers)


    def make_layers_deep(self):
        in_channels = self.low_conv_hidden_dim
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            layers_for_shortcut = []
            in_channel_for_shortcut = in_channels
            for _ in range(self.n_conv_layers_per_block):
                _padd = self.padding[i]
                self.conv.append(
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i],
                        dilation=self.dilation[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        padding=padding_same(
                            input=self.input_sz,
                            kernel=self.kernels[i],
                            stride=1,
                            dilation=1) if _padd == -1 else _padd 
                    ))
                self.bn.append(nn.BatchNorm1d(self._out_channels))
                self.act.append(nn.ReLU(inplace=True))
                if self.shortcut_conn:
                    layers_for_shortcut.extend([
                        self.conv[-1], self.bn[-1],
                    ])
                in_channels = self._out_channels

            if self.shortcut_conn:
                self.shortcut.append(
                    ShortcutBlock(
                        layers=nn.Sequential(*layers_for_shortcut),
                        in_channels=in_channel_for_shortcut,
                        out_channels=self._out_channels,
                        point_conv_group=self.conv_groups[i])
                )
            if i < self.n_blocks - 1:
                self.pool.append(nn.MaxPool1d(2, stride=2))
                self.input_sz //= 2
        # return nn.Sequential(*layers)
        r"If shortcut_conn is true, empty conv, and bn module-list. \
        This may be necessary to not to calculate gradients for the \
        same layer twice."
        if self.shortcut_conn:
            self.conv = nn.ModuleList([])
            self.bn = nn.ModuleList([])
            self.act = nn.ModuleList([])

    def make_low_conv(self):
        r"""Make low convolution block."""
        layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = self.in_channels
        for x in self.low_conv_options["cfg"]:
            if x == 'M':
                count_pooling += 1
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.input_sz /= 2
            else:
                stride_ = self.low_conv_options["stride"][i_kernel]
                conv = nn.Conv1d(
                    in_channels=in_chan, out_channels=x,
                    kernel_size=self.low_conv_options["kernel"][i_kernel],
                    groups=self.low_conv_options["conv_groups"][i_kernel],
                    stride=stride_,
                    padding=padding_same(
                        input=self.input_sz,
                        kernel=self.low_conv_options["kernel"][i_kernel],
                        stride=1,
                        dilation=1) if stride_==1 else 0,
                )
                layers += [
                    conv,
                    nn.BatchNorm1d(x),
                    nn.ReLU(inplace=True)
                    ]
                in_chan = self.low_conv_hidden_dim = x
                self.input_sz /= self.low_conv_options["stride"][i_kernel]
                i_kernel += 1
            pass    # for
        return nn.Sequential(*layers)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class MKDNet(nn.Module):
    def __init__(self, options, log=print) -> None:
        super().__init__()
        self.iter = 0
        self.options = options
        self.log = log

        self.bn0 = nn.BatchNorm1d(1, momentum=0.05)
        self.low_conv = self.create_low_conv(options)
        self.encoder = self.create_encoder(options)
        self.classifier = self.create_classifier(options)

    def forward(self, x):
        out = self.bn0(x)
        self.debug(f"input: {out.shape}")
        out = self.low_conv(out)
        self.debug(f"low_conv: {out.shape}")
        out = self.encoder(out)
        self.debug(f"encoder: {out.shape}")
        out = out.view(out.size(0), -1)  # reshape/flatten
        self.debug(f"  flatten: {out.shape}")
        out = self.classifier(out)
        self.debug(f"classifier: {out.shape}")
        self.iter += 1
        return out

    def create_low_conv(self, options):
        layers = []
        ch_in, ch_out, kr = 1, 8, 21, 
        # layers += [
        #     nn.Conv1d(
        #         in_channels=ch_in,
        #         out_channels=ch_out,
        #         kernel_size=kr,
        #         stride=2),
        #     nn.BatchNorm1d(ch_out),
        #     nn.ReLU(inplace=True)
        # ]
        layers += [
            nn.Conv1d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kr,
                stride=2,
                # padding=padding_same(
                #     input=ch_in,
                #     kernel=kr),
            ),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        ]
        self.out_sz = options["input_sz"]//2 - 10  # a general formula possible?
        return nn.Sequential(*layers)


    def create_encoder(self, options):
        layers = []
        groups = 1
        ch_out = 2 * 3 * 8
        layers += [
            MKDConv(
                input_size=self.out_sz, in_channels=8, conv_kernel=[5, 11], conv_stride=1,
                dilation_factor=[1, 2, 4], out_channels=8, log=print,
                add_unit_kernel=False, groups=groups
            ),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, stride=2)
        ]
        self.out_sz //= 2
        groups = 6
        layers += [
            MKDConv(
                input_size=self.out_sz, in_channels=ch_out, conv_kernel=[5, 11], conv_stride=1,
                dilation_factor=[1, 2, 4], out_channels=8, log=print,
                add_unit_kernel=False, groups=groups
            ),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, stride=2)
        ]
        self.out_sz //= 2
        self.hidden_sz = ch_out * self.out_sz
        return nn.Sequential(*layers)
    
    def create_classifier(self, options):
        return nn.Sequential(
            nn.Linear(self.hidden_sz, 1200),
            nn.Linear(1200, 600),
            nn.Linear(600, 2),
        )        

    def debug(self, *args):
        if self.iter == 0 and self.log:
            self.log(args)


class MKDConv(nn.Module):
    def __init__(
        self, input_size=None, in_channels=1, conv_kernel=5, conv_stride=1,
        dilation_factor=None, out_channels=1, log=None,
        add_unit_kernel=False, groups=1
    ):
        super(MKDConv, self).__init__()
        self.iter = 0
        self.log = log if log is not None else print
        self.input_size = input_size
        self.in_channels = in_channels
        self.dilation_factor = dilation_factor
        self.out_channels = out_channels
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.groups = groups
        self.add_unit_kernel = add_unit_kernel
        self.dilated_conv_filters = nn.ModuleList([])

        if self.add_unit_kernel:
            '''Always add a 1-kernel convolution (to provide identity connection?!)'''
            self.dilated_conv_filters.append(
                self.make_dilated_conv(kernel=1, dilation_factor=1)
            )

        for i_kr in range(len(self.conv_kernel)):
            for i_df in range(len(self.dilation_factor)):
                self.dilated_conv_filters.append(
                    self.make_dilated_conv(
                        self.conv_kernel[i_kr], self.dilation_factor[i_df])
                )

    def name(self):
        return f'{self.__class__.__name__}'

    def forward(self, x):
        self.debug(f'input: {x.shape}')
        _, c, _ = x.size()
        c_group = c // self.groups
        x_orig = x
        # x = self.pre_conv_spatial_scaling(x)
        for i, conv_filter in enumerate(self.dilated_conv_filters):
            if self.groups > 1:
                x = x_orig[:, i*c_group:(i+1)*c_group]
            if i == 0:
                out = conv_filter(x)
            else:
                out = torch.cat([out, conv_filter(x)], 1)
            self.debug(f"Groups:{i+1} of {self.groups}, x_slice:{x.shape}, out:{out.shape}")    
        self.iter += 1
        return out

    def make_dilated_conv(self, kernel, dilation_factor):
        conv1d = nn.Conv1d(
            in_channels=self.in_channels // self.groups,
            out_channels=self.out_channels,
            kernel_size=kernel,
            stride=self.conv_stride,
            padding=padding_same(
                input=self.input_size,
                kernel=kernel,
                stride=self.conv_stride,
                dilation=dilation_factor),
            dilation=dilation_factor,
            bias=False)
        return conv1d

    def debug(self, *args):
        if self.iter == 0:
            self.log(self.name(), args)


