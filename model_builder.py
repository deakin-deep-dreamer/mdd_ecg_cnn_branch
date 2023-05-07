import numpy as np
import torch

import models


def count_parameters(model):
    r"""Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(hz=100, seg_len_sec=10, in_chan=1, n_block=5, n_classes=2, log=print):
    return models.ClassicConv(
        segment_sz=hz*seg_len_sec, shortcut_conn=True,
        in_channels=in_chan,
        # kernels=[11 if i == 0 else 7 for i in range(N_BLOCK)],
        kernels=[21, 21, 11, 7, 5],
        # kernels=[5, 5, 5, 5, 5],
        # out_channels=[1*(2**i) for i in range(1, n_block+1)],
        out_channels=[32, 64, 128, 256, 512],
        conv_groups=[1 for i in range(n_block)],
        # conv_groups=[IN_CHAN for _ in range(N_BLOCK)],
        n_conv_layers_per_block=2, n_blocks=n_block, n_classes=n_classes,
        low_conv_options={
            "cfg": [8, 8, 'M'],
            'stride': [2, 1],
            "kernel": [21, 21],
            'conv_groups': [in_chan, in_chan]
            },
        log=log)


def create_rr_shallow_model(hz=100, seg_sz=10, n_block=1, n_classes=2, log=print):
    return models.ClassicConv(
        segment_sz=seg_sz, shortcut_conn=False,
        in_channels=1,
        kernels=[5, 5, 5, 5, 5],
        out_channels=[8, 8, 8, 8, 8],
        conv_groups=[1 for i in range(n_block)],
        n_conv_layers_per_block=2, n_blocks=n_block, n_classes=n_classes,
        gap_layer=False, hidden_neurons_classif=80,
        low_conv_options={
            "cfg": [1],
            'stride': [1],
            "kernel": [1],
            'conv_groups': [1]
            },
        log=log)


def create_multiscale_temporal_rr_model(hz=100, seg_len_sec=20, in_chan=1, n_classes=2, log=print):
    return models.MultiScaleTemporalNet(
        segment_sz=hz*seg_len_sec, in_chan=1, n_classes=n_classes, log=log, 
        temporal_options={
            "out_chan": [8, 8],
            "stride": [1, 1],
            "padding": [-1, -1],
            "kernel_sz_sm": [(3, 1), (3, 1)],
            "kernel_sz_medium": [(5, 1), (5, 1)],
            "kernel_sz_large": [(7, 1), (7, 1)],
            "kernel_sz_xl": [(9, 1), (9, 1)],
            "temporal_scale": [40, 40, 40, 40],
            "temporal_scale_target": [40, 40, 40, 40],
            "classifier_neurons": [(160, 100), (100, 50), (50, n_classes)]
        }
    )


def create_multiscale_temporal_model(hz=100, seg_len_sec=10, in_chan=10, n_classes=2, log=print):
    return models.MultiScaleTemporalNet(
        segment_sz=hz*seg_len_sec, in_chan=1, n_classes=n_classes, log=log, 
        temporal_options={
            "out_chan": [8, 8, 8],
            "stride": [1, 1, 1],
            "padding": [-1, -1, -1],
            "kernel_sz_sm": [(51, 2), (51, 2), (51, 1)],
            "kernel_sz_medium": [(51, 4), (51, 2), (51, 1)],
            "kernel_sz_large": [(51, 6), (51, 2), (51, 1)],
            "kernel_sz_xl": [(51, 8), (51, 1), (51, 1)],
            "temporal_scale": [800, 600, 400, 200],
        }
    )


def create_hybrid_model(hz=100, seg_len_sec=10, in_chan=10, n_block=3, n_classes=2, log=print):
    return models.HybridConv(
        segment_sz=hz*seg_len_sec, shortcut_conn=False,
        in_channels=in_chan,
        # kernels=[11 if i == 0 else 7 for i in range(N_BLOCK)],
        kernels=[11, 7, 3, 5],
        padding=[0, 0, 0, 0],
        dilation=[2, 1, 1, 1],
        # kernels=[5, 5, 5, 5, 5],
        # out_channels=[1*(2**i) for i in range(1, n_block+1)],
        out_channels=[80, 160, 320, 640],
        # conv_groups=[1 for i in range(n_block)],
        conv_groups=[10 for i in range(n_block)],
        n_conv_layers_per_block=2, n_blocks=n_block, n_classes=n_classes,
        low_conv_options={
            "cfg": [8, 8, 'M'],
            'stride': [2, 1],
            "padding": [0, 0],
            "kernel": [21, 21],
            'conv_groups': [in_chan, in_chan]
            },
        temporal_options={
            "in_chan": [1],
            "out_chan": [8, 8, 8],
            "stride": [1, 1, 1],
            "padding": [0, 0, 0],
            "kernel_sz_large": [(51, 4), (51, 2), (51, 1)],
            "kernel_sz_medium": [(51, 2), (51, 2), (51, 1)],
            "kernel_sz_small": [(25, 4), (25, 2), (25, 1)],
            "temporal_scale": [1088, 296, 96],
        },
        options={
            "feature_scale_h": [320],
            "feature_scale_nogroup_h": [320],
            "classifier_neuron_h": [500],
        },
        log=log)


def create_SincNet(
    conv0_out=4,
    kr_conv0_start=0,
    low_hz=0.5,
    high_hz=6,
    seg_sz=None,
    hz=None,
    n_classes=2,
    log=print,
    gru_module=False,
    sinc_conv=True,
):
    KERNEL = [21, 21, 5, 5, 5, 5, 5, 5, 5, 5]
    # CONV_STRIDE = [1 for _ in range(len(KERNEL))]
    CONV_STRIDE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    conv0_out_8 = 8
    # low_hz_list = np.linspace(low_hz, high_hz, conv0_out_8 + 1)[
    #     kr_conv0_start : kr_conv0_start + conv0_out + 1
    # ]
    # band_hz_list = np.tile((high_hz - low_hz) // conv0_out_8, conv0_out)
    low_hz_list = np.linspace(low_hz, high_hz, conv0_out+1)[:conv0_out+1]
    band_hz_list = np.tile(2, conv0_out)[:conv0_out]
    model_cfg = {
        "cfg": [
            conv0_out,
            32,
            "A",
            "M",
            64,
            64,
            "A",
            "M",
            128,
            128,
            "A",
            "M",
            256,
            256,
            "A",
            "M",
            256,
            256,
        ],
        "kernel": KERNEL,
        "stride": CONV_STRIDE,
        "kernel_group": [1 if i == 0 else conv0_out for i in range(len(KERNEL))],
        "fq_band_lists": (low_hz_list[:-1], band_hz_list),
        "n_class": n_classes,
        "input_sz": seg_sz,
        "hz": hz,
        "sinc_conv0": sinc_conv,
        "gap_layer": True,
        "flatten": True,
        "gru_module": gru_module,
    }
    net = models.SincNet(model_cfg, log=log)
    return net



def create_sincResNet(
    conv0_out=8,
    low_hz=0.5,
    high_hz=50,
    seg_sz=10*100,
    hz=100,
    n_classes=2,
    n_block=6,
    log=print,
    n_conv_per_blk=2,
    sinc_conv=True,
):
    KERNEL = [25, 25, 11, 7, 5, 5]
    DILATION = [2, 1, 1, 1, 1, 1]
    OUT_CHAN = [32, 64, 128, 256, 512, 512]
    # n_block = len(KERNEL)
    # low_hz_list = np.linspace(low_hz, high_hz, conv0_out + 1)
    low_hz_list = np.random.uniform(low_hz, high_hz, conv0_out + 1)
    # band_hz_list = np.tile((high_hz - low_hz) // conv0_out, conv0_out)
    band_hz_list = np.tile(np.random.randint(2, 5, 1)[0], conv0_out)
    net = models.SincResNet(
        segment_sz=seg_sz,
        kernels=KERNEL,
        dilations=DILATION,
        shortcut_conn=True,
        in_channels=1,
        out_channels=OUT_CHAN,
        conv_groups=(
            [conv0_out for _ in range(n_block)]
            if sinc_conv
            else [1 for _ in range(n_block)]
        ),
        n_conv_layers_per_block=n_conv_per_blk,
        n_blocks=n_block,
        n_classes=n_classes,
        log=log,
        low_conv_options={
            "cfg": [conv0_out, 16, "M"],
            "stride": [1, 1],
            "kernel": [51, 51],
            "dilation": [1, 2],
            "conv_groups": [1, conv0_out if sinc_conv else 1],
            "sinc_kernel": sinc_conv,
            "hz": hz,
            "fq_band_lists": (low_hz_list[:-1], band_hz_list),
        },
    )
    return net


def create_sincResNet(
    conv0_out=8,
    low_hz=0.5,
    high_hz=50,
    seg_sz=10*100,
    hz=100,
    n_classes=2,
    n_block=6,
    log=print,
    n_conv_per_blk=2,
    sinc_conv=True,
):
    KERNEL = [25, 13, 7, 5, 5, 5]
    DILATION = [1, 1, 1, 1, 1, 1]
    OUT_CHAN = [32, 64, 128, 256, 512, 512]
    # n_block = len(KERNEL)
    # low_hz_list = np.linspace(low_hz, high_hz, conv0_out + 1)
    low_hz_list = np.random.uniform(low_hz, high_hz, conv0_out + 1)
    # band_hz_list = np.tile((high_hz - low_hz) // conv0_out, conv0_out)
    band_hz_list = np.tile(np.random.randint(2, 5, 1)[0], conv0_out)
    net = models.SincResNet(
        segment_sz=seg_sz,
        kernels=KERNEL,
        dilations=DILATION,
        shortcut_conn=True,
        in_channels=1,
        out_channels=OUT_CHAN,
        conv_groups=(
            [conv0_out for _ in range(n_block)]
            if sinc_conv
            else [1 for _ in range(n_block)]
        ),
        n_conv_layers_per_block=n_conv_per_blk,
        n_blocks=n_block,
        n_classes=n_classes,
        log=log,
        low_conv_options={
            "cfg": [conv0_out, 16, "M"],
            "stride": [1, 1],
            "kernel": [51, 51],
            "dilation": [1, 1],
            "conv_groups": [1, conv0_out if sinc_conv else 1],
            "sinc_kernel": sinc_conv,
            "hz": hz,
            "fq_band_lists": (low_hz_list[:-1], band_hz_list),
        },
    )
    return net


def create_sinc_shallow_18240(
    conv0_out=32,
    seg_sz=None,
    hz=None,
    n_classes=2,
    log=print,
    gru_module=False,
    sinc_conv=True,
):
    KERNEL = [151, 151]
    # CONV_STRIDE = [1 for _ in range(len(KERNEL))]
    CONV_STRIDE = [5, 1]
    low_hz_list = np.random.uniform(0, hz//2, conv0_out+1)
    band_hz_list = np.tile(2, conv0_out)
    # conv0_out_8 = 8
    # low_hz_list = np.linspace(low_hz, high_hz, conv0_out_8 + 1)[
    #     kr_conv0_start : kr_conv0_start + conv0_out + 1
    # ]
    # band_hz_list = np.tile((high_hz - low_hz) // conv0_out_8, conv0_out)
    # # low_hz_list = np.linspace(low_hz, high_hz, conv0_out+1)[:conv0_out+1]
    # # band_hz_list = np.tile((high_hz-low_hz)//conv0_out, conv0_out)[:conv0_out]
    model_cfg = {
        "cfg": [
            conv0_out,
            "A",
            # "M",
            64,
            "A",
            "M",
            # 64,
            # "A",
            # "M",
        ],
        "kernel": KERNEL,
        "stride": CONV_STRIDE,
        "kernel_group": [1 if i == 0 else conv0_out for i in range(len(KERNEL))],
        "fq_band_lists": (low_hz_list[:-1], band_hz_list),
        "n_class": n_classes,
        "input_sz": seg_sz,
        "hz": hz,
        "sinc_conv0": sinc_conv,
        "gap_layer": False,
        "flatten": True,
        "gru_module": gru_module,
        "hidden_dim": 18240,
    }
    net = models.SincNet(model_cfg, log=log)
    return net


def create_model_multiDilatedKernel(
    input_sz=None
):
    net = models.MKDNet(
        options={
            "input_sz": input_sz,
        }
    )
    return net
    

def create_MSTNet(d_input=1000, d_output=2, d_model=256, n_layers=5, log=print):
    net = models.MSTNet(d_input=d_input, d_output=d_output, d_model=d_model, n_layers=n_layers, log=log)
    return net


def create_RRNet(d_input=10, d_output=2, d_model=10, dropout=0.1, log=print):
    net = models.RRNet(d_input=d_input, d_output=d_output, d_model=d_model, dropout=dropout, log=log)
    return net

def create_BranchConvNet(log=print):
    net = models.BranchConvNet({
        'kernels': [
            [(3, 8), (3, 8), 'M', (3, 16), (3, 16), 'M', (3, 32), (3, 32)],
            [(5, 8), (5, 8), 'M', (5, 16), (5, 16), 'M', (5, 32), (5, 32)],
            [(7, 8), (7, 8), 'M', (7, 16), (7, 16), 'M', (7, 32), (7, 32)]
        ],
        'feat_per_branch': 32,
        'h1': 32,
        'h2': 16,
        'input_sz': 10,
    }, log=log)
    return net


def create_BranchConvResNet(log=print, hz=100, seg_sec=30):
    net = models.BranchConvResNet({
        'kernels': [
            [(21, 2, 1, 16), (21, 2, 1, 16), 'M', (11, 1, 1, 32), (11, 1, 1, 32), 'R', 'M', (7, 1, 1, 64), (7, 1, 1, 64), 'R', 'M', (5, 1, 1, 128), (5, 1, 1, 128), 'R', 'M', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
            [(21, 2, 2, 16), (21, 2, 2, 16), 'M', (11, 1, 2, 32), (11, 1, 2, 32), 'R', 'M', (7, 1, 2, 64), (7, 1, 2, 64), 'R', 'M', (5, 1, 2, 128), (5, 1, 2, 128), 'R', 'M', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
            [(21, 2, 4, 16), (21, 2, 4, 16), 'M', (11, 1, 4, 32), (11, 1, 4, 32), 'R', 'M', (7, 1, 2, 64), (7, 1, 2, 64), 'R', 'M', (5, 1, 2, 128), (5, 1, 2, 128), 'R', 'M', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        ],
        'feat_per_branch': 128,
        # 'h1': 128,
        # 'h2': 32,
        'h': (128*3, 128, 32),
        'input_sz': seg_sec*hz,
    }, log=log)
    return net


# def create_BranchRawAndRRConvResNet(log=print, hz=100, seg_sec=30):
#     net = models.BranchRawAndRRConvResNet({
#         'kernels': [
#             [(21, 2, 1, 16), (21, 2, 1, 16), 'M', (11, 1, 1, 32), (11, 1, 1, 32), 'R', 'M', (7, 1, 1, 64), (7, 1, 1, 64), 'R', 'M', (5, 1, 1, 128), (5, 1, 1, 128), 'R', 'M', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
#             [(21, 2, 2, 16), (21, 2, 2, 16), 'M', (11, 1, 2, 32), (11, 1, 2, 32), 'R', 'M', (7, 1, 2, 64), (7, 1, 2, 64), 'R', 'M', (5, 1, 2, 128), (5, 1, 2, 128), 'R', 'M', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
#             [(21, 2, 4, 16), (21, 2, 4, 16), 'M', (11, 1, 4, 32), (11, 1, 4, 32), 'R', 'M', (7, 1, 2, 64), (7, 1, 2, 64), 'R', 'M', (5, 1, 2, 128), (5, 1, 2, 128), 'R', 'M', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
#         ],
#         'kernels_rr': [
#             [(3, 1, 1, 32), (3, 1, 1, 32), 'M', (3, 1, 1, 64), (3, 1, 1, 64), 'R', 'M', (3, 1, 1, 128), (3, 1, 1, 128), 'R'],
#             [(3, 1, 2, 32), (3, 1, 2, 32), 'M', (3, 1, 2, 64), (3, 1, 2, 64), 'R', 'M', (3, 1, 1, 128), (3, 1, 1, 128), 'R'],
#         ],
#         'feat_per_branch': 128,
#         # 'h1': 128,
#         # 'h2': 32,
#         'h': (128*5,),
#         'input_sz': seg_sec*hz,
#         'input_sz_rr': seg_sec,
#         'n_classes': 2,
#     }, log=log)
#     return net

def create_AutoEncTcn(options, log=print):
    return models.AutoEncTcn(options=options, log=log)

def test_create_AutoEncTcn():
    net = create_AutoEncTcn({
        'kernels': [
            (21, 1, 1, 16), (21, 1, 1, 16), 'R', 'M', (11, 1, 1, 32), (11, 1, 1, 32), 'R', 'M'
        ],
        'dropout': 0.2,
        # 'h': (250, 410),
        'h': (500, 660),
    })
    print(net)
    x = torch.randn(32, 1, 2000)
    out, out_refined, out_decoder = net(x)
    print(f"out: {out.shape}, out*:{out_refined.shape}, decoder: {out_decoder.shape}")


def create_base_PyramidNet(log=print):
    group = 32
    net = models.PyramidNet({
        "hz": 500,
        "sinc_conv0": group,
        "fq_band_lists": [100*np.random.uniform(0, 0.5, group), [2]*group],
        "layers": [
            (5, 1, 1, 32, group),
            (5, 1, 1, 32, group), 'M',
            (5, 1, 1, 64, group),
            (5, 1, 1, 64, group), 'M',
            (5, 1, 1, 128, group),
            (5, 1, 1, 128, group), 'M',
            (5, 1, 1, 256, group),
            (5, 1, 1, 256, group), 'M',
        ],
        "h": (19968, 250, 100)
    }, log=log)
    return net

def test_create_base_PyramidNet():
    net = create_base_PyramidNet()
    print(net)
    x = torch.randn(32, 1, 500*20)
    out = net(x)
    print(f"out: {out.shape}")
    print(count_parameters(net), "parameters")


def create_BranchRawAndRRConvResNet(options, log=print):
    return models.BranchRawAndRRConvResNet(options, log=log)


def test_create_BranchRawAndRRConvResNet():
    net = create_BranchRawAndRRConvResNet({
        'kernels_0': [
            [(21, 1, 1, 32), (21, 1, 1, 32), 'R', 'M', (11, 1, 1, 64), (11, 1, 1, 64), 'R', 'M', (7, 1, 1, 128), (7, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        ],
        'kernels_1': [
            [(3, 1, 1, 32), (3, 1, 1, 32), 'R', (3, 1, 1, 64), (3, 1, 1, 64), 'R', (3, 1, 1, 128), (3, 1, 1, 128), 'R', (3, 1, 1, 256), (3, 1, 1, 256), 'R'],
        ],
        'feat_per_branch': 128,
        'h': (128*2),
        'n_classes': 2,
        'dropout': 0.2,
        'hz': 100,
    }, log=print)

    print(net)
    x = torch.randn(32, 2, 1000)
    out = net(x)
    print(f"out: {out.shape}")



def create_BranchRawConvResNet(options, log=print):
    return models.BranchRawAndRRConvResNet(options, log=log)


def test_create_BranchRawAndRRConvResNet():
    net = create_BranchRawConvResNet({
        'kernels_0': [
            [(17, 1, 1, 32), (17, 1, 1, 32), 'R', 'M', (5, 1, 1, 64), (5, 1, 1, 64), 'R', 'M', (5, 1, 1, 128), (5, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        ],
        'kernels_1': [
            [(3, 1, 1, 32), (3, 1, 1, 32), 'R', (3, 1, 1, 64), (3, 1, 1, 64), 'R', (3, 1, 1, 128), (3, 1, 1, 128), 'R', (3, 1, 1, 256), (3, 1, 1, 256), 'R'],
        ],
        'feat_per_branch': 128,
        'h': (128*2),
        'n_classes': 2,
        'dropout': 0.2,
        'hz': 100,
    }, log=print)

    print(net)
    x = torch.randn(32, 2, 500*20)
    out = net(x)
    print(f"out: {out.shape}")


def test_BranchRawAndRRConvResNet():
    net = create_BranchRawAndRRConvResNet({
        'kernels_0': [
            [(21, 1, 1, 32), (21, 1, 1, 32), 'R', 'M', (11, 1, 1, 64), (11, 1, 1, 64), 'R', 'M', (7, 1, 1, 128), (7, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        ],
        'kernels_1': [
            [(21, 1, 1, 32), (21, 1, 1, 32), 'R', 'M', (11, 1, 1, 64), (11, 1, 1, 64), 'R', 'M', (7, 1, 1, 128), (7, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        ],
        'feat_per_branch': 128,
        'h': (128*2),
        'n_classes': 2,
        'dropout': 0.2,
        'hz': 100,
    }, log=print)

    print(net)
    x = torch.randn(32, 2, 1000)
    out = net(x)
    print(f"out: {out.shape}")

    
def test_create_BranchConvNet():
    net = create_BranchConvNet()
    print(net)
    x = torch.randn(32, 1, 30)
    out = net(x)
    print(f"out: {out.shape}")


def test_create_BranchConvResNet():
    net = create_BranchConvResNet()
    print(net)
    x = torch.randn(32, 1, 30*100)
    out = net(x)
    print(f"out: {out.shape}")


def test_create_BranchRawAndRRConvNet():
    net = create_BranchRawAndRRConvNet()
    print(net)
    x = torch.randn(32, 2, 30*100)
    out = net(x)
    print(f"out: {out.shape}")    

   


def test_create_RRNet():
    net = create_RRNet()
    print(net)
    x = torch.randn(32, 1, 10)
    out = net(x)
    print(f"out: {out.shape}")

def testMSTNet():
    net = create_MSTNet()
    print(net)
    x = torch.randn(32, 1, 1000)
    out = net(x)
    print("out:", out.shape)


def test_create_multiscale_temporal_rr_model():
    net = create_multiscale_temporal_rr_model()
    print(net)

    x = torch.randn(32, 1, 20)
    out = net(x)
    print("out:", out.shape)


def test_create_hybrid_model():
    net = create_hybrid_model()
    print(net)

    x = torch.randn(1, 20, 100)
    out = net(x)
    print("out:", out.shape)

def test_create_rr_shallow_model():
    net = create_rr_shallow_model()
    print(net)
    x = torch.randn(32, 1, 10)
    out = net(x)
    print("out:", out.shape)


def test_create_model_few_kernel___():
    Hz = 100
    SEG_SEC = 10
    CONV0_OUT_CHAN = 16
    model = create_model_few_kernel(
        conv0_out=CONV0_OUT_CHAN,
        hz=Hz,
        seg_sz=Hz * SEG_SEC,
        high_hz=50,
        gru_module=True
    )
    print(model)

    x = torch.randn(1, 1, Hz * SEG_SEC)
    (out,) = model(x)[0]
    # print(out)
    print(f"out: {out.shape}")

    # if model.options.get("sinc_conv0"):
    #     r"Store band-pass filter optimisation."
    #     low_hz = model.conv[0].low_hz_.data.detach().cpu().numpy()[:, 0]
    #     band_hz = model.conv[0].band_hz_.data.detach().cpu().numpy()[:, 0]
    #     fband_line = ""
    #     for i_sinc_chan in range(CONV0_OUT_CHAN):
    #         fband_line += f"{low_hz[i_sinc_chan]:.02f}, {band_hz[i_sinc_chan]:.02f}, "
    #     r"Remove trailing ,"
    #     fband_line = fband_line[: fband_line.rfind(",")]
    #     print(fband_line)


def test_create_SincNet():
    Hz = 1000 // 10
    SEG_SEC = 10
    CONV0_OUT_CHAN = 16
    model = create_SincNet(
        conv0_out=CONV0_OUT_CHAN,
        kr_conv0_start=0,
        hz=Hz,
        seg_sz=Hz * SEG_SEC,
        low_hz=0.5,
        high_hz=64,
        gru_module=True
    )
    print(model)

    x = torch.randn(1, 1, Hz * SEG_SEC)
    (out,) = model(x)[0]
    # print(out)
    print(f"out: {out.shape}")

    if model.options.get("sinc_conv0"):
        r"Store band-pass filter optimisation."
        low_hz = model.conv[0].low_hz_.data.detach().cpu().numpy()[:, 0]
        band_hz = model.conv[0].band_hz_.data.detach().cpu().numpy()[:, 0]
        fband_line = ""
        for i_sinc_chan in range(CONV0_OUT_CHAN):
            fband_line += f"{low_hz[i_sinc_chan]:.02f}, {band_hz[i_sinc_chan]:.02f}, "
        r"Remove trailing ,"
        fband_line = fband_line[: fband_line.rfind(",")]
        print(fband_line)

    # for p in model.parameters():
    #     print(p.name, p.data, p.requires_grad)

    # for n, p in model.named_parameters():
    #     print(n, p.data, p.requires_grad)

    print("direct access:", model.conv[0].low_hz_.data)


def test_multiDilatedKernel():
    x = torch.randn(32, 8, 0)
    net = models.MKDConv(
        input_size=x.size()[-1], in_channels=x.size()[1], conv_kernel=[11, 22], conv_stride=1,
        dilation_factor=[1], out_channels=8, log=None,
        add_unit_kernel=False, groups=2)
    print(net)
    out = net(x)
    print(out.shape)


def test_model():
    net = create_model(seg_len_sec=20)
    print(net)
    x = torch.randn(5, 1, 2000)
    out = net(x)
    print("out:", out.shape)



def test_sincResNet():
    Hz = 100
    model = create_sincResNet(
        low_hz=0.5, high_hz=Hz, seg_sz=10 * Hz, hz=Hz, n_classes=2
    )
    print(model)
    x = torch.randn(32, 1, Hz * 10)
    out = model(x)
    # print(out)
    print(f"out: {out.shape}")


def main():
    test_create_BranchRawAndRRConvResNet()


if __name__ == "__main__":
    main()