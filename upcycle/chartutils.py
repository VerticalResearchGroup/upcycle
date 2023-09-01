
from . import apps as A

apps : dict[str, A.Trace] = {
    'resnet50-infer-online':  A.mlperf_v1_apps['resnet50'].default_infer_online(),
    'resnet50-infer-offline': A.mlperf_v1_apps['resnet50'].default_infer_offline(),
    'resnet50-train-small':   A.mlperf_v1_apps['resnet50'].default_train_small(),
    'resnet50-train-large':   A.mlperf_v1_apps['resnet50'].default_train_large(),

    'ssdrn34-infer-online':  A.mlperf_v1_apps['ssdrn34-1200'].default_infer_online(),
    'ssdrn34-infer-offline': A.mlperf_v1_apps['ssdrn34-1200'].default_infer_offline(),
    'ssdrn34-train-small':   A.mlperf_v1_apps['ssdrn34-300'].default_train_small(),
    'ssdrn34-train-large':   A.mlperf_v1_apps['ssdrn34-300'].default_train_large(),

    'unet-infer-online':  A.mlperf_v1_apps['unet'].default_infer_online(),
    'unet-infer-offline': A.mlperf_v1_apps['unet'].default_infer_offline(),
    'unet-train-small':   A.mlperf_v1_apps['unet'].default_train_small(),
    'unet-train-large':   A.mlperf_v1_apps['unet'].default_train_large(),

    'bert-infer-online':  A.mlperf_v1_apps['bert-large-squad'].default_infer_online(),
    'bert-infer-offline': A.mlperf_v1_apps['bert-large-squad'].default_infer_offline(),
    'bert-train-small':   A.mlperf_v1_apps['bert-large-pretrain'].default_train_small(),
    'bert-train-large':   A.mlperf_v1_apps['bert-large-pretrain'].default_train_large(),

    'rnnt-infer-online':  A.mlperf_v1_apps['rnnt'].default_infer_online(),
    'rnnt-infer-offline': A.mlperf_v1_apps['rnnt'].default_infer_offline(),
    'rnnt-train-small':   A.mlperf_v1_apps['rnnt'].default_train_small(),
    'rnnt-train-large':   A.mlperf_v1_apps['rnnt'].default_train_large(),
}

infer_online = {appname: app for appname, app in apps.items() if appname.endswith('-infer-online')}
infer_offline = {appname: app for appname, app in apps.items() if appname.endswith('-infer-offline')}
train_small = {appname: app for appname, app in apps.items() if appname.endswith('-train-small')}
train_large = {appname: app for appname, app in apps.items() if appname.endswith('-train-large')}

appnames = [
    'resnet50',
    'ssdrn34',
    'unet',
    'bert',
    'rnnt'
]

prettynames = {
    'resnet50': 'Resnet50',
    'ssdrn34': 'SSD-Resnet34',
    'unet': 'UNet',
    'bert': 'BERT-Large',
    'rnnt': 'RNN-T'
}

shortnames = {
    'resnet50': 'rn50',
    'ssdrn34': 'ssd',
    'unet': 'unet',
    'bert': 'bert',
    'rnnt': 'rnnt'
}

base_membw = 900 * 2**30
