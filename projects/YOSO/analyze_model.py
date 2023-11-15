# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py

import logging
import time

import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table  # can also try flop_count_str
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch import nn
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.utils.analysis import (
#     activation_count_operators,
#     parameter_count_table,
#     find_unused_parameters
# )
from detectron2.utils.logger import setup_logger

# fmt: off
import os
import sys
import csv

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from yoso import add_yoso_config

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("detectron2")


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_yoso_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg


def do_flop(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()
    #model.half()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        if args.use_fixed_input_size and isinstance(cfg, CfgNode):
            crop_size = cfg.INPUT.CROP.SIZE[0]
            data[0]["image"] = torch.zeros((3, crop_size, crop_size * 2))

        if cfg.DATASETS.TRAIN[0].split("_")[0] == "ade20k":
                data[0]['image'] = F.interpolate(data[0]['image'].unsqueeze(0), size=[640, 2560]).squeeze(0)

        logger.info(f"Image size: {data[0]['image'].shape}")
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )

def do_fps(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    data = next(iter(data_loader))
    inputs = data
    image = inputs[0]['image']

    if cfg.DATASETS.TRAIN[0].split("_")[0] == "ade20k":
        inputs[0]['image'] = F.interpolate(inputs[0]['image'].unsqueeze(0), size=[640, 2560]).squeeze(0)
    logger.info(f"Image size: {inputs[0]['image'].shape}")

    with open(f'results_fps_{args.csv_dataset_name}.csv','a') as f:
        writer = csv.writer(f)
        try:
            iterations = None
            with torch.no_grad():
                for _ in range(10):
                    model(inputs)

                if iterations is None:
                    elapsed_time = 0
                    iterations = 100
                    while elapsed_time < 1:
                        torch.cuda.synchronize()
                        torch.cuda.synchronize()
                        t_start = time.time()
                        for _ in range(iterations):
                            model(inputs)
                        torch.cuda.synchronize()
                        torch.cuda.synchronize()
                        elapsed_time = time.time() - t_start
                        iterations *= 2
                    FPS = iterations / elapsed_time
                    iterations = int(FPS * 6)

                iterations = 200

                logger.info('=========Speed Testing=========')
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(inputs)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                latency = elapsed_time / iterations * 1000
            torch.cuda.empty_cache()
            FPS = 1000 / latency
            logger.info(f"FPS: {FPS}")
            logger.info(f"latency: {latency}")

            writer.writerow([FPS, latency])
        except:
            writer.writerow(["Failed", "Failed"])



def do_activation(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )

def do_parameter(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Model Structure:\n" + str(model))


def do_unused_parameter(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_train_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    unused_params = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        unused_params = find_unused_parameters(model, data)

    logger.info("Unused parameters:\n" + str(unused_params))


def do_torch_profiling(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    if not os.path.exists("profiler_output"):
        os.makedirs("profiler_output")

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        # print(output)
        print(f"{p.step_num}")
        p.export_chrome_trace(f"profiler_output/trace_{args.profiling_name}_" + str(p.step_num) + ".json")

    device = torch.device('cuda')
    image = torch.randn(3, 1024, 2048).to(device)
    inputs = [{'image': image}]
    with torch.no_grad():
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    skip_first=10,
                    wait=5,
                    warmup=1,
                    active=1
                ),
                on_trace_ready=trace_handler,
                record_shapes=True,
        ) as p:
            for idx in range(42):
                model(inputs)
                p.step()


if __name__ == "__main__":
    parser = default_argument_parser(
        epilog="""
Examples:
To show parameters of a model:
$ ./analyze_model.py --tasks parameter \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
Flops and activations are data-dependent, therefore inputs and model weights
are needed to count them:
$ ./analyze_model.py --num-inputs 100 --tasks flop \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \\
    MODEL.WEIGHTS /path/to/model.pkl
"""
    )
    parser.add_argument(
        "--tasks",
        choices=["flop", "fps", "activation", "parameter", "structure", "unused", "profile"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num-inputs",
        default=100,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
             "both are data dependent.",
    )
    parser.add_argument(
        "--use-fixed-input-size",
        action="store_true",
        help="use fixed input size when calculating flops",
    )
    parser.add_argument(
        "--profiling-name",
        help="name for profiling file",
    )
    parser.add_argument(
        "--csv-dataset-name",
        help="name for profiling file",
    )
    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1

    cfg = setup(args)

    for task in args.tasks:
        {
            "flop": do_flop,
            "fps": do_fps,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
            "unused": do_unused_parameter,
            "profile": do_torch_profiling,
        }[task](cfg)
