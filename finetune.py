#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import json
import time
import numpy
import random
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler

from utils import misc
from utils.logger import setup_logger
from utils.slconfig import SLConfig
from utils.get_param_dicts import get_param_groups_and_set_lr
from data_sets.coco_ovdino import build_ovdino_dataset
from utils.misc import BestMetricHolder, frozen_parameter
from engine import train_one_epoch, inference_on_dataset
from data_sets.coco_evaluation import COCOEvaluator
from engine import print_csv_format
# from transformers import ViTFeatureExtractor, ViTModel, AutoImageProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# 当前脚本文件
script_path = os.path.abspath(__file__)
# 工程包路径
project_path = script_path.rsplit('/', 1)[0] + '/'
# 设置环境变量
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():

    parser = argparse.ArgumentParser('Set transformer ov-dino', add_help=False)
    parser.add_argument(
        '--config_file', type=str,
        default=project_path + 'configs/model_cfg/ovdino_swinb384_bert_small_ft_24ep.py',
    )
    # dataset parameters
    parser.add_argument(
        "--datasets", type=str, 
        help='path to datasets json'
    )
    parser.add_argument(
        '--output_dir',
        help='path where to save, empty for no saving'
    )
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--template', type=str,
                        # default='full',
                        # default='simple',
                        default='identity',
                        choices=["full", "subset", "simple", "identity"])
    parser.add_argument('--inference_template', type=str,
                        # default='full',
                        # default='simple',
                        default='identity',
                        choices=["full", "subset", "simple", "identity"])

    # training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument(
        '--seed',
        # default=None,
        default=33091922,
        type=int
    )
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        '--pretrain_model_path', type=str,
        help='load from other checkpoint'
    )

    # parser.add_argument(
    #     '--train_num_classes',
    #     type=int,
    #     required = True
    # )

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--print_fre', type=int, default=100)
    parser.add_argument('--save_checkpoint_interval', type=int, default=10)
    parser.add_argument('--onecyclelr', type=bool, default=False)

    # distributed training parameters（警告：使用DDP数据分布式并行训练必须且必需在代码中添加以下参数）
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true', help="Train with mixed precision")
    parser.add_argument('--inference', default=False, help="inference a image")
    parser.add_argument('--train_num_classes', default=6, type=int, required=True, help="inference a image")


    parser.add_argument('--use_yoloxhsvrandomaug', default=False, type=str2bool, help="use_yoloxhsvrandomaug")
    parser.add_argument('--use_randomrotation', default=False, type=str2bool, help="use_randomrotation")
    parser.add_argument('--use_randomcrop', default=False, type=str2bool, help="use_randomcrop")
    parser.add_argument('--use_copy_paste', default=False, type=str2bool, help="use_copy_paste")
    parser.add_argument('--cp_threshold', default=-1, type=int, help="cp_threshold")

    # learning rate related stuff
    # parser.add_argument('--lr_model', type=float, help="lr_model")
    # parser.add_argument('--lr_backbone', type=float, help="lr_backbone")

    args = parser.parse_args()  # 可以添加其它namespace进行融合

    return args


def build_model(args, logger=None):

    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.model_name in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.model_name)
    model = build_func(args, logger=logger)
    return model


def seed_all_rng(seed=None, logger=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    # from datetime import datetime
    # if seed is None:
    #     seed = (os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big"))
    #     if logger is None:
    #         print("Using a generated random seed {}".format(seed))
    #     else:
    #         logger.info("Using a generated random seed {}".format(seed))
    #
    # numpy.random.seed(seed)
    # torch.manual_seed(seed)
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    #
    # return seed

    from datetime import datetime
    if seed is None:
        seed = (os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big"))
        if logger is None:
            print("Using a generated random seed {}".format(seed))
        else:
            logger.info("Using a generated random seed {}".format(seed))

    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    numpy.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set cuDNN to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed (note: this should ideally be set before program starts)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed




def main_fun():

    # 获取参数
    args = get_args_parser()
    if args.output_dir:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # 识别分布式参数
    misc.setup_distributed(args)
    # 日志管理器
    logger = setup_logger(
        output=os.path.join(args.output_dir, 'log.txt'),
        distributed_rank=args.rank,
        color=False,
        name="ov dino"
    )
    logger.info('launching logger ...')
    # 设置随机种子
    # seed = args.seed + misc.get_rank()
    seed = seed_all_rng(seed=args.seed, logger=logger)
    args.seed = seed
    # 加载配置文件参数并整合到args参数中
    logger.info("Loading config file from {}".format(args.config_file))
    config = SLConfig.fromfile(args.config_file)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, args.config_file.split('/')[-1])
        config.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
    cfg_dict = config._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    # 整合评估数据集参数
    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]
    if args.rank == 0:
        logger.info('all args = {}'.format(json.dumps(vars(args), indent=4)))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))

    # 保存所有参数
    if not getattr(args, 'debug', None):
        args.debug = False
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger.info("Full config saved to {}".format(save_json_path))

    # 构建模型
    device = torch.device(args.device)
    logger.info("build model: {} ...".format(args.model_name))
    model = build_model(args, logger=logger)

    model.to(device)
    # 修改model.num_classes参数否则会报错
    # model.num_classes = args.finetune_num_classes # 修改为具体ft数据类别数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("build model done: {},\tparams size = {}M".format(args.model_name, n_parameters / 1000000))
    # 权重冻结
    model = frozen_parameter(model=model, args=args, logger=logger)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("frozen model done,\tlearning params size = {}M".format(n_parameters / 1000000))

    # 分布式训练设置（添加DDP封装）
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params
        )
        model._set_static_graph()
        model_without_ddp = model.module

    # 将模型权重拆分成参数组设置不同的学习率
    param_groups_dicts = get_param_groups_and_set_lr(args=args, model_without_ddp=model_without_ddp)

    # 模型权重冻结（略）

    # 构建优化器和设置衰减策略
    optimizer = torch.optim.AdamW(param_groups_dicts, lr=args.lr_model, weight_decay=args.weight_decay)
    if args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    logger.info('lr_scheduler = {}'.format(lr_scheduler))
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.MultiStepLR):
        logger.info('lr_scheduler = MultiStepLR, \tstep_size: {}, \tlast_epoch: {}'.format(
            lr_scheduler.milestones,
            lr_scheduler.last_epoch)
        )
    elif isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
        logger.info('lr_scheduler = StepLR, \tstep_size: {}'.format(lr_scheduler.step_size))

    # 权重加载（继续训练优先，加载预训练模型其次，完全初始化训练最后）
    # 继续训练权重检测
    output_dir = Path(args.output_dir)  # 为了保证不同系统路径斜杠的兼容性
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        logger.info('load model checkpoint from {}'.format(args.resume))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # 加载模型权重 和 加载断点优化器等状态信息
        _load_state_output = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1

    # 如果没有中断epoch训练权重则加载 pretrain_model_path权重
    if (not args.resume) and args.pretrain_model_path:
        logger.info('load model from args.pretrain_model_path = {}'.format(args.pretrain_model_path))
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')

        # 微调时label_enc根据具体类别数加载权重
        label_enc_weights = checkpoint['model']['label_enc.weight'] # 150x256
        label_enc_cut = label_enc_weights[:args.train_num_classes, :] # 80x256
        checkpoint['model']['label_enc.weight'] = label_enc_cut

        _load_state_output = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        logger.info('_load_state_output = {}'.format(str(_load_state_output)))


    # args.eval = True


    # 构建数据集
    logger.info('data sets = {}'.format(json.dumps(dataset_meta, indent=4)))
    # 训练集
    if not args.eval:
        logger.info("build training dataset ...")
        num_of_dataset_train = len(dataset_meta["train"])
        logger.info('train dataset num = {}'.format(num_of_dataset_train))
        if num_of_dataset_train == 1:
            dataset_train = build_ovdino_dataset(is_train=True, args=args, datasetinfo=dataset_meta["train"][0], logger=logger)
        else:
            # 拼接数据集
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(
                    build_ovdino_dataset(is_train=True, args=args, datasetinfo=dataset_meta["train"][idx], logger=logger)
                )
            dataset_train = ConcatDataset(dataset_train_list)
        logger.info('done, number of training dataset: {}, samples: {}'.format(num_of_dataset_train, len(dataset_train)))
    # 验证集
    dataset_val = build_ovdino_dataset(is_train=False, args=args, datasetinfo=dataset_meta["val"][0], logger=logger)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train, shuffle=True, drop_last=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size=args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset=dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=misc.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=1, sampler=sampler_val, drop_last=False,
                                 collate_fn=misc.collate_fn, num_workers=args.num_workers)

    # 模型评估器
    evaluator = COCOEvaluator(annotation_file=dataset_meta["val"][0]['anno'])
    # 初始化模型评估
    logger.info('start evaluating on validation set ...')
    model.eval()
    result = inference_on_dataset(model=model, data_loader=data_loader_val, evaluator=evaluator, logger=logger)
    current_map = print_csv_format(results=result, logger=logger)
    logger.info('current_map = {}'.format(current_map))

    # return
    #
    # # 开始训练
    logger.info('start training ...')
    # start_time = time.time()
    # best_map_holder = BestMetricHolder(use_ema=False)
    best_map_holder = current_map

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # 训练
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            logger=logger
        )
        # 评估
        model.eval()
        result = inference_on_dataset(model=model, data_loader=data_loader_val, evaluator=evaluator, logger=logger)
        current_map = print_csv_format(results=result, logger=logger)
        logger.info('current_map = {}'.format(current_map))
        # 学习率调整
        if not args.onecyclelr:
            lr_scheduler.step()
        # 模型保存
        if args.output_dir:
            checkpoint_paths = []
            # extra checkpoint before LR drop and every 100 epochs
            # if (epoch + 1) % args.save_checkpoint_interval == 0:
            #     checkpoint_paths.append(args.output_dir + f'checkpoint{epoch:04}.pth')
            # 关闭，因为测试data没有标注用于选择
            if current_map > best_map_holder:
                checkpoint_paths.append(args.output_dir + f'checkpoint_best.pth')
                best_map_holder = current_map
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if 'best' in checkpoint_path:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'args': args,
                    }
                logger.info('saving {}'.format(checkpoint_path))
                misc.save_on_master(weights, checkpoint_path)


if __name__ == "__main__":

    main_fun()

