import os
import sys
import torch
from demo.predictors import OVDINODemo
import json
from tqdm import tqdm
from nms import perform_nms
from detrex.data.datasets import clean_words_or_phrase
from detectron2.data.detection_utils import read_image
# from vis import vis
import argparse
import random
import numpy
from utils import misc
from utils.logger import setup_logger
from utils.slconfig import SLConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# 当前脚本文件
script_path = os.path.abspath(__file__)
# 工程包路径
project_path = script_path.rsplit('/', 1)[0] + '/'
# 设置环境变量
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

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

def get_args_parser():

    parser = argparse.ArgumentParser('Set transformer ov-dino', add_help=False)
    parser.add_argument(
        '--config_file', type=str,
        default=project_path + 'configs/model_cfg/ovdino_swinb384_bert_small_ft_24ep.py', # coco: 53.8
    )
    # dataset parameters
    parser.add_argument(
        "--datasets", type=str,
        default=None,
        help='path to datasets json'
    )
    parser.add_argument(
        '--output_dir',
        default='./',
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
        default=None,
        help='load from other checkpoint'
    )
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

    parser.add_argument('--original_data_config', type=str, required=True, help="inference a image")
    parser.add_argument('--out_dir', type=str, required=True, help="inference a image")
    parser.add_argument('--model_path', type=str, required=True, help="inference a image")
    parser.add_argument('--train_num_classes', default=6, type=int, required=True, help="inference a image")

    args = parser.parse_args()  # 可以添加其它namespace进行融合

    return args


def build_model(args, logger=None):

    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.model_name in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.model_name)
    model = build_func(args, logger=logger)
    return model

def relabel(args, logger, device, dataset_meta, box_threshold, nms_threshold, out_dir, model_path):
    # current_model_weight_path = args.output_dir + "checkpoint_best.pth"
    # current_model_weight_path = args.output_dir + "checkpoint.pth"

    model_for_inference = build_model(args, logger=logger)
    model_for_inference.to(device)

    checkpoint_c = torch.load(model_path, map_location='cpu')

    label_enc_weights = checkpoint_c['model']['label_enc.weight']  # 150x256
    label_enc_cut = label_enc_weights[:args.train_num_classes, :]  # 80x256
    checkpoint_c['model']['label_enc.weight'] = label_enc_cut

    _load_state_output = model_for_inference.load_state_dict(checkpoint_c['model'], strict=False)
    # logger.info('_load_state_output = {}'.format(str(_load_state_output)))

    model_for_inference.eval()

    demo = OVDINODemo(
        model=model_for_inference,
        sam_predictor=None,
        min_size_test=800,
        max_size_test=1333,
        img_format="RGB",
        metadata_dataset="coco_2017_val",
    )

    train_data_meta = dataset_meta['train'][0]
    with open(train_data_meta['anno'], "r") as f:
        train_anns_ori = json.load(f)
        images, annotations, categories = train_anns_ori['images'], train_anns_ori['annotations'], train_anns_ori[
            'categories']

    ANN_ID = 0
    annotations_new = []

    categories_dict = {}
    for cat in categories:
        if cat['id'] == 0: # roboflow数据中id=0忽略
            continue
        categories_dict[clean_words_or_phrase(cat['name'])] = cat['id']
    category_names = list(categories_dict.keys())

    for img_info in tqdm(images):
        img = read_image(train_data_meta['root'] + img_info['file_name'], format="BGR")

        image_id = img_info['id']
        annotations_to_process = [anno for anno in annotations if anno['image_id'] == image_id]
        for anno in annotations_to_process:
            anno['score'] = 1.1  # to make sure it wont be filtered out by nms
            # anno['label'] = category_names[anno['category_id'] - 1]

        predictions, visualized_output = demo.run_on_image(
            img, category_names, box_threshold
        )

        preds = predictions["instances"]  # .to(self.cpu_device)
        boxes = preds.pred_boxes  # xyx'y'
        scores = preds.scores
        classes = (
            preds.pred_classes.tolist()
        )

        for box, score, label in zip(boxes, scores, classes):
            box = box.float()
            x, y, w, h = float(box[0]), float(box[1]), float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
            category_name = category_names[label]
            category_id = categories_dict[category_name]

            annotations_to_process.append({
                "image_id": img_info['id'],
                "bbox": [x, y, w, h],
                "area": w * h,
                "category_id": category_id, # 1~n
                "score": score.item(),
                "iscrowd": 0,
                # "label": category_names[category_id - 1]
                "label": category_name
            })
        # annotations_to_process = [anno for anno in annotations_to_process if anno['score'] > box_threshold]
        annotations_to_process = perform_nms(annotations_to_process, nms_threshold)

        for ann in annotations_to_process:
            ann['label'] = category_names[ann['category_id'] - 1]

        # vis(img, annotations_to_process)

        for anno in annotations_to_process:
            anno['id'] = ANN_ID
            annotations_new.append(anno)
            ANN_ID += 1

    train_anns_ori['annotations'] = annotations_new
    with open(out_dir + "/instances_train.json", "w") as f:
        json.dump(train_anns_ori, f)

    print("finished writing")
    return

if __name__ == "__main__":
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
    device = torch.device(args.device)

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


    box_threshold = 0.18
    nms_threshold = 0.5

    with open(args.original_data_config, "r") as f:
        dataset_meta = json.load(f)

    relabel(args, logger, device, dataset_meta, box_threshold, nms_threshold, args.out_dir, args.model_path)