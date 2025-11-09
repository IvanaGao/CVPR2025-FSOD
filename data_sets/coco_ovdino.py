# Copyright (c) Facebook, Inc. and its affiliates.
import os
import io
import sys
import copy
import time
import numpy
import torch
import random
import contextlib
from typing import Callable, Optional
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from data_sets.imagenet_template import template_meta
from utils.misc import clean_words_or_phrase
from data_sets.detectron_transforms.augmentation_impl import RandomFlip, ResizeShortestEdge, RandomCrop, RandomRotation, YOLOXHSVRandomAug
from data_sets.detectron_transforms.augmentation import apply_augmentations
from data_sets.detectron_transforms.detection_utils import transform_instance_annotations, annotations_to_instances
from data_sets.detectron_transforms.detection_utils import filter_empty_instances


# def fill_background(instances, h, w, image):
#     for box in instances.gt_boxes:
#         box = box.tolist()
#
#         print()


def create_transforms(is_train, args=None):
    if is_train:
        augmentation = [
            RandomFlip(),
            ResizeShortestEdge(
                short_edge_length=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                max_size=1333,
                sample_style="choice",
            ),
        ]

        augmentation_with_crop = [
            RandomFlip(),
            ResizeShortestEdge(
                short_edge_length=[400, 500, 600],
                sample_style="choice",
            ),
        ]

        # if args.use_randomrotation:
        #     augmentation_with_crop.append(RandomRotation(
        #             angle=[-45, 45],
        #         ))

        if args.use_yoloxhsvrandomaug:
            augmentation_with_crop.append(YOLOXHSVRandomAug())

        if args.use_randomcrop:
            augmentation_with_crop.append(RandomCrop(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ))

        augmentation_with_crop.append(ResizeShortestEdge(
                    short_edge_length=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                    max_size=1333,
                    sample_style="choice",
                ))

    else:
        augmentation = [
            ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
        ]
        augmentation_with_crop = None

    return augmentation, augmentation_with_crop


class OVDinoDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            anno: str,
            data_type: str = None,
            max_labels: int = 80,
            is_train=True,
            template='full',
            augmentation1: Optional[Callable] = None,
            augmentation2: Optional[Callable] = None,
            logger=None,
            rm_train_null_sample=False,
            args=None,
    ) -> None:
        super().__init__(root)
        assert data_type is not None
        self.root = root
        self.is_train = is_train
        self.data_type = data_type
        self.max_labels = max_labels
        self.rm_train_null_sample = rm_train_null_sample
        self.obj_count = 0
        self.bad_obj_count = 0
        # self.neg_prefixes = ['a', 'an', 'the']  # 用于当负类数量不足时补充负类名称
        if is_train:
            self.template = template
            logger.info('train data use template = {}'.format(self.template))
        else:
            self.template = 'identity'  # 警告评估的模版参数由model.inference_template决定，但该参数影响类别名称
            logger.info('val data use template = {}'.format(self.template))
        self._load_metas(anno_file=anno, logger=logger)
        # 构建数据增强器
        if augmentation1 is None and augmentation2 is None:
            logger.info('augmentation1 and augmentation2 is None, run create_transforms()')
            self.augmentation1, self.augmentation2 = create_transforms(is_train=is_train, args=args)
        else:
            logger.warning('use custom augmentation1 or augmentation2, may not compatible!')
        self.sample_count = len(self.metas)
        self.get_dataset_info()

        self.args = args

    def _valid(self, anns=None):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    def _object_valid(self, obj=None, img_w=None, img_h=None):
        # 删除坐标极端越界，线式目标的bbox
        if obj['bbox'][0] >= img_w or obj['bbox'][1] >= img_h or obj['bbox'][2] <= 3 or obj['bbox'][3] <= 3:
            # print('illegal object, img(w, h) = ({}, {}) bbox = {}'.format(img_w, img_h, obj['bbox']))
            return False
        else:
            if obj['bbox'][0] + obj['bbox'][2] <= 0 or obj['bbox'][1] + obj['bbox'][3] <= 0:
                return False
            else:
                return True

    def _load_det_mates(self, anno_file=None, logger=None):

        # 屏蔽coco加载过程中的日志输出
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(anno_file)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # coco类别索引从1开始
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            print("warning: Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.")
        # 将类别索引连续化
        self.id_map = {v: i for i, v in enumerate(cat_ids)}
        self.id2name = {i: c["name"] for i, c in enumerate(sorted(cats, key=lambda x: x["id"]))}

        image_ids = sorted(coco_api.imgs.keys())
        images = coco_api.loadImgs(image_ids)
        # 标注和图像匹配有效性检查
        annotations = [coco_api.imgToAnns[img_id] for img_id in image_ids]
        total_num_valid_anns = sum([len(x) for x in annotations])
        total_num_anns = len(coco_api.anns)
        if total_num_valid_anns < total_num_anns:
            print(f"{anno_file} contains {total_num_anns} annotations, but only "
                  f"{total_num_valid_anns} of them match to images in the file.")
        # 将coco标注协议转换为detectron标注协议
        images_annotations = list(zip(images, annotations))
        dataset_dicts_list = []
        ann_keys = ["iscrowd", "bbox", "category_id"]
        if logger is not None:
            logger.info("Loading {} images in COCO format from: {}".format(len(images), anno_file))
        else:
            print("Loading {} images in COCO format from {}".format(len(images), anno_file))
        for img_dict, anno_dict_list in tqdm(images_annotations, desc="Loading dataset: "):
            record = {
                "file_name": os.path.join(self.root, img_dict["file_name"]),
                "height": img_dict["height"],
                "width": img_dict["width"]
            }
            image_id = record["image_id"] = img_dict["id"]
            objs = []
            self.obj_count += len(anno_dict_list)
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert (anno.get("ignore", 0) == 0), '"ignore" in COCO json file is not supported.'
                obj = {key: anno[key] for key in ann_keys if key in anno}
                if "bbox" in obj and len(obj["bbox"]) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'bbox' value! "
                        f"This json does not have valid COCO format."
                    )

                obj["bbox_mode"] = 'xywh'
                # 将原类别索引映射到连续类别索引中
                if self.id_map:
                    annotation_category_id = obj["category_id"]
                    try:
                        obj["category_id"] = self.id_map[annotation_category_id]
                    except KeyError as e:
                        raise KeyError(
                            f"Encountered category_id={annotation_category_id} "
                            "but this id does not exist in 'categories' of the json file.") from e

                # 只清洗训练集(防止影响测试集)
                if self.is_train:
                    if self._object_valid(obj=obj, img_w=img_dict["width"], img_h=img_dict["height"]):
                        objs.append(obj)
                    else:
                        self.bad_obj_count += 1
                else:
                    objs.append(obj)

            record["annotations"] = objs

            # 训练数据集（添加负类，并重映射类别索引，并随机添加 prompt template，为训练做准备）
            # 警告：ov-dino中的类别增强是固定的，grounding dino中是训练过程中动态采样的
            if self.is_train and self.max_labels > 0:
                continous_cate_ids = set(sorted(self.id_map.values()))
                # 获取当前图像中所有bbox类别
                obj_cate_ids = [obj["category_id"] for obj in objs]
                pos_cate_ids = set(obj_cate_ids)
                assert len(pos_cate_ids) <= self.max_labels
                # 负类
                neg_cate_ids = continous_cate_ids - pos_cate_ids
                add_neg_num = min(len(neg_cate_ids), self.max_labels - len(pos_cate_ids))
                sampled_cate_ids = list(pos_cate_ids)
                if add_neg_num > 0:
                    sampled_cate_ids = sampled_cate_ids + list(random.sample(neg_cate_ids, add_neg_num))
                sampled_cate_names = [clean_words_or_phrase(self.id2name[cat_id]) for cat_id in sampled_cate_ids]

                # if '/coco' in self.root:
                #     # 只针对coco数据集加入预训练类别过少无法满足最大类别数要求，另行添加负类别名称
                #     neg_cls_gap_num = self.max_labels - len(sampled_cate_names)
                #     if neg_cls_gap_num > 0:
                #         neg_cls_gap = []
                #         for pref in self.neg_prefixes:
                #             for neg_cat_id in neg_cate_ids:
                #                 neg_cls_gap.append(pref + ' ' + clean_words_or_phrase(self.id2name[neg_cat_id]))
                #                 if len(neg_cls_gap) >= neg_cls_gap_num:
                #                     break
                #             if len(neg_cls_gap) >= neg_cls_gap_num:
                #                 break
                #         sampled_cate_names = sampled_cate_names + neg_cls_gap

                # 当前样本类别索引再次重映射（也就是对于一张image需要对其caption的类别分配连续索引）
                sampled_id_map = {cat_id: i for i, cat_id in enumerate(sampled_cate_ids)}
                for obj in objs:
                    cat_id = obj["category_id"]
                    obj["category_id"] = sampled_id_map[cat_id]
                # 每一个类别随机抽取一个template
                sampled_cate_names = [
                    random.choice(template_meta[self.template]).format(cat_name) for cat_name in sampled_cate_names
                ]

            # 测试数据集
            if not self.is_train and self.max_labels > 0:
                sampled_cate_names = [
                    clean_words_or_phrase(cat_name) for _, cat_name in self.id2name.items()
                ]
                # 测试数据集是每个类别对所有的template都构建一次
                sampled_cate_names = [
                    [template.format(cat_name) for template in template_meta[self.template]]
                    for cat_name in sampled_cate_names
                ]

            record["category_names"] = sampled_cate_names
            dataset_dicts_list.append(record)

        return dataset_dicts_list

    def _load_rec_mates(self, anno_file=None, logger=None):

        if not self.is_train:
            raise ValueError('rec dataset not support as validate, only train')

        # 屏蔽coco加载过程中的日志输出
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(anno_file)
            # 数据类型校验
            if 'caption' not in coco_api.dataset['images'][0].keys():
                raise ValueError('{} data_type = "rec", but annotation has no "caption" information, check please!')

        image_ids = sorted(coco_api.imgs.keys())
        images = coco_api.loadImgs(image_ids)
        # 构建当前数据集的名词短语库，用于负类采集
        # 警告！！！gqa和其他rec数据的 tokens_positive_eval 标注格式不一致，需要特殊处理
        category_hub = []
        for image in images:
            for indexes in image['tokens_positive_eval']:

                if '/gqa/' in self.root:
                    temp_str = ''
                    for tp_item in indexes:
                        temp_str += image['caption'][tp_item[0]: tp_item[-1]] + ' '
                    category_hub.append(temp_str[:-1])
                    # if temp_str[:-1] == 'Who' or temp_str[:-1] == 'What':
                    #     # 获取相应的标注
                    #     t = coco_api.imgToAnns[image['id']]
                    #     print()
                else:
                    for tp_item in indexes:
                        category_hub.append(image['caption'][tp_item[0]: tp_item[1]])

        from collections import Counter
        category_hub_ = Counter(category_hub)

        category_hub = list(set(category_hub))
        seg_count = len(category_hub) // 1000 - 1   # 采用分块采样防止短语库过大导致处理效率低（-1是为了避免溢出）

        # 标注和图像匹配有效性检查
        annotations = [coco_api.imgToAnns[img_id] for img_id in image_ids]
        total_num_valid_anns = sum([len(x) for x in annotations])
        total_num_anns = len(coco_api.anns)
        if total_num_valid_anns < total_num_anns:
            print(f"{anno_file} contains {total_num_anns} annotations, but only "
                  f"{total_num_valid_anns} of them match to images in the file.")
        # 将coco标注协议转换为detectron标注协议
        ann_keys = ["iscrowd", "bbox", "category_id"]
        if logger is not None:
            logger.info("Loading {} images in COCO format from: {}".format(len(images), anno_file))
        else:
            print("Loading {} images in COCO format from {}".format(len(images), anno_file))
        dataset_dicts_list = []
        images_annotations = list(zip(images, annotations))
        for img_dict, anno_dict_list in tqdm(images_annotations, desc="Loading dataset: "):
            record = {
                "file_name": os.path.join(self.root, img_dict["file_name"]),
                "height": img_dict["height"],
                "width": img_dict["width"]
            }
            image_id = record["image_id"] = img_dict["id"]
            objs = []
            self.obj_count += len(anno_dict_list)
            for anno in anno_dict_list:

                # 忽略没有标签的bbox
                if len(anno['tokens_positive']) == 0:
                    continue

                assert anno["image_id"] == image_id
                assert (anno.get("ignore", 0) == 0), '"ignore" in COCO json file is not supported.'
                obj = {key: anno[key] for key in ann_keys if key in anno}
                if "bbox" in obj and len(obj["bbox"]) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'bbox' value! "
                        f"This json does not have valid COCO format."
                    )

                obj["bbox_mode"] = 'xywh'

                # 警告！！！gqa和其他rec数据的 tokens_positive_eval 标注格式不一致，需要特殊处理
                if '/gqa/' in self.root:
                    temp_np = ''
                    for tp_item in anno['tokens_positive']:
                        temp_np += img_dict['caption'][tp_item[0]:tp_item[-1]] + ' '
                    obj["noun_phrase"] = temp_np[:-1]
                else:
                    # 同一个目标框多个描述短语
                    tokens_positive_count = len(anno['tokens_positive'])
                    tp_index = random.randint(0, tokens_positive_count - 1)
                    tp_item = anno['tokens_positive'][tp_index]
                    obj["noun_phrase"] = img_dict['caption'][tp_item[0]:tp_item[-1]]

                # 只清洗训练集(防止影响测试集)
                if self.is_train:
                    if self._object_valid(obj=obj, img_w=img_dict["width"], img_h=img_dict["height"]):
                        objs.append(obj)
                    else:
                        self.bad_obj_count += 1
                else:
                    objs.append(obj)

            record["annotations"] = objs

            # 构建当前样本的短语索引
            # 1、获取当前图像中所有bbox类别
            obj_noun_phrases = [obj['noun_phrase'] for obj in objs]
            pos_noun_phrases = list(set(obj_noun_phrases))
            pos_noun_phrases.sort()
            noun_phrases_id_map = {noun_phrase: i for i, noun_phrase in enumerate(pos_noun_phrases)}
            # 设置标注的短语索引
            for obj in objs:
                obj['category_id'] = noun_phrases_id_map[obj['noun_phrase']]

            # 2、训练数据集采样负类（重映射类别索引，并随机添加 prompt template，为训练做准备）
            # 警告：ov-dino中的类别增强是固定的，grounding dino中是训练过程中动态采样的
            neg_noun_phrases = []
            if self.is_train and self.max_labels > 0:
                assert len(pos_noun_phrases) <= self.max_labels
                # 负类（采用分块随机采样是为了节约时间，否则短语库过大集合采样效率低）
                seg_idx = random.randint(0, seg_count)
                category_hub_sub = set(category_hub[1000 * seg_idx: 1000 * (seg_idx + 1)])
                neg_noun_phrases_ = category_hub_sub - set(pos_noun_phrases)  # 耗时
                add_neg_num = min(len(neg_noun_phrases_), self.max_labels - len(pos_noun_phrases))
                if add_neg_num > 0:
                    neg_noun_phrases = list(random.sample(neg_noun_phrases_, add_neg_num))  # 耗时

            sampled_noun_phrases = pos_noun_phrases + neg_noun_phrases
            sampled_noun_phrases = [clean_words_or_phrase(noun_phrase) for noun_phrase in sampled_noun_phrases]
            # 3、每一个类别随机抽取一个template
            sampled_cate_names = [
                random.choice(template_meta[self.template]).format(cat_name) for cat_name in sampled_noun_phrases
            ]

            record["category_names"] = sampled_cate_names
            dataset_dicts_list.append(record)

        return dataset_dicts_list

    def _load_metas(self, anno_file=None, logger=None):

        if self.data_type == 'det':
            dataset_dicts_list = self._load_det_mates(anno_file=anno_file, logger=logger)
        elif self.data_type == 'rec':
            dataset_dicts_list = self._load_rec_mates(anno_file=anno_file, logger=logger)
        else:
            raise ValueError("self.data_type = {} not supported, must be 'det' or 'rec'".format(self.data_type))

        # 训练集剔除空样本
        if self.is_train:
            all_num = len(dataset_dicts_list)
            if self.rm_train_null_sample:
                dataset_dicts_list = [
                    sample for sample in dataset_dicts_list if (len(sample['annotations']) > 0) and (self._valid(sample['annotations']))
                ]
                valid_num = len(dataset_dicts_list)
                logger.info('all sample = {}, remove {} null annotation sample, left {} sample.'.format(
                    all_num, all_num - valid_num, valid_num
                ))
            else:
                logger.info('self.rm_null_anno = {}, use all sample = {}'.format(self.rm_null_anno, all_num))
        self.metas = dataset_dicts_list

    def get_dataset_info(self):
        print(f"  == total images: {len(self.metas)}")
        if self.data_type == "det":
            print(f"  == total labels: {len(self.id_map)}")

    def __get_one_item(self, index: int):

        # 因为后面有 meta_dict["image"] =  需要添加deepcopy操作，防止 "Too many open files" 问题
        meta_dict = copy.deepcopy(self.metas[index])
        abs_path = meta_dict["file_name"]
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        try:
            image = Image.open(abs_path).convert('RGB')
            # print('good image: {}'.format(abs_path))
            # with open(abs_path, 'rb') as file:
            #     image = Image.open(file).convert('RGB')
        except Exception:
            print('bad image: {}'.format(abs_path))
            sys.exit(2)

        w, h = image.size
        # 核对读取图像大小和标注图像大小的一致性
        if w != meta_dict['width'] or h != meta_dict['height']:
            print('{} img.w, h: {}, {} != ann.w h {}, {}'.format(abs_path, w, h, meta_dict['width'], meta_dict['height']))
            return None
        image = numpy.asarray(image)
        # 数据增强
        if self.is_train:
            if numpy.random.rand() > 0.5:
                image, augmentation = apply_augmentations(self.augmentation1, image)
            else:
                image, augmentation = apply_augmentations(self.augmentation2, image)
        else:
            image, augmentation = apply_augmentations(self.augmentation1, image)

        meta_dict["image"] = torch.as_tensor(numpy.ascontiguousarray(image.transpose(2, 0, 1))) # chw

        # 根据增强器调整标注信息
        image_shape = image.shape[:2]  # h, w
        img_h, img_w = image.shape[:2]
        annos = [
            transform_instance_annotations(obj, augmentation, image_shape)
            for obj in meta_dict.pop("annotations")  # 删除'annotations'字段
            if obj.get("iscrowd", 0) == 0
        ]
        instances = annotations_to_instances(annos, image_shape)
        meta_dict["instances"] = filter_empty_instances(instances)

        return meta_dict

    def __getitem__(self, index: int):

        # 获取三次
        for _ in range(3):
            meta_dict = self.__get_one_item(index=index)
            if meta_dict is None:
                index = random.randint(0, self.sample_count)
            else:
                break

        if meta_dict is None:
            raise RuntimeError('3 times img.w h != ann.w h')

        return meta_dict

    def __len__(self) -> int:
        return len(self.metas)


def build_ovdino_dataset(is_train, args, datasetinfo, logger=None):

    if datasetinfo["anno_protocol"] == 'coco':
        return OVDinoDataset(
            root=datasetinfo['root'],
            anno=datasetinfo['anno'],
            data_type=datasetinfo['data_type'],
            is_train=is_train,
            template=args.template,
            max_labels=args.train_num_classes,
            # max_labels=args.max_class_num,
            logger=logger,
            rm_train_null_sample=args.rm_train_null_sample,
            args=args,
        )

    raise ValueError('dataset {} anno_protocol = {} not supported'.format(args.dataset_file, datasetinfo["anno_protocol"]))



