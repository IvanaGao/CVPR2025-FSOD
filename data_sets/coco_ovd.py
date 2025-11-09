# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import json
import logging
import os
import random

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from data_sets.imagenet_template import template_meta
from utils.misc import clean_words_or_phrase
from data_sets.dataset_mapper import DetrDatasetMapper

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_coco_json"]


def load_coco_json(
    json_file,
    image_root,
    dataset_name=None,
    extra_annotation_keys=None,
    num_sampled_classes=-1,
    template="simple",
    test_mode=True,
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation, and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.
        num_sampled_classes (int): the number of sampled classes.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    id_map = None
    if dataset_name is not None:
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # In COCO, certain category ids are artificially removed, and by convention they are always ignored.
        # We deal with COCO's id issue and translate the category ids to contiguous ids in [0, 80).
        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning("Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.")
        id_map = {v: i for i, v in enumerate(cat_ids)}
        id2name = {
            i: c["name"] for i, c in enumerate(sorted(cats, key=lambda x: x["id"]))
        }

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0
    for img_dict, anno_dict_list in tqdm(imgs_anns, desc="Loading dataset"):
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id
            assert (anno.get("ignore", 0) == 0), '"ignore" in COCO json file is not supported.'
            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            # obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["bbox_mode"] = 'xywh'
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs

        if test_mode and num_sampled_classes > 0:
            obj_cat_ids = [obj["category_id"] for obj in objs]
            sampled_cat_names = [
                clean_words_or_phrase(cat_name) for _, cat_name in id2name.items()
            ]
            sampled_cat_names = [
                    [template.format(cat_name) for template in template_meta[template]]
                    for cat_name in sampled_cat_names
                ]

        # sample category from category_list
        if not test_mode and num_sampled_classes > 0:
            obj_cat_ids = [obj["category_id"] for obj in objs]
            continous_cat_ids = sorted(id_map.values())
            pos_cat_ids = set(obj_cat_ids)
            assert len(pos_cat_ids) <= num_sampled_classes
            neg_cat_ids = random.sample(set(continous_cat_ids) - pos_cat_ids, num_sampled_classes - len(pos_cat_ids),)
            sampled_cat_ids = list(pos_cat_ids) + list(neg_cat_ids)
            sampled_cat_names = [clean_words_or_phrase(id2name[cat_id]) for cat_id in sampled_cat_ids]
            sampled_id_map = {cat_id: i for i, cat_id in enumerate(sampled_cat_ids)}
            for obj in objs:
                cat_id = obj["category_id"]
                obj["category_id"] = sampled_id_map[cat_id]

            sampled_cat_names = [
                random.choice(template_meta[template]).format(cat_name)
                for cat_name in sampled_cat_names
            ]

        record["category_names"] = sampled_cat_names
        dataset_dicts.append(record)

    print(
        f"Loaded {len(dataset_dicts)} data points from {dataset_name}, template: {template}\n"
        + f"Sample: {sampled_cat_names[0]}"
    )

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(num_instances_without_valid_segmentation)
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


from data_sets.coco_ovd import load_coco_json


def build_ovdino_dataset(image_set, args, datasetinfo):
    if datasetinfo["dataset_mode"] == 'coco':
        return load_coco_json(
            json_file=datasetinfo['anno'],
            image_root=datasetinfo['root'],
            dataset_name=datasetinfo['data_name'],
            num_sampled_classes=datasetinfo['class_num'],
            template=args.template,
        )
    if datasetinfo["dataset_mode"] == 'odvg':
        # from .odvg import build_odvg
        # return build_odvg(image_set, args, datasetinfo)
        pass
    raise ValueError(f'dataset {args.dataset_file} not supported')


def build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        *,
        aspect_ratio_grouping=False,
        num_workers=0,
        collate_fn=None,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)

    if aspect_ratio_grouping:
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
        )


def build_ovdino_detection_train_loader(
        dataset,
        *,
        mapper,
        sampler=None,
        total_batch_size,
        aspect_ratio_grouping=True,
        num_workers=0,
        collate_fn=None,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """

    from data_sets.augmentation.augmentation_impl import RandomFlip, RandomCrop
    from data_sets.augmentation.augmentation_impl import ResizeShortestEdge

    mapper = DetrDatasetMapper(
        augmentation=[
            RandomFlip(),
            ResizeShortestEdge(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            RandomFlip(),
            ResizeShortestEdge(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            RandomCrop(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            ResizeShortestEdge(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    )

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"

    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )





#
#
# if __name__ == "__main__":
#     """
#     Test the COCO json dataset loader.
#
#     Usage:
#         python -m detectron2.data.datasets.coco \
#             path/to/json path/to/image_root dataset_name
#
#         "dataset_name" can be "coco_2014_minival_100", or other
#         pre-registered ones
#     """
#     import sys
#
#     import detectron2.data.datasets  # noqa # add pre-defined metadata
#     from detectron2.utils.logger import setup_logger
#     from detectron2.utils.visualizer import Visualizer
#
#     logger = setup_logger(name=__name__)
#     assert sys.argv[3] in DatasetCatalog.list()
#     meta = MetadataCatalog.get(sys.arvg[3])
#
#     dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
#     logger.info("Done loading {} samples.".format(len(dicts)))
#
#     dirname = "coco-data-vis"
#     os.makedirs(dirname, exist_ok=True)
#     for d in dicts:
#         img = np.array(Image.open(d["file_name"]))
#         visualizer = Visualizer(img, metadata=meta)
#         vis = visualizer.draw_dataset_dict(d)
#         fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
#         vis.save(fpath)
