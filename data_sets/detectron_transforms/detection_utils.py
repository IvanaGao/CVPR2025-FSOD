import torch
import math
import numpy as np
from data_sets.detectron_transforms.transform import TransformList
from data_sets.detectron_transforms.instances import Instances
from data_sets.detectron_transforms.boxes import Boxes, BoxMode


# class BoxMode(object):
#     """
#     Enum of different ways to represent a box.
#     """
#
#     XYXY_ABS = 0
#     """
#     (x0, y0, x1, y1) in absolute floating points coordinates.
#     The coordinates in range [0, width or height].
#     """
#     XYWH_ABS = 1
#     """
#     (x0, y0, w, h) in absolute floating points coordinates.
#     """
#     XYXY_REL = 2
#     """
#     Not yet supported!
#     (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
#     """
#     XYWH_REL = 3
#     """
#     Not yet supported!
#     (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
#     """
#     XYWHA_ABS = 4
#     """
#     (xc, yc, w, h, a) in absolute floating points coordinates.
#     (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
#     """
#
#     @staticmethod
#     def convert(box, from_mode, to_mode):
#         """
#         Args:
#             box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
#             from_mode, to_mode (BoxMode)
#
#         Returns:
#             The converted box of the same type.
#         """
#         if from_mode == to_mode:
#             return box
#
#         original_type = type(box)
#         is_numpy = isinstance(box, np.ndarray)
#         single_box = isinstance(box, (list, tuple))
#         if single_box:
#             assert len(box) == 4 or len(box) == 5, (
#                 "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor, where k == 4 or 5"
#             )
#             arr = torch.tensor(box)[None, :]
#         else:
#             # avoid modifying the input box
#             if is_numpy:
#                 arr = torch.from_numpy(np.asarray(box)).clone()
#             else:
#                 arr = box.clone()
#
#         assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
#             BoxMode.XYXY_REL,
#             BoxMode.XYWH_REL,
#         ], "Relative mode not yet supported!"
#
#         if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
#             assert (arr.shape[-1] == 5), "The last dimension of input shape must be 5 for XYWHA format"
#             original_dtype = arr.dtype
#             arr = arr.double()
#
#             w = arr[:, 2]
#             h = arr[:, 3]
#             a = arr[:, 4]
#             c = torch.abs(torch.cos(a * math.pi / 180.0))
#             s = torch.abs(torch.sin(a * math.pi / 180.0))
#             # This basically computes the horizontal bounding rectangle of the rotated box
#             new_w = c * w + s * h
#             new_h = c * h + s * w
#
#             # convert center to top-left corner
#             arr[:, 0] -= new_w / 2.0
#             arr[:, 1] -= new_h / 2.0
#             # bottom-right corner
#             arr[:, 2] = arr[:, 0] + new_w
#             arr[:, 3] = arr[:, 1] + new_h
#
#             arr = arr[:, :4].to(dtype=original_dtype)
#         elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
#             original_dtype = arr.dtype
#             arr = arr.double()
#             arr[:, 0] += arr[:, 2] / 2.0
#             arr[:, 1] += arr[:, 3] / 2.0
#             angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
#             arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
#         else:
#             if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
#                 arr[:, 2] += arr[:, 0]
#                 arr[:, 3] += arr[:, 1]
#             elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
#                 arr[:, 2] -= arr[:, 0]
#                 arr[:, 3] -= arr[:, 1]
#             else:
#                 raise NotImplementedError(
#                     "Conversion from BoxMode {} to {} is not supported yet".format(from_mode, to_mode)
#                 )
#
#         if single_box:
#             return original_type(arr.flatten().tolist())
#         if is_numpy:
#             return arr.numpy()
#         else:
#             return arr


# class BoxMode(object):
#     # 支持 xyxy2xywh xywh2xyxy
#     @staticmethod
#     def convert(box, from_mode=None, to_mode=None):
#         """
#         Args:
#             box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
#             from_mode, to_mode (BoxMode)
#
#         Returns:
#             The converted box of the same type.
#         """
#         assert from_mode is not None and to_mode is not None
#         assert from_mode in ['xyxy', 'xywh'] and to_mode in ['xyxy', 'xywh'], 'only support xyxy and xywh'
#
#         if from_mode == to_mode:
#             return box
#
#         original_type = type(box)
#         is_numpy = isinstance(box, np.ndarray)
#         single_box = isinstance(box, (list, tuple))
#         if single_box:
#             assert len(box) == 4 or len(box) == 5, (
#                 "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor, where k == 4 or 5"
#             )
#             arr = torch.tensor(box)[None, :]
#         else:
#             # avoid modifying the input box
#             if is_numpy:
#                 arr = torch.from_numpy(np.asarray(box)).clone()
#             else:
#                 arr = box.clone()
#
#         if from_mode == 'xywh' and to_mode == 'xyxy':
#             arr[:, 2] += arr[:, 0]
#             arr[:, 3] += arr[:, 1]
#         elif from_mode == 'xyxy' and to_mode == 'xywh':
#             arr[:, 2] -= arr[:, 0]
#             arr[:, 3] -= arr[:, 1]
#         else:
#             raise NotImplementedError("Conversion from BoxMode {} to {} is not supported yet".format(from_mode, to_mode))
#
#         if single_box:
#             return original_type(arr.flatten().tolist())
#         if is_numpy:
#             return arr.numpy()
#         else:
#             return arr



def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], 'xyxy')
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = 'xyxy'

    if "segmentation" in annotation:
        raise ValueError('not support segmentation augment')

    if "keypoints" in annotation:
        raise ValueError('not support keypoints augment')

    return annotation


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack([BoxMode.convert(obj["bbox"], obj["bbox_mode"], 'xyxy') for obj in annos])
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos):
        if "segmentation" in annos[0]:
            raise ValueError('not support segmentation augment, you can remove segmentation out')

        if "keypoints" in annos[0]:
            raise ValueError('not support keypoints augment, you can remove keypoints out')

    return target


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return instances[m], m

    return instances[m]

