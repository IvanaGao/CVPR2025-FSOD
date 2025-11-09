import torch
import torchvision.ops as ops

# per-class nms
# def perform_nms(result, iou_threshold=1.0):
#     IMAGE_ID = result[0]['image_id']
#
#     nms_result = []
#     class_preds = {}
#     for res in result:
#         cat_id = res['category_id']
#         if cat_id not in class_preds:
#             class_preds[cat_id] = {'score': [], 'bbox': []}
#         class_preds[cat_id]['score'].append(res['score'])
#         class_preds[cat_id]['bbox'].append(res['bbox'])
#     for cat_id, preds in class_preds.items():
#         class_boxes = torch.tensor(preds['bbox'])  # xywh
#         class_logits = torch.tensor(preds['score'])
#         class_boxes[:, 2:] = class_boxes[:, :2] + class_boxes[:, 2:]
#         nms_keep_indices = ops.nms(class_boxes.float(), class_logits, iou_threshold)
#         nms_keep_boxes = class_boxes[nms_keep_indices].tolist()
#         nms_keep_logits = class_logits[nms_keep_indices].tolist()
#         # generate return data
#         for i in range(len(nms_keep_boxes)):
#             bbox = nms_keep_boxes[i]
#             logit = nms_keep_logits[i]
#             nms_result.append({
#                 'image_id': IMAGE_ID,
#                 'category_id': cat_id,
#                 'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])],
#                 'score': round(logit, 3),
#             })
#
#     return nms_result


# overall nms
def perform_nms(result, iou_threshold=1.0):
    IMAGE_ID = result[0]['image_id']

    nms_result = []
    class_preds = {}
    class_preds[0] = {'score': [], 'bbox': [], "category_id": []}
    for res in result:
        # cat_id = res['category_id']
        # if cat_id not in class_preds:
        #     class_preds[cat_id] = {'score': [], 'bbox': []}
        class_preds[0]['score'].append(res['score'])
        class_preds[0]['bbox'].append(res['bbox'])
        class_preds[0]['category_id'].append(res['category_id'])
        # class_preds[0]['label'].append(res['label'])
    for _, preds in class_preds.items():
        cat_ids = torch.tensor(preds['category_id'])
        class_boxes = torch.tensor(preds['bbox'])  # xywh
        class_logits = torch.tensor(preds['score'])
        # category_names = torch.tensor(preds['label'])
        class_boxes[:, 2:] = class_boxes[:, :2] + class_boxes[:, 2:]
        nms_keep_indices = ops.nms(class_boxes.float(), class_logits, iou_threshold)
        nms_keep_category_ids = cat_ids[nms_keep_indices].tolist()
        nms_keep_boxes = class_boxes[nms_keep_indices].tolist()
        nms_keep_logits = class_logits[nms_keep_indices].tolist()
        # nms_keep_category_names = category_names[nms_keep_indices].tolist()
        # generate return data
        for i in range(len(nms_keep_boxes)):
            bbox = nms_keep_boxes[i]
            logit = nms_keep_logits[i]
            nms_result.append({
                'image_id': IMAGE_ID,
                'category_id': nms_keep_category_ids[i],
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])],
                'score': round(logit, 4),
                # 'label': nms_keep_category_names[i]
            })

    return nms_result