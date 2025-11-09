import torch
from models.visual_backbone.backbone import build_swin_backbone
from ..registry import MODULE_BUILD_FUNCS
from detrex.modeling.language_backbone.bert import BERTEncoder
from detrex.layers.position_embedding import PositionEmbeddingSine
from detrex.modeling.neck.channel_mapper import ChannelMapper
from projects.ovdino.modeling.dino_transformer import DINOTransformer, DINOTransformerEncoder, DINOTransformerDecoder
from projects.ovdino.modeling.dn_criterion import DINOCriterion
# from projects.ovdino.modeling.ovdino import OVDINO
from models.OVDino.ovdino import OVDINO
from detrex.modeling.matcher.matcher import HungarianMatcher
from detectron2.layers import ShapeSpec


@MODULE_BUILD_FUNCS.registe_with_name(module_name="ovdino_grounding_base")
def build_ovdino_grounding_base(args, logger=None):

    backbone = build_swin_backbone(
        args,
        out_indices=args.out_indices,
        drop_path_rate=0.1,
        use_checkpoint=args.use_checkpoint,
        logger=logger,
    )
    language_backbone = BERTEncoder(
        tokenizer_cfg={'tokenizer_name': args.text_encoder},
        model_name=args.text_encoder,
        # output_dim=256,
        output_dim=args.ovdino_embed_dim,
        pooling_mode="mean",
        post_tokenize=True,
    )
    position_embedding = PositionEmbeddingSine(
        # num_pos_feats=128,
        num_pos_feats=args.ovdino_embed_dim // 2,   # 位置编码使用xy位置编码cat，所以为args.ovdino_embed_dim的 1/2
        temperature=10000,
        normalize=True,
        offset=-0.5,
    )
    neck = ChannelMapper(
        input_shapes={
            # "p1": ShapeSpec(channels=192),
            "p1": ShapeSpec(channels=backbone.num_features[1]),
            # "p2": ShapeSpec(channels=384),
            "p2": ShapeSpec(channels=backbone.num_features[2]),
            # "p3": ShapeSpec(channels=768),
            "p3": ShapeSpec(channels=backbone.num_features[3]),
        },
        in_features=["p1", "p2", "p3"],
        # out_channels=256,
        out_channels=args.ovdino_embed_dim,
        num_outs=4,
        kernel_size=1,
        norm_layer=torch.nn.GroupNorm(
            num_groups=32,
            # num_channels=256
            num_channels=args.ovdino_embed_dim
        ),
    )

    dino_transformer_encoder = DINOTransformerEncoder(
        # embed_dim=256,
        embed_dim=args.ovdino_embed_dim,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        # num_layers=6,
        num_layers=args.transformer_layers,
        post_norm=False,
        num_feature_levels=args.num_feature_levels,
        use_checkpoint=False,
    )
    dino_transformer_decoder = DINOTransformerDecoder(
        # embed_dim=256,
        embed_dim=args.ovdino_embed_dim,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        # num_layers=6,
        num_layers=args.transformer_layers,
        return_intermediate=True,
        num_feature_levels=args.num_feature_levels,
        use_checkpoint=False,
    )
    transformer = DINOTransformer(
        encoder=dino_transformer_encoder,
        decoder=dino_transformer_decoder,
        num_feature_levels=args.num_feature_levels,
        two_stage_num_proposals=args.num_queries,
    )

    criterion = DINOCriterion(
        # num_classes=args.train_num_classes,
        # num_classes=args.max_class_num if args.train_class_num is None else min(args.max_class_num, args.train_class_num),
        num_classes=args.train_num_classes,
        matcher=HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    )

    model = OVDINO(
        backbone=backbone,
        language_backbone=language_backbone,
        position_embedding=position_embedding,
        neck=neck,
        transformer=transformer,
        criterion=criterion,
        embed_dim=args.ovdino_embed_dim,
        num_classes=args.train_num_classes,
        # num_classes=args.max_class_num,
        # num_classes=150,    # 预训练模型默认是150，模型创建后再将 model.num_classes
        # test_num_classes=args.test_class_num,
        num_queries=args.num_queries,
        pixel_mean=args.pixel_mean,
        pixel_std=args.pixel_std,
        aux_loss=args.aux_loss,
        select_box_nums_for_evaluation=args.select_box_nums_for_evaluation,
        device=args.device,
        dn_number=args.dn_number,
        label_noise_ratio=args.label_noise_ratio,
        box_noise_scale=args.box_noise_scale,
        input_format=args.input_format,
        vis_period=args.vis_period,
        inference_template=args.inference_template,
        use_pn_dist_loss=args.use_pn_dist_loss,
    )

    if logger is not None:
        logger.info('model.inference_template = {}'.format(model.inference_template))
    else:
        print('model.inference_template = {}'.format(model.inference_template))

    # set aux loss weight dict
    import copy
    base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
    if model.aux_loss:
        weight_dict = model.criterion.weight_dict
        aux_weight_dict = {}
        aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
        aux_weight_dict.update({k + "_bcls_enc": v for k, v in base_weight_dict.items()})
        for i in range(model.transformer.decoder.num_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        model.criterion.weight_dict = weight_dict

    return model