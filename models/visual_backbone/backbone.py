import torch
from models.visual_backbone.swin import SwinTransformer
# from models.visual_backbone.vmamba.vmamba import Backbone_VSSM


def build_swin_backbone(args, logger=None, **kwargs):

    try:
        backbone = args.backbone
    except:
        if logger is None:
            print('args error: no args.visual_backbone')
        else:
            logger.info('args error: no args.visual_backbone')

    assert backbone in ["swin_T_224_1k", "swin_B_224_22k", "swin_B_384_22k", "swin_L_224_22k", "swin_L_384_22k", "swin_H_384_22k"]

    model_params_dict = {
        "swin_T_224_1k": dict(pretrain_img_size=224, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7),
        "swin_B_224_22k": dict(pretrain_img_size=224, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7),
        "swin_B_384_22k": dict(pretrain_img_size=384, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12),
        "swin_L_224_22k": dict(pretrain_img_size=224, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=7),
        "swin_L_384_22k": dict(pretrain_img_size=384, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12),
        "swin_H_384_22k": dict(pretrain_img_size=384, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12),
    }
    kw_cgf = model_params_dict[backbone]
    kw_cgf.update(kwargs)   # 更新可选的参数，例如 'out_indices'、'drop_path_rate'等
    model = SwinTransformer(**kw_cgf)

    # 如果提供初始化权重那么加载权重（创建模型后如果加载预训练模型权重，该部分权重参数将被覆盖）
    if args.backbone_path is not None:
        if logger is None:
            print('loading {} from {}'.format(args.backbone, args.backbone_path))
        else:
            logger.info('loading {} from {}'.format(args.backbone, args.backbone_path))
        _load_state_output = model.load_state_dict(torch.load(args.backbone_path, map_location='cpu')['model'], strict=False)
        if logger is None:
            print('missing_keys = {}'.format(_load_state_output.missing_keys))
            print('unexpected_keys = {}'.format(_load_state_output.unexpected_keys))
        else:
            logger.info('missing_keys = {}'.format(_load_state_output.missing_keys))
            logger.info('unexpected_keys = {}'.format(_load_state_output.unexpected_keys))

    return model


def build_vmamba_backbone(args, logger=None, **kwargs):

    try:
        backbone = args.backbone
    except:
        if logger is None:
            print('args error: no args.backbone')
        else:
            logger.info('args error: no args.backbone')

    assert backbone in ["vmamba_tiny_s2l5", "vmamba_tiny_s1l8"]

    model_params_dict = {
        "vmamba_tiny_s1l8": dict(imgsize=224, embed_dim=96, depths=[2, 2, 8, 2], drop_path_rate=0.2, ssm_ratio=1.0),
        "vmamba_tiny_s2l5": dict(imgsize=224, embed_dim=96, depths=[2, 2, 5, 2], drop_path_rate=0.2, ssm_ratio=2.0),
    }
    # 共用参数
    channel_first = True
    comm_params = dict(
        ssm_d_state=1, ssm_dt_rank="auto", ssm_act_layer="silu", patch_size=4, in_chans=3, num_classes=1000, ssm_conv=3,
        ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v05_noz", mlp_ratio=4.0,
        mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False, patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2", use_checkpoint=False, posembed=False, imgsize=224,
    )
    # 添加共用参数
    for key in model_params_dict.keys():
        model_params_dict[key].update(comm_params)

    kw_cgf = model_params_dict[backbone]
    kw_cgf.update(kwargs)   # 更新可选的参数，例如 'out_indices'、'drop_path_rate'等
    # model = SwinTransformer(**kw_cgf)
    model = Backbone_VSSM(**kw_cgf)

    # 如果提供初始化权重那么加载权重（创建模型后如果加载预训练模型权重，该部分权重参数将被覆盖）
    if args.backbone_path is not None:
        if logger is None:
            print('loading {} from {}'.format(args.backbone, args.backbone_path))
        else:
            logger.info('loading {} from {}'.format(args.backbone, args.backbone_path))
        _load_state_output = model.load_state_dict(torch.load(args.backbone_path, map_location='cpu')['model'], strict=False)
        if logger is None:
            print('missing_keys = {}'.format(_load_state_output.missing_keys))
            print('unexpected_keys = {}'.format(_load_state_output.unexpected_keys))
        else:
            logger.info('missing_keys = {}'.format(_load_state_output.missing_keys))
            logger.info('unexpected_keys = {}'.format(_load_state_output.unexpected_keys))

    return model

