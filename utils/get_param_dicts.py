import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_groups_and_set_lr(args, model_without_ddp: nn.Module):

    # by default
    # import pdb;pdb.set_trace()
    param_dicts = []
    # （1）模型学习率
    temp_dict = {"params": [], "lr": args.lr_model}
    for n, p in model_without_ddp.named_parameters():
        if not match_name_keywords(n, args.lr_backbone_names) and p.requires_grad:
            temp_dict['params'].append(p)
    param_dicts.append(temp_dict)
    # （2）backbone学习率
    temp_dict = {"params": [], "lr": args.lr_backbone}
    for n, p in model_without_ddp.named_parameters():
        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad:
            temp_dict['params'].append(p)
    param_dicts.append(temp_dict)

    return param_dicts

