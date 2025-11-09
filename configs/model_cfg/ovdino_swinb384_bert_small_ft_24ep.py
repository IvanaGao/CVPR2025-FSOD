model_name = 'ovdino_grounding_base'

text_encoder = '/your_path_to_bert/bert_uncased_L4_H512_A8/'

backbone = 'swin_B_384_22k'
out_indices = [1, 2, 3]

# backbone路径，设置None表示在创建模型过程中不单独加载backbone权重，在微调场景如果单独加载该权重，在加载预训练权重时会将其覆盖
# backbone_path = None
backbone_path = '/your_path_to_swin/swin_base_patch4_window12_384_22k.pth'

clip_grad_enabled = True
clip_grad_params_max_norm = 0.1
clip_grad_params_norm_type = 2

# train_class_num = 365   # 数据集相关参数
# max_class_num = 150     # caption训练类别最大数
# train_num_classes = 80   # 设置150用于模型初始化
# train_num_classes = 6  # 设置150用于模型初始化
# finetune_num_classes = 80 # 具体微调数据集的类别数
# test_num_classes = 6

# batch size 需要配合学习率使用（ov-dino参考的是总batch size，而不是单卡batch size）
# batch_size = 8
batch_size = 1

# modify optimizer config
# optimizer_lr = 1e-4
# optimizer_betas = (0.9, 0.999)
weight_decay = 1e-4
lr_model = 1e-5
lr_backbone = 1e-6
lr_backbone_names = ['language_backbone']   # 模型权重名称
multi_step_lr = True
epochs = 24
lr_drop_list = [max(int(0.8*epochs), epochs-6), max(int(0.9*epochs), epochs-3)]

freeze_keywords = []

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 64/4 = 4
total_batch_size = 1

use_coco_eval = True
use_checkpoint = True

num_feature_levels = 4
num_queries = 900

# transformer_layers = 6
transformer_layers = 10
ovdino_embed_dim = 512

pixel_mean = [123.675, 116.280, 103.530]
pixel_std = [58.395, 57.120, 57.375]
aux_loss = True
select_box_nums_for_evaluation = 300
dn_number = 100
label_noise_ratio = 0.5
box_noise_scale = 1.0
input_format = "RGB"
vis_period = 0

rm_train_null_sample = True

use_pn_dist_loss = False
