# Solution For Foundational FSOD with RoboFlow-20VL
Foundational FSOD challenge is a part of the Workshop on Visual Perception and Learning in an Open World at **CVPR 2025** 🔥🔥

Challenge Link: https://github.com/anishmadan23/foundational_fsod/tree/fsod_rf20vl?tab=readme-ov-file&ref=blog.roboflow.com

## About the Challenge
Vision-Language Models (VLMs) like GroundingDINO have demonstrated remarkable zero-shot 2D detection performance on standard benchmarks like COCO. 
However, such foundational models may still be sub-optimal for specific target applications like medical and aerial image analysis.

The Foundational Few-shot Object Detection (Foundational FSOD) challenge is about finding solutions to quickly adapt pre-trained VLMs to novel domains (such as medical and aerial images) only using a few samples for object detection. 

## Environment
```
python                        3.8.10
torch                         2.0.1+cu118
torchaudio                    2.0.2+cu118
torchtext                     0.13.0a0+fae8e8c
torchvision                   0.15.2+cu118
triton                        2.0.0
triton-nightly                2.1.0.dev20230822000928
pycocotools                   2.0.7
Jinja2                        3.1.2
```
The above library versions are used throughout our experiments. You can try other versions as well. 

## Data Preparation
For data preparation, please follow the instructions from the competition official website: https://eval.ai/web/challenges/challenge-page/2459/overview?ref=blog.roboflow.com.
When downloading the datasets, remember to select the COCO JSON format to download. 
Then you need to generate the configure file for the dataset, please follow the format in *configs/dataset_cfg/fsod/grounding_ft_dataset1_1.json*

## Model Training
Our solution consists of four steps. We now provide detailed instructions to reproduce our results. 

### Step 1: text prompt optimization
We notice that the text category descriptions in the Robowflow20vl benchmark is sub-optimal, so we prompt Qwen2.5-VL to generate better class descriptions. 
For each dataset and for each class, Qwen2.5-VL is prompted with an image with a red bounding box surrounding a target object to generate descriptive category names. 

An example prompt could be:

*”You are assisting in improving object detection by generating optimal category. Given an image with a red bounding box and an initial category description [original class description], your task is to produce five enhanced short category terms.”*

After generating improved class descriptions for each class, we prompt GroundingDINO to select the best set of class descriptions based on the model's zero-shot performance on the test set. 

For the deployment of Qwen2.5-VL, you can follow instructions from lmdeploy, https://github.com/InternLM/lmdeploy.

For evaluating the zero-shot performance of GroundingDINO, you can run:
```
python3 zero-shot-eval.py --output_dir [output directory] --pretrain_model_path [path to pretrained model weight] --datasets [path to the data configuration file]
```

### Step 2: finetune GroundingDINO on original annotations with improved class descriptions
To run the model, you need to download the weights of *bert_uncased_L4_H512_A8* and *swin_base_patch4_window12_384_22k.pth* by yourself.
Then you need to configure paths of these weights in configs/model_cfg/ovdino_swinb384_bert_small_ft_24ep.py, for 'text_encoder' and 'backbone_path' field respectively. 

To finetune GroundingDINO:
```
python3 finetune.py --output_dir [output directory] --pretrain_model_path [path to pretrained model weight] --datasets [path to the data configuration file] --train_num_classes [number of classes]
```

### Step 3: prompt the finetuned GroundingDINO on the training data to generate pseudo-labels
We notice that the data annotation on the training data of the Roboflow20-VL benchmark is sparse (i.e., many target objects are missed). So we prompt the GroundingDINO finetuned in Step2 on the training data to generate pseudo-labels. 

To perform inference:
```
python3 pseudolabel.py --original_data_config [path to the data configuration file] --out_dir [output directory] --model_path [path to model weight obtained in Step2] --train_num_classes [number of classes]
```

### Step 4: finetune GroundingDINO on pseudo-labeled annotations
As before, you need to generate the configuration file for the pseudo-labeled data following *configs/dataset_cfg/fsod/grounding_ft_dataset1_1.json*

To finetune GroundingDINO:
```
python3 finetune.py --output_dir [output directory] --pretrain_model_path [path to model weight obtained in Step2] --datasets [path to the data configuration file for pseudo-labeled data in Step 3] --train_num_classes [number of classes]
```
