# Training DETR (Detection Transformer)



A repository for training DETR on custom datasets.

Soon to be merged with the [Vision Transformers](https://github.com/sovit-123/vision_transformers) library. Some experiments for learning rate and transformer layer tuning may still continue here. **But all the major updates will be integrated into [Vision Transformers](https://github.com/sovit-123/vision_transformers)**.

## Currently Supported Models

* `detr_resnet50`
* `detr_resnet50_dc5`
* `detr_resnet101`
* `detr_resnet101_dc5`

All the models are loaded from PyTorch Hub - `facebookresearch/detr`.

## Quick Setup

```
git clone https://github.com/sovit-123/detr-custom-training.git
```

**Recommended to install [PyTorch 1.12.0 and Torchvision 0.13.0 from official](https://pytorch.org/get-started/previous-versions/#v1120) website with CUDA support and then do**:  

```
pip install -r requirements.txt
```

## Custom Training

* Check how to prepare dataset?

```
python train.py --epochs 100 --data data/custom_data.yaml --name custom_data
```

## Inference on Images

* Pretrained model inference

```
python inference.py --input path/to/image
```

* Using custom trained model (`--weights` should correspond to the correct `--model` )

```
python inference.py --input path/to/image --weights outputs/training/custom_training/best_model.pth --model detr_resnet50
                                                                                                            detr_resnet50_dc5
                                                                                                            detr_resnet101
                                                                                                            detr_resnet101_dc5
```

## Inference on Videos

* Pretrained model inference

```
python inference_video.py --input path/to/video
```

* sing custom trained model (`--weights` should correspond to the correct `--model` )

```
python inference.py --input path/to/video --weights outputs/training/custom_training/best_model.pth --model detr_resnet50
                                                                                                            detr_resnet50_dc5
                                                                                                            detr_resnet101
                                                                                                            detr_resnet101_dc5
```

## Custom Dataset Preparation

The content of the `custom_data.yaml` (in `data` directory) should be the following:

```
# Images and labels direcotry should be relative to train.py
TRAIN_DIR_IMAGES: ../custom_data/train/images
TRAIN_DIR_LABELS: ../custom_data/train/annotations
# VALID_DIR should be relative to train.py
VALID_DIR_IMAGES: ../custom_data/valid/images
VALID_DIR_LABELS: ../custom_data/valid/annotations

# Class names.
CLASSES: [
    '__background__',
    'smoke'
]

# Number of classes (object classes + 1 for background class).
NC: 2

# Whether to save the predictions of the validation set while training.
SAVE_VALID_PREDICTION_IMAGES: True
```

* `images` directory should contain all the images.
* `annotations` directory should contain all the annotations in XML format (similar to Pascal VOC XML files).
* **OR** both images and annotations can be in the same directory for a particular split (train/valid). Just provide the same path in the *DIR_IMAGES and *DIR_LABELS attribute in the YAML file.
