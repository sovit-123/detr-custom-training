# Training DETR (Detection Transformer)



A repository for training DETR on custom datasets.

Soon to be merged with the [Vision Transformers](https://github.com/sovit-123/vision_transformers) library.

## Quick Setup

```
git clone https://github.com/sovit-123/detr-custom-training.git
```

**Recommended to install [PyTorch 1.12.0 and Torchvision 0.13.0 from official](https://pytorch.org/get-started/previous-versions/#v1120) website with CUDA support and then do**:  

```
pip install -r requirements.txt
```

## Custom Training

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

