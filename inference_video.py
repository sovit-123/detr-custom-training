import torch
import cv2
import numpy as np
import albumentations as A
import argparse
import yaml
import glob
import os
import time

from model import DETRModel
from utils.general import (
    rescale_bboxes,
    set_infer_dir,
    load_weights
)
from utils.transforms import infer_transforms, resize

np.random.seed(2023)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', 
        '--weights',
    )
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '--model', 
        default='detr_resnet50',
        help='name of the model'
    )
    parser.add_argument(
        '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', 
        '--img-size', 
        default=640,
        dest='img_size',
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-t', 
        '--threshold',
        type=float,
        default=0.5,
        help='confidence threshold for visualization'
    )
    parser.add_argument(
        '--name', 
        default=None, 
        type=str, 
        help='training result dir name in outputs/training/, (default res_#)'
    )
    args = parser.parse_args()
    return args

def read_return_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height

def main(args):
    NUM_CLASSES = None
    CLASSES = None
    data_configs = None
    if args.data is not None:
        with open(args.data) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']
    
    DEVICE = args.device
    OUT_DIR = set_infer_dir(args.name)

    model, CLASSES = load_weights(
        args, DEVICE, DETRModel, data_configs, NUM_CLASSES, CLASSES
    )
    _ = model.to(DEVICE).eval()

    # Colors for visualization.
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    VIDEO_PATH = args.input

    cap, frame_width, frame_height = read_return_video_data(VIDEO_PATH)

    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))
    if args.img_size != None:
        RESIZE_TO = args.img_size
    else:
        RESIZE_TO = frame_width

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            orig_frame = frame.copy()
            frame = resize(frame, RESIZE_TO, square=True)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = infer_transforms(image)
            image = torch.tensor(image, dtype=torch.float32)
            image = torch.permute(image, (2, 0, 1))
            image = image.unsqueeze(0)
            
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                outputs = model(image.to(args.device))
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time
            
            # Get the current fps.
            fps = 1 / (forward_pass_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > args.threshold
        
            # convert boxes from [0; 1] to image scales
            boxes = rescale_bboxes(
                outputs['pred_boxes'][0, keep].detach().cpu(), 
                (orig_frame.shape[1], orig_frame.shape[0])
            )
            probas = probas[keep]

            for i, box in enumerate(boxes):
                label = int(probas[i].argmax())
                class_name = CLASSES[label]
                color = COLORS[CLASSES.index(class_name)]
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                cv2.rectangle(
                    orig_frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    color, 
                    2
                )
                cv2.putText(
                    orig_frame,
                    text=class_name,
                    org=(xmin, ymin-5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=color,
                    thickness=2,
                    fontScale=1
                )
            cv2.putText(
                orig_frame, 
                f"FPS: {str(int(fps))}", 
                (25, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.8, 
                color=(0, 0, 255), 
                thickness=2, 
                lineType=cv2.LINE_AA
            )
            cv2.imshow('Image', orig_frame)
            # Press `q` to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)