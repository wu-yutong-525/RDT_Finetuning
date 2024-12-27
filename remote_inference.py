import os
import yaml
import torch
import argparse
import numpy as np
import json
import cv2
import torchvision.transforms as transforms
from math import pi

from collections import deque
from PIL import Image as PImage
from RDT_UR5 import create_model
import numpy as np

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
global observation_window
observation_window = None
observation_window = deque(maxlen=2)

observation_window.append({
        'qpos': [],
        'images': {
            CAMERA_NAMES[0]: np.zeros((480,640,3)),
            CAMERA_NAMES[1]: np.zeros((480,640,3)),
            CAMERA_NAMES[2]: np.zeros((480,640,3)),
        }
})

def jpeg_mapping(img):
    img = cv2.imread(img)
    img = cv2.imencode('.jpg', img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl",
        pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
        control_frequency=args.ctrl_freq,
        device_1=args.device_1,
        device_2=args.device_2
    )
    return model

import os

def contains_png_files(path):
    """
    Return True if 'path' is:
        - an existing directory containing any .png files, or
        - a single .png file.
    Return False otherwise.
    """
    if not os.path.exists(path):
        return False
    if os.path.isfile(path):
        return path.lower().endswith('.png')
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith('.png'):
                return True
        return False

    return False

def inference_fn(policy, args, observation=observation_window):
    with torch.inference_mode():
        # while True and not rospy.is_shutdown():
        with open('./language_instruction.txt', 'r') as f:
            text = f.read().strip()
        with open('./qpos.txt', 'r') as f:
            # Read the line and strip any extra whitespace
            line = f.readline().strip()
            # Convert the string to a list of evaluated values
            print(line)
            values = [float(x) for x in line.strip("[]").split(", ")]
        # print(f"qpos: {values}")
        print(values)
        qpos_right = values
        qpos_right.append(0)
        print(f"qpos right: {qpos_right}")
        qpos = np.concatenate((np.zeros_like(qpos_right), qpos_right), axis=0)
        qpos = torch.from_numpy(qpos).to(torch.bfloat16)
        proprio = torch.tensor(qpos).unsqueeze(0).to(policy.device_2)
        img_right = "/home/yutong/RoboticsDiffusionTransformer/observations/observation.png"
        img_right = torch.from_numpy(jpeg_mapping(img_right))
        img_right = img_right.permute(2, 0, 1)

        # Resize the image to [C, 480, 640]
        resize = transforms.Resize((480, 640))
        img_right = resize(img_right)

        # Permute dimensions back to [H, W, C]text_embedp
        img_right = img_right.permute(1, 2, 0)
        
        img_front = None
        img_left = None
        
        img_front_prev = None
        img_left_prev = None
        img_right_prev =  "/home/yutong/RoboticsDiffusionTransformer/observations/prev_obs/observation.png"
        if contains_png_files(img_right_prev):
            img_right_prev = torch.from_numpy(jpeg_mapping(img_right_prev))
        else:
            img_right_prev = None
        img_right_prev = None

        print(f"image data type: {img_right.dtype}")
        print(f"image input size: {img_right.size()}")
        images = [img_front_prev,
                img_right_prev,
                img_left_prev,
              
                img_front,
                img_right,
                img_left]
        
        images = [PImage.fromarray(img.numpy()) if img is not None else None for img in images]

        print(proprio.size())
        lang_embeddings = policy.encode_instruction(text)
        actions = policy.step(proprio=proprio, images=images, text_embeds=lang_embeddings).squeeze(0).cpu().numpy()
        return actions

def _process_action(actions):
        actions[..., -1] = np.where(actions[..., -1] > 0.5, 1, -1)
        return actions

def main(args):
    policy = make_policy(args)

    # while True:
        # with open('~/RoboticsDiffusionTransformer/observation.json', 'r') as f:
        #     observation = json.load(f)
    print('Update image into image arrays')
    observation_window.append({
        'qpos': [],
        'images': {
            CAMERA_NAMES[0]: np.array(PImage.open('./observations/observation.png')),
            CAMERA_NAMES[1]: np.zeros((480,640,3)),
            CAMERA_NAMES[2]: np.zeros((480,640,3)),
            }
    }) 
    actions = inference_fn(policy, args)
    actions = np.array([_process_action(a) for a in actions])
    print(f"****INFERRED ACTIONS LENGTH***: {len(actions)}")
    print(f"****INFERRED ACTIONS TYPE***: {type(actions)}")  
    print(f"INFERRED ACTIONS: {actions.tolist()}")
    with open('./action.json', 'w') as f:
        for a in actions.tolist():
            f.write(str(a))
            f.write('\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_1', default=2)
    parser.add_argument('--device_2', default=3)
    parser.add_argument('--config_path', type=str, default="configs/base.yaml")
    # parser.add_argument('--pretrained_model_name_or_path', type=str,default="robotics-diffusion-transformer/rdt-170m")
    # parser.add_argument('--pretrained_model_name_or_path', type=str,default="robotics-diffusion-transformer/rdt-1b")
    parser.add_argument('--pretrained_model_name_or_path', type=str,default="/home/yutong/RDT/RoboticsDiffusionTransformer/checkpoints/rdt_ft_170m/checkpoint-58000")
    parser.add_argument('--ctrl_freq', type=int, default=25)
    args = parser.parse_args()

    main(args)
