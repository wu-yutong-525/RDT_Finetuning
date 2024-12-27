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
# t5_tokenizer = AutoTokenizer.from_pretrained(
#             "google/t5-v1_1-xxl",
#             model_max_length=120, # From RDT/model/t5_encoder.py
#             cache_dir="./cache/t5_tk_cache",
#             local_files_only=False, # From RDT/model/t5_encoder.py
#         )
# t5_model = T5EncoderModel.from_pretrained(
#             "google/t5-v1_1-xxl",
#             cache_dir="./cache/t5_m_cache",
#             local_files_only=False,
#     ).eval()

# t5_model = torch.nn.DataParallel(t5_model, device_ids=[args.device_1, args.device_2])
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

    # 1. Check if the path exists at all
    if not os.path.exists(path):
        return False

    # 2. If the path is a file, just check its extension
    if os.path.isfile(path):
        return path.lower().endswith('.png')

    # 3. If the path is a directory, list files and check
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith('.png'):
                return True
        return False

    # Fallback (unlikely case)
    return False


def get_text_embeddings(args, texts, model):
    tokenizer,encoder = model.get_text_enboder()
    text_tokens_and_mask = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = text_tokens_and_mask["input_ids"].to(args.device_2)
    attention_mask = text_tokens_and_mask["attention_mask"].to(args.device_2)
    with torch.no_grad():
        text_encoder_embs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"].detach()
    return text_encoder_embs, attention_mask

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

        # img_front = torch.zeros_like(img_right)
        # img_left = torch.zeros_like(img_right)
        
        img_front = None
        img_left = None

        # img_front_prev = torch.zeros_like(img_right)
        # img_left_prev = torch.zeros_like(img_right)
        # img_right_prev = torch.zeros_like(img_right)
        
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
        # images = [observation[-2]['images'][name] for name in CAMERA_NAMES]+ [observation[-1]['images'][name] for name in CAMERA_NAMES]
        # print(len(images))
        # print(type(images[0]))
        # images = [np.uint8(img * 255) if img.dtype != np.uint8 else img for img in images]
        images = [PImage.fromarray(img.numpy()) if img is not None else None for img in images]
        # print(len(images))
        print(proprio.size())
        lang_embeddings = policy.encode_instruction(text)
        actions = policy.step(proprio=proprio, images=images, text_embeds=lang_embeddings).squeeze(0).cpu().numpy()
        return actions

def _process_action(actions):
        # if self.args.use_actions_interpolation:
        #     processed_actions = self._interpolate_action(action)
        # else:
        # actions = np.squeeze(actions)
        # print(f"actions shape: {actions.shape}, {actions}")
        actions[..., -1] = np.where(actions[..., -1] > 0.5, 1, -1)
        # eef_position = actions[..., :3]
        # eef_euler = actions[..., 3:6]

        # return (eef_position, eef_euler, gripper_action)
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
