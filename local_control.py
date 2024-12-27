import time
import torch
import yaml
import os
import json
import argparse
from PIL import Image
import numpy as np
import paramiko

from scp import SCPClient
from realworld.realenv import UR5RealEnv
from realworld.wrapper import ActionWrapper, DofWrapper, FrameResizeWrapper, FrameStackWrapper, PropStackWrapper
from collections import deque
import cv2

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
global observation_window
observation_window = None
# observation_window = deque(maxlen=2)

# observation_window.append({
#         'qpos': [],
#         'images': {
#             CAMERA_NAMES[0]: torch.zeros((480,640,3)).tolist(),
#             CAMERA_NAMES[1]: torch.zeros((480,640,3)).tolist(),
#             CAMERA_NAMES[2]: torch.zeros((480,640,3)).tolist(),
#         }
# })
# def jpeg_mapping(img):
#     img = cv2.imencode('.jpg', img)[1].tobytes()
#     img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
#     return img
def create_ssh_client(server, port,user, password, kp='qzpubrsa'):
    client=paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pk=paramiko.RSAKey.from_private_key_file(kp, password=password)
    client.connect(server, port=port, username=user, pkey=pk)
    return client

def update_observation_window(env):
    global observation_window

    img_left = env.get_obs()['rdt']
    print(f"image type:{type(img_left)}")
    print(f"image type:{img_left.shape}")
    if observation_window is None:
        observation_window = deque(maxlen=2)
        # observation_window.append({'qpos': None, 'images': {name: None for name in CAMERA_NAMES}})
        observation_window.append({
        'qpos': [],
        'images': {
            CAMERA_NAMES[0]: torch.zeros((480,640,3)).tolist(),
            CAMERA_NAMES[1]: torch.zeros((480,640,3)).tolist(),
            CAMERA_NAMES[2]: torch.zeros((480,640,3)).tolist(),
            }
        })

    # img_right = jpeg_mapping(img_right)
    img_right = np.zeros_like(img_left)
    img_front = np.zeros_like(img_left)
    qpos_right = env.cur_qpos
    qpos = np.concatenate((np.zeros_like(qpos_right), qpos_right), axis=0)
    qpos = torch.from_numpy(qpos).float()

    observation_window.append({
        'qpos': qpos.tolist(),
        'images': {
            CAMERA_NAMES[0]: img_front.tolist(),
            CAMERA_NAMES[1]: img_right.tolist(),
            CAMERA_NAMES[2]: img_left.tolist(),
        }
    })

def send_observations_to_server(ssh, env, remote_path):
    print('Now send observation picture to remote sensor')
    image_array = env.get_obs()['trans']
    image_array = np.transpose(image_array, (1,2,0))
    image = Image.fromarray(image_array)
    image.save('observation.png')
    with SCPClient(ssh.get_transport()) as scp:
        scp.put('observation.png', remote_path)
        print('Observation picture sent to remote sensor successfully')


def write_imaghes_to_json(observation_window,filename):
    images_data = []
    for obs in observation_window:
        images = obs['images']
        images_data.append(images)
    with open(filename, 'w') as f:
        json.dump(images_data, f)

def send_data_to_server(ssh, data, chunk_size=100000):
    print("Sending observation data to server...")
    data_json = json.dumps(data)
    data_size = len(data_json.encode('utf-8'))
    for i in range(0, data_size, chunk_size):
        chunk = data_json[i:i+chunk_size]
        stdin, stdout, stderr = ssh.exec_command(f"echo '{chunk}' >> ~/RoboticsDiffusionTransformer/observation.json")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            print(f"Failed to send observation data. Exit status: {exit_status}, Error: {stderr.read().decode('utf-8')}")
            break
    else:
        print("Observation data sent successfully.")

def get_action_from_server(ssh):
    print("Retrieving action data from server...")
    stdin, stdout, stderr = ssh.exec_command("cat ~/RoboticsDiffusionTransformer/action.json")
    action = stdout.read().decode('utf-8')
    if action:
        print("Action data retrieved successfully.")
        return np.fromstring(action, sep=',')
    else:
        print("Failed to retrieve action data.")
        return np.array([])  # Return empty array in case of failure

def load_actions_from_json(filename='action.json'):
    actions = [] 
    with open(filename, 'r') as data:
        for _, l in enumerate(data):
            data = json.loads(l.strip())
            action_vectors = np.array(data, dtype=np.float32)
            actions.append(action_vectors)
    return actions

def get_action_from_server(ssh, local_path='action.json'):
    print("Retrieving action data from server...")
    # stdin, stdout, stderr = ssh.exec_command("cat ~/RoboticsDiffusionTransformer/action.json")
    # error = stderr.read().decode('utf-8')
    
    # if error:
    #     print(f"get action error: {error}")
    #     return np.array([])  # Return empty array in case of error

    # action_data = stdout.read().decode('utf-8')
    # if action_data:
    #     print("Action data retrieved successfully.")
    #     try:
    #         # Parse the JSON data
    #         actions = load_actions_from_json(action_data)
    #         # Assuming actions is a list of actions
    #         return actions # Convert to NumPy array if needed
    #     except json.JSONDecodeError as e:
    #         print(f"Failed to parse JSON: {e}")
    #         return np.array([])  # Return empty array in case of failure
    # else:
    #     print("Failed to retrieve action data.")
    #     return np.array([])  # Return empty array in case of failure
    try:
        sftp = ssh.open_sftp()
        sftp.get('./RoboticsDiffusionTransformer/action.json', local_path)
        print("Action data retrieved successfully.")
        actions  = load_actions_from_json()
        return actions
    except Exception as e:
        print(f"Failed to retrieve action data: {e}")
        return np.array([])

def send_language_instruction(ssh, instruction):
    print("Sending language instruction to server...")
    stdin, stdout, stderr = ssh.exec_command(f"echo '{instruction}' > ~/RoboticsDiffusionTransformer/language_instruction.txt")
    exit_status = stdout.channel.recv_exit_status()  # Wait for command to finish
    if exit_status == 0:
        print("Language instruction sent successfully.")
    else:
        print(f"Failed to send language instruction. Exit status: {exit_status}, Error: {stderr.read().decode('utf-8')}")

def send_cur_qpos(ssh, env):
    print("Sending current position to server...")
    curpos = env._tcp_pos
    # curpos.append(0)
    stdin, stdout, stderr = ssh.exec_command(f"echo '{curpos}' > ~/RoboticsDiffusionTransformer/qpos.txt")
    print(f"Current position: {env._tcp_pos}")
    exit_status = stdout.channel.recv_exit_status()  # Wait for command to finish
    if exit_status == 0:
        print("qpos sent successfully.")
    else:
        print(f"Failed to send qpos. Exit status: {exit_status}, Error: {stderr.read().decode('utf-8')}")


def main(args):
    #config = {'episode_len': args.max_publish_step, 'state_dim': 14, 'chunk_size': args.chunk_size, 'camera_names': CAMERA_NAMES}
    
    env = UR5RealEnv(robot_ip='192.169.0.10', dt=0.08, multi_process=True, control_mode=UR5RealEnv.ControlMode.JOINT, delta_action=True, moving_average=1)
    env = ActionWrapper(env, minimum=-1.0, maximum=1.0)
    # env = DofWrapper(env, [None, None, None, 0., 0., 0., None])
    env = FrameResizeWrapper(env, args.img_size_1, args.img_size_2, rot=-1)
    env = FrameStackWrapper(env, num_frames=1)
    env = PropStackWrapper(env, num_frames=3)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect(args.server_ip, port=10259, username=args.username)


    kp = "qz_pubrsa"
    k = 'zmnYTW#24686197'
    pk = paramiko.RSAKey.from_private_key_file(kp, password=k)
    ssh.connect(args.server_ip, port=10259, username=args.username,pkey=pk)

    # Change directory and run the inference script on the server
    # commands = [
    #     "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rdt && cd ~/RoboticsDiffusionTransformer && python3 remote_inference.py"
    # ]
    # for cmd in commands:
    #     stdin, stdout, stderr = ssh.exec_command(cmd)
    #     print(f"Command: {cmd}")
    #     print(f"Output: {stdout.read().decode('utf-8')}")
    #     print(f"Error: {stderr.read().decode('utf-8')}")

    env.reset()
    # env.control_mode = UR5RealEnv.ControlMode.TCP
    pre_action = env.init_qpos
    action = None
    step = 0
    remote_path = '~/RoboticsDiffusionTransformer/observations'
    # print(env._tcp_pos)
    # print(pre_action)
    while True:
        send_observations_to_server(ssh, env, remote_path)
        send_cur_qpos(ssh, env)
        update_observation_window(env)
        step += 1
        images_data = {}
        
        for i, obs in enumerate(observation_window):
            images = obs['images']
            if i < 1:
                for k,v in images.items():
                    images_data[f"{k}_prev"] = v
            else:
                for k,v in images.items():
                    images_data[k] = v
        send_data_to_server(ssh, images_data)

        # Get language instruction from user
        instruction = input("Enter the language instruction: ")
        send_language_instruction(ssh, instruction)
        print("Tesing pre-defined instruction")
        commands = [
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rdt && cd ~/RoboticsDiffusionTransformer && python3 remote_inference.py && cd ~/RoboticsDiffusionTransformer/observations && mv observation.png prev_obs"
        ]
        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            print(f"Command: {cmd}")
            print(f"Output: {stdout.read().decode('utf-8')}")
            print(f"Error: {stderr.read().decode('utf-8')}")
        actions = get_action_from_server(ssh)
        print(f"*****Num Action Steps Inferred******: {len(actions)}")
        for a in actions[0:64]:
            print(f"Action: {a}")
            env.act(a[7:14])
            time.sleep(0.5)
        print(f"Published Step: {step}")

    env.stop()
    ssh.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='10.220.5.14')
    parser.add_argument('--username', type=str, default='yutong')
    parser.add_argument('--img_size_1',type=int, default=640)
    parser.add_argument('--img_size_2', type=int, default=480)
    # parser.add_argument('--max_publish_step', type=int, default=1000)
    # parser.add_argument('--chunk_size', type=int, default=64)
    args = parser.parse_args()

    main(args)
