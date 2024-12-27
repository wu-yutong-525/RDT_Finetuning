import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

# from test_6drot import convert_euler_to_rotation_matrix,compute_ortho6d_from_rotation_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
from configs.state_vec import STATE_VEC_IDX_MAPPING

def convert_euler_to_rotation_matrix(euler):
    """
    Convert Euler angles (rpy) to rotation matrix (3x3).
    """
    quat = R.from_euler('xyz', euler).as_matrix()

    return quat

def compute_ortho6d_from_rotation_matrix(matrix):
    # The ortho6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d

def convert_action(input):
    # Extract EEF position (3 values)
    eef_pos = input[:, :3]  # (N, 3)
    
    # Convert EEF rotation (3 Euler angles to rotation matrix, then to 6D)
    eef_ang = convert_euler_to_rotation_matrix(input[:, 3:6])
    eef_ang = compute_ortho6d_from_rotation_matrix(eef_ang)  # (N, 6)
    
    # Gripper state (normalized to range [0, 1])
    gripper_open = (input[:, 6] + 1) / 2  # Assuming gripper state is at index 6
    gripper_open = gripper_open[:, np.newaxis]  # (N, 1)
    
    # Action vector: [gripper_state, eef_pos_x, eef_pos_y, eef_pos_z, eef_angle_0, ..., eef_angle_5]
    output = np.concatenate([gripper_open, eef_pos, eef_ang], axis=1)  # (N, 10)
    
    return output


def convert_robot_obs(input):
    # Extract EEF position (3 values)
    eef_pos = input[:, :3]  # (N, 3)
    
    # Convert EEF rotation (3 Euler angles to rotation matrix, then to 6D)
    eef_ang = convert_euler_to_rotation_matrix(input[:, 3:6])
    eef_ang = compute_ortho6d_from_rotation_matrix(eef_ang)  # (N, 6)
    
    # Gripper state (normalized to range [0, 1])
    gripper_open = (input[:, 12] + 1) / 2  # Assuming gripper state is at index 12 in raw state data
    gripper_open = gripper_open[:, np.newaxis]  # (N, 1)
    
    # State vector: [gripper_state, eef_pos_x, eef_pos_y, eef_pos_z, eef_angle_0, ..., eef_angle_5]
    output = np.concatenate([gripper_open, eef_pos, eef_ang], axis=1)  # (N, 10)
    
    return output


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = "/home/yutong/RoboticsDiffusionTransformer/data/datasets/UR5_data/"
        self.DATASET_NAME = "berkeley_autolab_ur5"
        
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['robot_state'][()]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            
            # Load the instruction
            dir_path = os.path.dirname(file_path)
            # with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
            #     instruction_dict = json.load(f_instr)
            # We have 1/3 prob to use original instruction,
            # 1/3 to use simplified instruction,
            # and 1/3 to use expanded instruction.
            # instruction_type = np.random.choice([
            #     'instruction', 'simplified_instruction', 'expanded_instruction'])
            # instruction = instruction_dict[instruction_type]
            # if isinstance(instruction, list):
            #     instruction = np.random.choice(instruction)
            # You can also use precomputed language embeddings (recommended)
            instruction = "/home/yutong/RoboticsDiffusionTransformer/out/pick_cube_embeddings.pt"
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Rescale gripper to [0, 1]
            # qpos = qpos / np.array(
            #    [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
            # )
            # target_qpos = f['action'][step_id:step_id+self.CHUNK_SIZE] / np.array(
            #    [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
            # )
            
            # Parse the state and action
            qpos = convert_robot_obs(qpos)
            state = qpos[step_id:step_id+1]
            # print(f"state size: {state.size()}")
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))

            actions = f["action"][()]
            actions = actions[step_id : step_id + self.CHUNK_SIZE]
            actions = convert_action(actions)
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector
            # def fill_in_action(values):
            #     # Target indices corresponding to your state space
            #     # In this example: 6 joints + 1 gripper for each arm
            #     # print(f"state[11]: {values[11]}")
            #     # print(f"shape of values: {values.shape}")
            #     # UNI_STATE_INDICES = (
            #     UNI_STATE_INDICES = [
            #         # TCP pose indices (6 entries)
            #         STATE_VEC_IDX_MAPPING['eef_pos_x'],
            #         STATE_VEC_IDX_MAPPING['eef_pos_y'],
            #         STATE_VEC_IDX_MAPPING['eef_pos_z'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_0'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_1'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_2'],
            #         # Gripper state (1 entry)
            #         STATE_VEC_IDX_MAPPING['gripper_open']
            #     ]
            #     # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
            #     uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            #     # print(f"uni_vec before: {uni_vec[11]}")
            #     # print(f"uni_vec:{uni_vec.shape}")
            #     uni_vec[..., UNI_STATE_INDICES] = values
            #     # print(f"uni_vec after: {uni_vec[11]}")
            #     return uni_vec
            
            # def fill_in_state(values):
            #     # print(f"state[0]: {values[0]}")
            #     # Target indices corresponding to your state space
            #     # In this example: 6 joints + 1 gripper for each arm
            #     UNI_STATE_INDICES = [
            #         # TCP pose indices (6 entries)
            #         STATE_VEC_IDX_MAPPING['eef_pos_x'],
            #         STATE_VEC_IDX_MAPPING['eef_pos_y'],
            #         STATE_VEC_IDX_MAPPING['eef_pos_z'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_0'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_1'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_2'],
            #         # Joint positions (6 entries)
            #         STATE_VEC_IDX_MAPPING['arm_joint_0_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_1_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_2_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_3_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_4_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_5_pos'],
            #         # Gripper state (1 entry)
            #         STATE_VEC_IDX_MAPPING['gripper_open']
            #     ]
            #     # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
            #     # print(f"shape of values: {values.shape}")
            #     uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            #     # print(f"uni_vec before: {uni_vec[0]}")
            #     # print(f"uni_vec:{uni_vec.shape}")
            #     uni_vec[..., UNI_STATE_INDICES] = values
            #     # print(f"uni_vec after: {uni_vec[0]}")
            #     return uni_vec
            def fill_in_state(values):
                # print(f"state[0]: {values[0]}")
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = (
                    # [STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)]
                    [STATE_VEC_IDX_MAPPING["gripper_open"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_x"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_y"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_z"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_0"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_1"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_2"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_3"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_4"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_5"]]
                )
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                # print(f"uni_vec before: {uni_vec[0]}")
                uni_vec[..., UNI_STATE_INDICES] = values
                # print(f"uni_vec after: {uni_vec[0]}")
                return uni_vec
            
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_state(actions)
            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    img = f[key][i]

                # Check if img is raw pixel data
                if img.ndim == 3 and img.shape[-1] == 3 and img.dtype == np.uint8:
                    decoded_img = img
                else:
                    # If not raw pixel, treat as encoded
                    encoded_data = np.frombuffer(img.tobytes(), np.uint8)
                    decoded_img = cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)
                if decoded_img is None:
                    raise ValueError("Failed to decode the image. Ensure data is a valid encoded image.")

                imgs.append(decoded_img)
                imgs = np.stack(imgs)
                # Pad if needed
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    imgs = np.concatenate([
                            np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                        imgs
                     ], axis=0)

                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_img("color_image")
            # print(cam_high)
            # print(f"cam_high shape: {cam_high.shape}")
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            # print(f"valid_len: {valid_len}")
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            # print(f"cam_high_mask: {cam_high_mask}, shape: {cam_high_mask.shape}")

            cam_right_wrist = parse_img("color_image")
            # print(f"cam_right_wrist shape: {cam_right_wrist.shape}")
            cam_right_wrist_mask = cam_high_mask.copy()
            # print(f"cam_right_wrist_mask: {cam_right_wrist_mask}, shape: {cam_right_wrist.shape}")
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            # print(f"cam_left_wrist shape: {cam_left_wrist.shape}")
            # print(f"cam_left_wrist_mask shape: {cam_left_wrist_mask.shape}")
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['robot_state'][:]
            actions = f['action'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            # Rescale gripper to [0, 1]
            # qpos = qpos / np.array(
            #    [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
            # )
            # target_qpos = f['action'][:] / np.array(
            #    [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
            # )
            
            # Parse the state and action
            state = qpos[first_idx-1:]
            actions = actions[first_idx-1:]
            state = convert_robot_obs(state)
            actions = convert_action(actions)
            
            # Fill the state/action into the unified vector
            def fill_in_action(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                # print(f"state[11]: {values[11]}")
                # print(f"shape of values: {values.shape}")
                # UNI_STATE_INDICES = (
                UNI_STATE_INDICES = [
                    # TCP pose indices (6 entries)
                    STATE_VEC_IDX_MAPPING['eef_pos_x'],
                    STATE_VEC_IDX_MAPPING['eef_pos_y'],
                    STATE_VEC_IDX_MAPPING['eef_pos_z'],
                    STATE_VEC_IDX_MAPPING['eef_angle_0'],
                    STATE_VEC_IDX_MAPPING['eef_angle_1'],
                    STATE_VEC_IDX_MAPPING['eef_angle_2'],
                    # Gripper state (1 entry)
                    STATE_VEC_IDX_MAPPING['gripper_open']
                ]
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                # print(f"uni_vec before: {uni_vec[11]}")
                # print(f"uni_vec:{uni_vec.shape}")
                uni_vec[..., UNI_STATE_INDICES] = values
                # print(f"uni_vec after: {uni_vec[11]}")
                return uni_vec
            
            # def fill_in_state(values):
            #     # print(f"state[0]: {values[0]}")
            #     # Target indices corresponding to your state space
            #     # In this example: 6 joints + 1 gripper for each arm
            #     UNI_STATE_INDICES = [
            #         # TCP pose indices (6 entries)
            #         STATE_VEC_IDX_MAPPING['eef_pos_x'],
            #         STATE_VEC_IDX_MAPPING['eef_pos_y'],
            #         STATE_VEC_IDX_MAPPING['eef_pos_z'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_0'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_1'],
            #         STATE_VEC_IDX_MAPPING['eef_angle_2'],
            #         # Joint positions (6 entries)
            #         STATE_VEC_IDX_MAPPING['arm_joint_0_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_1_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_2_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_3_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_4_pos'],
            #         STATE_VEC_IDX_MAPPING['arm_joint_5_pos'],
            #         # Gripper state (1 entry)
            #         STATE_VEC_IDX_MAPPING['gripper_open']
            #     ]
            #     # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
            #     # print(f"shape of values: {values.shape}")
            #     uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            #     # print(f"uni_vec before: {uni_vec[0]}")
            #     # print(f"uni_vec:{uni_vec.shape}")
            #     uni_vec[..., UNI_STATE_INDICES] = values
            #     # print(f"uni_vec after: {uni_vec[0]}")
            #     return uni_vec
            def fill_in_state(values):
                # print(f"state[0]: {values[0]}")
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = (
                    # [STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)]
                    [STATE_VEC_IDX_MAPPING["gripper_open"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_x"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_y"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_z"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_0"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_1"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_2"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_3"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_4"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_5"]]
                )
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                # print(f"uni_vec before: {uni_vec[0]}")
                uni_vec[..., UNI_STATE_INDICES] = values
                # print(f"uni_vec after: {uni_vec[0]}")
                return uni_vec
            state = fill_in_state(state)
            actions = fill_in_state(actions)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": actions
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    print(f"length of hdf5 ur5 dataset: {len(ds)}")
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
