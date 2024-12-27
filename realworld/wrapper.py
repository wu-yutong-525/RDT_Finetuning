import numpy as np
import cv2
from collections import deque
from .realenv import UR5RealEnv


class DofWrapper(UR5RealEnv):
    """A wrapper to change dof of an environment.
    """
    def __init__(self, env: UR5RealEnv, dof_setting):
        """Initialize a DofWrapper instance.

        Args:
            env (UR5RealEnv): environment to be wrapped
            dof_setting (array-like): specify which dof to be enabled and disabled.
                `None` stand for enable, `number` for disable. For example, for a 6-dof env, 
                dof_setting=[None, None, None, 1, 1, 1.5] can fix last three dof.
        """
        self._env = env

        orig_dtype = env.action_space['dtype']
        orig_minimum = env.action_space['low']
        orig_maximum = env.action_space['high']

        dof_setting = np.asarray(dof_setting)
        free_action_idxs = [i for i, val in enumerate(dof_setting) if val is None]

        def transform(action):
            new_action = dof_setting.copy()
            new_action[free_action_idxs] = action
            return new_action.astype(orig_dtype)
        
        self._transform = transform

        self._action_space = {
            'shape': np.array(free_action_idxs).shape,
            'dtype': orig_dtype,
            'low': orig_minimum[free_action_idxs],
            'high': orig_maximum[free_action_idxs]
        }

    def reset(self):
        return self._env.reset()

    def act(self, action):
        self._env.act(self._transform(action))

    def stop(self):
        self._env.stop()

    def emergency_stop(self):
        self._env.emergency_stop()

    def reset(self, qpos=None):
        return self._env.reset(qpos)

    def get_obs(self):
        return self._env.get_obs()

    @property
    def action_space(self):
        return self._action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionWrapper(UR5RealEnv):
    """Rescale simulation action to realworld action.
    """
    def __init__(self, real_env: UR5RealEnv, minimum, maximum, sim_env=None):
        """Initializes a new action scale wrapper.

        Args:
            real_env (UR5RealEnv): real environment to wrap.
            minimum (array-like): lower bound of input action.
            maximum (array-like): upper bound of input action.
            sim_env (dm_env.Environment): simulation environment
              corresponding to realworld environment.
        """
        self._env = real_env
        orig_shape = real_env.action_space['shape']
        orig_dtype = real_env.action_space['dtype']
        orig_minimum = real_env.action_space['low']
        orig_maximum = real_env.action_space['high']

        if isinstance(minimum, (int, float)):
            minimum = np.ones(orig_shape) * minimum
        else:
            minimum = np.asarray(minimum)
        if isinstance(maximum, (int, float)):
            maximum = np.ones(orig_shape) * maximum
        else:
            maximum = np.asarray(maximum)
        
        scale = (orig_maximum - orig_minimum) / (maximum - minimum)
        base_action = None if sim_env is None else np.asarray(sim_env.physics.init_qpos)

        def transform(action):
            new_action = orig_minimum + scale * (action - minimum)
            if base_action:
                new_action = new_action - base_action + real_env.init_qpos
            return new_action.astype(orig_dtype, copy=False)
        
        self._transform = transform

        self._action_space = {
            'shape': orig_shape,
            'dtype': orig_dtype,
            'low': minimum,
            'high': maximum
        }

    def act(self, action):
        self._env.act(self._transform(action))

    def stop(self):
        self._env.stop()

    def emergency_stop(self):
        self._env.emergency_stop()

    def reset(self, qpos=None):
        return self._env.reset(qpos)

    def get_obs(self):
        return self._env.get_obs()

    @property
    def action_space(self):
        return self._action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

class BinaryGripperWrapper(UR5RealEnv):
    """Convert a continuous control gripper into a binary gripper.
    """
    def __init__(self, real_env: UR5RealEnv):
        self._env = real_env
        orig_dtype = real_env.action_space['dtype']
        orig_gripper_minimum = real_env.action_space['low'][6:]
        orig_gripper_maximum = real_env.action_space['high'][6:]
        thre = (orig_gripper_minimum + orig_gripper_maximum) / 2

        def transform(action):
            arm_action = action[:6]
            gripper_action = action[6:]
            gripper_action[gripper_action < thre] = orig_gripper_minimum[gripper_action < thre]
            gripper_action[gripper_action >= thre] = orig_gripper_maximum[gripper_action >= thre]
            new_action = np.concatenate([arm_action, gripper_action])

            return new_action.astype(orig_dtype, copy=False)
        
        self._transform = transform

    def act(self, action):
        self._env.act(self._transform(action))

    def stop(self):
        self._env.stop()

    def emergency_stop(self):
        self._env.emergency_stop()

    def reset(self, qpos=None):
        return self._env.reset(qpos)

    def get_obs(self):
        return self._env.get_obs()

    @property
    def action_space(self):
        return self._env.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

class FrameResizeWrapper(UR5RealEnv):
    """Resize observation frame to a specific size.
    """
    def __init__(self, real_env: UR5RealEnv, width: int, height: int, rot: int=0):
        """Initialize a frame resize wrapper.

        Args:
            real_env (UR5RealEnv): real environment to wrap.
            width (int): target frame width
            height (int): target frame height
            rot (int): see param `k` in np.rot90
        """
        self._env = real_env

        def transform(img):
            orig_h, orig_w, _ = img.shape
            orig_ratio = orig_w / orig_h
            target_ratio = width / height
            if orig_ratio > target_ratio:
                new_w = int(orig_h * target_ratio)
                start = (orig_w - new_w) // 2
                end = start + new_w
                crop_img = img[:, start:end, :]
            else:
                new_h = int(orig_w / target_ratio)
                start = (orig_h - new_h) // 2
                end = start + new_h
                crop_img = img[start:end, :, :]
            
            crop_img = cv2.resize(crop_img, (height, width))
            crop_img = np.rot90(crop_img, rot, axes=(0, 1))

            if len(crop_img.shape) == 2:
                crop_img = crop_img[..., None]

            return crop_img.transpose(2, 0, 1)
        
        self._transform = transform
    
    def get_obs(self):
        """Return frame of shape (C, H, W)
        """
        obs = self._env.get_obs()
        obs['trans'] = self._transform(obs['trans'])
        obs['trans_depth'] = self._transform(obs['depth'])
        return obs
    
    def act(self, action):
        self._env.act(action)

    def stop(self):
        self._env.stop()

    def emergency_stop(self):
        self._env.emergency_stop()

    def reset(self, qpos=None):
        return self._env.reset(qpos)

    def __getattr__(self, name):
        return getattr(self._env, name)

class FrameStackWrapper(UR5RealEnv):
    def __init__(self, real_env: UR5RealEnv, num_frames):
        self._env = real_env
        self._num_frames = num_frames
        self._frame_buffer = deque([], maxlen=num_frames)

    def get_obs(self):
        obs = self._env.get_obs()
        self._frame_buffer.append(obs['trans'])
        # fill the frame buffer for first step of a new episode
        while len(self._frame_buffer) < self._num_frames:
            self._frame_buffer.append(obs['trans'])
        obs['trans'] = np.concatenate(list(self._frame_buffer), axis=0)
        # import ipdb; ipdb.set_trace()
        obs['input'] = np.concatenate([obs['trans'], obs['trans_depth']], axis=0)
        # obs['input'] = obs['trans']
        return obs

    def act(self, action):
        self._env.act(action)

    def stop(self):
        self._env.stop()

    def emergency_stop(self):
        self._env.emergency_stop()

    def reset(self, qpos=None):
        return self._env.reset(qpos)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
class DepthStackWrapper(UR5RealEnv):
    def __init__(self, real_env: UR5RealEnv, num_frames):
        self._env = real_env
        self._num_frames = num_frames
        self._frame_buffer = deque([], maxlen=num_frames)

    def get_obs(self):
        obs = self._env.get_obs()
        self._frame_buffer.append(obs['trans_depth'])
        # fill the frame buffer for first step of a new episode
        while len(self._frame_buffer) < self._num_frames:
            self._frame_buffer.append(obs['trans_depth'])
        obs['trans_depth'] = np.concatenate(list(self._frame_buffer), axis=0)
        # import ipdb; ipdb.set_trace()
        obs['input'] = np.concatenate([obs['trans'], obs['trans_depth']], axis=0)
        # obs['input'] = obs['trans']
        return obs

    def act(self, action):
        self._env.act(action)

    def stop(self):
        self._env.stop()

    def emergency_stop(self):
        self._env.emergency_stop()

    def reset(self, qpos=None):
        return self._env.reset(qpos)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
class PropStackWrapper(UR5RealEnv):
    def __init__(self, real_env: UR5RealEnv, num_frames):
        self._env = real_env
        self._num_frames = num_frames
        self._frame_buffer = deque([], maxlen=num_frames)

    def get_obs(self):
        obs = self._env.get_obs()
        prop = np.asarray(self._env.cur_qpos, dtype=np.float32)
        prop[0] -= 0.75 * np.pi
        self._frame_buffer.append(prop)
        # fill the frame buffer for first step of a new episode
        while len(self._frame_buffer) < self._num_frames:
            self._frame_buffer.append(prop)
        stacked_prop = np.concatenate(list(self._frame_buffer), axis=0)
        obs['prop'] = stacked_prop
        return obs

    def act(self, action):
        self._env.act(action)

    def stop(self):
        self._env.stop()

    def emergency_stop(self):
        self._env.emergency_stop()

    def reset(self, qpos=None):
        return self._env.reset(qpos)

    def __getattr__(self, name):
        return getattr(self._env, name)