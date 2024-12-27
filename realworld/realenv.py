from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from .realsense import RealSense
from .robotiq_gripper import RobotiqGripper
from .supervisor import Supervisor
import numpy as np
import cv2
import enum
from math import pi
from multiprocessing.managers import SharedMemoryManager
from .controller import UR5Controller
from .command import Command
from .utils import vec2euler, euler2vec

class UR5RealEnv:
    """Single process real world environment
    """
    class ControlMode(enum.Enum):
        JOINT = 0
        TCP = 1

    def __init__(
        self, 
        robot_ip,
        init_qpos=[0.75*pi, -pi/2, pi/2, -pi/2, -pi/2, 0, 0], 
        multi_process=False,
        delta_action=False,
        control_mode=ControlMode.JOINT,
        dt=0.08,
        use_camera=True,
        moving_average=1,
    ):
        if use_camera:
            # camera
            self.camera = RealSense(width=640, height=480)
            # self.camera = RealSense(width=1280, height=720)
            self.camera.start()

        self._multi_process = multi_process
        self._rtde_r = RTDEReceiveInterface(robot_ip)

        if multi_process:
            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()
            self._controller = UR5Controller(self.shm_manager, robot_ip)
            self._controller.start()
        else:
            # robot
            self._rtde_c = RTDEControlInterface(robot_ip)
            # gripper
            self._gripper = RobotiqGripper()
            self._gripper.connect(robot_ip, 63352)
            self._gripper.activate()

        # supervisor, for safety
        self.supervisor = Supervisor(self)
        self.supervisor.start()

        self._init_qpos = init_qpos
        self._delta_action = delta_action
        self._contorl_mode = control_mode
        self._dt = dt
        self._use_camera = use_camera
        self._moving_average = moving_average
        self._prev_gripper_action = np.zeros(1)

        self._lock_gripper = False
        self._open_gripper = False

        self._cur_qpos = None

    def stop(self):
        if self._multi_process:
            self._controller.send_action({
                'cmd': Command.STOP.value
            })
            self.shm_manager.shutdown()
        else:
            self._rtde_c.stopScript()
        self.supervisor.stop()

    def emergency_stop(self):
        # TODO: rethink start() & stop()
        if self._multi_process:
            self._controller.send_action({
                'cmd': Command.EMERGENCY_STOP.value
            })
        else:
            self._rtde_c.stopScript()

    def grasp(self):
        if self._multi_process:
            self._controller.send_action({
                'cmd': Command.GRASP.value,
            })
        else:
            pass
    
    def lock_gripper(self):
        self._lock_gripper = not self._lock_gripper
        self._open_gripper = False
    
    def open_gripper(self):
        self._open_gripper = not self._open_gripper
        self._lock_gripper = False

    def reset(self, qpos=None):
        print("Resetting robot to initial pose!")
        qpos = qpos or self._init_qpos
        self._cur_qpos = np.asarray(qpos[:6])
        if self._multi_process:
            self._controller.send_action({
                'cmd': Command.MOVEJ.value,
                'action': qpos
            })
        else:
            self._rtde_c.servoStop()
            self._rtde_c.moveJ(qpos[:-1])
            self._gripper.move(qpos[-1], 255, 0)

    def get_obs(self):
        if self._use_camera:
            frame = self.camera.get_frame()
            color = frame['color'].copy()
            return {
                'origin': color,
                'trans': cv2.cvtColor(color, cv2.COLOR_BGR2RGB),
                'depth': frame['depth']
            }
        else:
            return {}

    def act(self, action):
        """An act function, which will block until the robot action is completed.
        """
        action = np.asarray(action)
        
        if self._lock_gripper:
            action[-1] = 255
        if self._open_gripper:
            action[-1] = 0

        action[-1] = self._moving_average * action[-1] + (1 - self._moving_average) * self._prev_gripper_action
        self._prev_gripper_action = action[-1].copy()
        
        if self._contorl_mode == UR5RealEnv.ControlMode.JOINT:
            if self._delta_action:
                self._cur_qpos += action[:6] * 0.025
                # self._cur_qpos = self._rtde_r.getActualQ() + action[:6] * 0.025
            else:
                self._cur_qpos = action
            if self._multi_process:
                self._controller.send_action({
                    'cmd': Command.SERVOJ.value,
                    'action': np.concatenate((self._cur_qpos, action[-1:]))
                })
            else:
                # FIXME
                q = self._rtde_r.getActualQ()
                max_diff = np.max(np.abs(action[:-1] - q))
                max_crop = 0.01
                if max_diff > max_crop:
                    action[:-1] = q + (action[:-1] - q) / max_diff * max_crop
                self._rtde_c.servoJ(action[:-1], 0, 0, self._dt, 0.08, 300)
                self._gripper.move(action[-1], 255, 0)
        
        elif self._contorl_mode == UR5RealEnv.ControlMode.TCP:
            tcp = self._rtde_r.getActualTCPPose()
            cur_pos = tcp[:3]
            cur_euler = vec2euler(tcp[3:6])

            if self._delta_action:
                action[:3] += cur_pos
                action[3:6] += cur_euler
            else:
                delta_pos = action[:3] - tcp[:3]
                delta_euler = action[3:6] - cur_euler
                
                action[:3] = cur_pos + delta_pos
                action[3:6] = euler2vec(cur_euler + delta_euler)

            if self._multi_process:
                self._controller.send_action({
                    'cmd': Command.SERVOL.value,
                    'action': action
                })
            else:
                raise NotImplementedError

    @property
    def cur_qpos(self):
        return self._cur_qpos

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def action_space(self):
        if self._delta_action:
            return {
                'shape': (7,),
                'dtype': np.float32,
                'low': np.append(-np.ones(6), 0),
                'high': np.append(np.ones(6), 255)
            }
        else:
            return {
                'shape': (7,),
                'dtype': np.float32,
                'low': np.array([-2*pi, -2*pi, -pi, -2*pi, -2*pi, -2*pi, 0]),
                'high': np.array([2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi, 255])
            }


if __name__ == '__main__':
    env = UR5RealEnv(
        robot_ip='192.169.0.10',
        dt=0.08,
        multi_process=True,
        control_mode=UR5RealEnv.ControlMode.JOINT,
        delta_action=True
    )
