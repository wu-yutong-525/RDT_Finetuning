import numpy as np
from multiprocessing import Process
import threading
import queue
from .shared_memory.shared_queue import SharedMemoryQueue, Full, Empty
from .interpolator import interpolate
from .supervisor import Supervisor
from .command import Command
import time
from .utils import vec2euler, euler2vec

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from .robotiq_gripper import RobotiqGripper

class UR5Controller(Process):
    def __init__(self, shm_manager, robot_ip) -> None:
        super().__init__(daemon=True)

        self._queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={'cmd': Command.SERVOJ.value, 'action': np.zeros(7, dtype=np.float32)},
            buffer_size=256
        )
        self._robot_ip = robot_ip

    def send_action(self, action):
        self._queue.put(action)

    def emergency_stop(self):
        self._rtde_c.stopScript()

    def stop(self):
        self._rtde_c.stopScript()
        self.close()

    def run(self):
        print("Running controller...")
        # robot
        self._rtde_r = RTDEReceiveInterface(self._robot_ip)
        self._rtde_c = RTDEControlInterface(self._robot_ip)
        
        # gripper
        self._gripper = RobotiqGripper()
        self._gripper.connect(self._robot_ip, 63352)
        # self._gripper.activate()

        self._gripper_queue = queue.Queue(maxsize=256)
        self._gripper_thread = threading.Thread(target=self._gripper_control)
        self._gripper_thread.start()

        

        while True:
            try:
                # print("begin controller get")
                data = self._queue.get()
                
                if data['cmd'] == Command.SERVOJ.value:
                    st_time = time.monotonic()
                    print("begin controller", self._rtde_r.getActualQ())

                    action = data['action']
                    self._gripper_queue.put(action[-1])
                    print("gripper q:", self._gripper_queue.qsize())
                    # q = self._rtde_r.getActualQ()
                    # delta_action = action[:-1] - q
                    # action[:-1] = q + delta_action * 0.025
                    # print("in ctrl", action[1])
                    self._rtde_c.servoJ(action[:-1], 0, 0, 0.2, 0.1, 300)

                    # action = data['action']
                    # self._gripper_queue.put(action[-1])
                    # q = self._rtde_r.getActualQ()
                    # delta_action = action[:-1] - q
                    # action[:-1] = q + delta_action * 0.025
                    # interp_actions = np.linspace(q, action[:-1], num=25)
                    # for action in interp_actions:
                    #     t_start = self._rtde_c.initPeriod()
                    #     self._rtde_c.servoJ(action, 1.0, 1.0, 0.008, 0.1, 300)
                    #     self._rtde_c.waitPeriod(t_start)

                    print("end controller", time.monotonic() - st_time, self._rtde_r.getActualQ())

                elif data['cmd'] == Command.SERVOL.value:
                    tcp = self._rtde_r.getActualTCPPose()
                    cur_pos = tcp[:3]
                    cur_euler = vec2euler(tcp[3:6])
                    action = data['action']

                    delta_pos = action[:3] - tcp[:3]
                    delta_euler = action[3:6] - cur_euler
                    
                    target_pos = cur_pos + delta_pos * 0.005
                    target_euler = euler2vec(cur_euler + delta_euler * 0.005)
                    target_pose = np.concatenate([target_pos, target_euler])
                    target_q = self._rtde_r.getInverseKinematics(target_pose)

                    self._gripper_queue.put(action[6:])
                    self._rtde_c.servoJ(target_q, 0, 0, 0.08, 0.1, 300)
                elif data['cmd'] == Command.MOVEJ.value:
                    self._movej(data['action'])
                    self._gripper_queue.put(data['action'][6:])
                elif data['cmd'] == Command.EMERGENCY_STOP.value:
                    self._stop_script()
                elif data['cmd'] == Command.STOP.value:
                    self.stop()
                elif data['cmd'] == Command.GRASP.value:
                    self.grasp()
            except Empty:
                pass
            except Exception as e:
                print("Error in controller: ", e)

    def grasp(self):
        q = self._rtde_r.getActualQ()
        for _ in range(50):
            q += np.array([0, -0.003, 0, 0, 0, 0])
            self._gripper_queue.put(255)
            t_start = self._rtde_c.initPeriod()
            self._rtde_c.servoJ(q, 0, 0, 0.08, 0.1, 300)
            self._rtde_c.waitPeriod(t_start)

    def _gripper_control(self):
        while True:
            angle = int(self._gripper_queue.get())
            self._gripper.move(angle, 255, 0)

    def _servoj(self, qpos):
        t_start = self._rtde_c.initPeriod()
        self._rtde_c.servoJ(qpos, 0, 0, 0.008, 0.1, 300)
        self._rtde_c.waitPeriod(t_start)

    def _movej(self, qpos):
        self._rtde_c.servoStop()
        self._rtde_c.moveJ(qpos[:-1])
        q = self._rtde_r.getActualQ()
        self._rtde_c.servoJ(q, 0, 0, 0.1, 0.1, 300)

    def _stop_script(self):
        self._rtde_c.stopScript()

    def _rescale_action(self, action, action_scale):
        minimum = -1.0
        maximum = 1.0
        scale = 2.0 * action_scale * np.ones_like(action) / (maximum - minimum)
        return -action_scale + (action - minimum) * scale
    
    @property
    def rtde_c(self):
        return self._rtde_c