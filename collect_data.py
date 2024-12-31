import os
import time
import numpy as np
from pynput import keyboard
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_gripper import RobotiqGripper
from utils import vec2euler, vec2mat
import cv2
import h5py
from realsense import RealSense  # Assumes RealSense class for image capture is available


class UR5DataCollector:
    def __init__(self, ip, data_dir, task_name, instruction, acc=0.5, vel=0.5, frequency=125, use_camera=True):
        self.rtde_c = RTDEControlInterface(ip)
        self.rtde_r = RTDEReceiveInterface(ip)
        self.gripper = RobotiqGripper()
        self.gripper.connect(ip, 63352)
        self.gripper.activate()
        self.gripper_state = 0
        self.acc = acc
        self.vel = vel
        self.dt = 1. / frequency
        self.lookahead_time = 0.1
        self.gain = 500

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.task_name = task_name
        self.instruction = instruction
        self.trajectory_id = self._get_next_trajectory_id()
        self.current_trajectory = []
        self.trajectory_count = 0

        self.use_camera = use_camera
        if use_camera:
            self.camera = RealSense(width=640, height=480)
            self.camera.start()

        self.init_pose = [0.75 * np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self.listener = keyboard.Listener(on_press=self._on_press)

    def _get_next_trajectory_id(self):
        """
        Automatically determines the next trajectory ID based on existing files.
        Resets ID if task name changes.
        """
        existing_files = [f for f in os.listdir(self.data_dir) if f.startswith(f"trajectory_{self.task_name}_")]
        if not existing_files:
            return 1  # Start from ID 1 if no files exist
        ids = [
            int(f.split(f"trajectory_{self.task_name}_")[1].split(".")[0])
            for f in existing_files
        ]
        return max(ids) + 1

    def _record_data(self):
        """
        Records the current robot state, action, and observation into a trajectory list.
        """
        tcp_pose = self.rtde_r.getActualTCPPose()
        joint_positions = self.rtde_r.getActualQ()
        gripper_state = self.gripper_state
        timestamp = time.time()

        robot_state = np.array(tcp_pose + joint_positions + [gripper_state], dtype=np.float32)
        action = np.array(tcp_pose + [gripper_state], dtype=np.float32)  # Assuming actions are TCP positions; adjust as needed.

        if self.use_camera:
            frame = self.camera.get_frame()
            color_img = frame['color']
            depth_img = frame['depth']

            if hasattr(self, 'prev_color_image') and np.array_equal(self.prev_color_image, color_img):
                print(f"Warning: Repeated color image detected at timestamp {timestamp}")
                if np.array_equal(self.prev_color_image, color_img):
                    print("Restarting camera...")
                    self.camera.stop()
                    self.camera.start()
            self.prev_color_image = color_img.copy()  # Save the current image for the next comparison

        else:
            color_img = None
            depth_img = None

        self.current_trajectory.append({
            "timestamp": timestamp,
            "robot_state": robot_state,
            "action": action,
            "color_image": color_img,
            "depth_image": depth_img
        })
        print(f"Data recorded at timestamp {timestamp}")
        return color_img
    # def _record_data(self):
    #     """
    #     Records the current robot state, action (including gripper action), and observation into an HDF5 file.
    #     """
    #     # Get robot state and action
    #     tcp_pose = self.rtde_r.getActualTCPPose()
    #     joint_positions = self.rtde_r.getActualQ()
    #     gripper_state = self.gripper_state
    #     timestamp = time.time()

    #     # Combine TCP pose and gripper state into the action
    #     action = np.array(tcp_pose + [gripper_state], dtype=np.float32)  # Append gripper state to TCP pose
    #     robot_state = np.array(tcp_pose + joint_positions + [gripper_state], dtype=np.float32)

    #     # Get observations (camera data)
    #     if self.use_camera:
    #         frame = self.camera.get_frame()
    #         color_img = frame['color']
    #         depth_img = frame['depth']
    #     else:
    #         color_img = None
    #         depth_img = None

    #     # Define HDF5 file path
    #     hdf5_path = os.path.join(self.data_dir, f"trajectory_{self.task_name}_{self.trajectory_id}.hdf5")
    #     with h5py.File(hdf5_path, "a") as f:
    #         # Create datasets if they don't exist
    #         if "robot_state" not in f:
    #             f.create_dataset("robot_state", (0, len(robot_state)), maxshape=(None, len(robot_state)), dtype=np.float32)
    #         if "action" not in f:
    #             f.create_dataset("action", (0, len(action)), maxshape=(None, len(action)), dtype=np.float32)
    #         if "color_image" not in f and color_img is not None:
    #             f.create_dataset("color_image", (0, *color_img.shape), maxshape=(None, *color_img.shape), dtype=np.uint8)
    #         if "depth_image" not in f and depth_img is not None:
    #             f.create_dataset("depth_image", (0, *depth_img.shape), maxshape=(None, *depth_img.shape), dtype=np.float32)
    #         if "instruction" not in f:
    #             f.create_dataset("instruction", data=self.task_name)

    #         # Append data
    #         f["robot_state"].resize((f["robot_state"].shape[0] + 1), axis=0)
    #         f["robot_state"][-1] = robot_state

    #         f["action"].resize((f["action"].shape[0] + 1), axis=0)
    #         f["action"][-1] = action

    #         if self.use_camera:
    #             f["color_image"].resize((f["color_image"].shape[0] + 1), axis=0)
    #             f["color_image"][-1] = color_img

    #             f["depth_image"].resize((f["depth_image"].shape[0] + 1), axis=0)
    #             f["depth_image"][-1] = depth_img

    #     print(f"Data recorded at timestamp {timestamp}")

    def _save_trajectory(self):
        """
        Saves the current trajectory to an HDF5 file.
        """
        hdf5_path = os.path.join(self.data_dir, f"trajectory_{self.task_name}_{self.trajectory_id}.hdf5")
        with h5py.File(hdf5_path, "w") as f:
            f.attrs["instruction"] = self.instruction

            robot_states = [entry["robot_state"] for entry in self.current_trajectory]
            actions = [entry["action"] for entry in self.current_trajectory]
            color_images = [entry["color_image"] for entry in self.current_trajectory if entry["color_image"] is not None]
            depth_images = [entry["depth_image"] for entry in self.current_trajectory if entry["depth_image"] is not None]

            f.create_dataset("robot_state", data=np.array(robot_states, dtype=np.float32))
            f.create_dataset("action", data=np.array(actions, dtype=np.float32))

            if color_images:
                f.create_dataset("color_image", data=np.array(color_images, dtype=np.uint8))
            if depth_images:
                f.create_dataset("depth_image", data=np.array(depth_images, dtype=np.float32))

        print(f"Trajectory saved to {hdf5_path}")
        self.current_trajectory = []
        self.trajectory_id += 1
        self.trajectory_count += 1

    def _translate(self, vec):
        tcp = self.rtde_r.getActualTCPPose()
        t_start = self.rtde_c.initPeriod()
        tcp[:3] = [a + b for a, b in zip(tcp[:3], vec)]
        self.rtde_c.servoL(tcp, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)
        self.rtde_c.waitPeriod(t_start)
        self._record_data()

    def _on_press(self, key):
        try:
            key = key.char
        except AttributeError:
            key = key

        if key == 'w':
            self._translate((0, 0, 0.01))
        elif key == 's':
            self._translate((0, 0, -0.01))
        elif key == 'a':
            self._translate((0, 0.01, 0))
        elif key == 'd':
            self._translate((0, -0.01, 0))
        elif key == 'q':
            self._translate((0.01, 0, 0))
        elif key == 'e':
            self._translate((-0.01, 0, 0))
        elif key == 'g':
            if self.gripper_state == 0:
                self.gripper.move(255, 255, 0)
                self.gripper_state = 1
            else:
                self.gripper.move(0, 255, 0)
                self.gripper_state = 0
            self._record_data()
        elif key == 'r':
            self.rtde_c.servoStop()
            self.rtde_c.moveJ(self.init_pose)
            self._record_data()
        elif key == keyboard.Key.enter:
            self._save_trajectory()
            if self.trajectory_count % 50 == 0:
                action = input("Press '0' to stop collecting or 'n' to start a new task: ")
                if action == '0':
                    self.close()
                elif action == 'n':
                    new_task_name = input("Enter new task name: ")
                    new_instruction = input("Enter new task instruction: ")
                    self.task_name = new_task_name
                    self.instruction = new_instruction
                    self.trajectory_id = self._get_next_trajectory_id()

    def run(self):
        """
        Starts the keyboard control loop.
        """
        print(f"Starting data collection for task: {self.task_name}")
        self.listener.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.close()

    def close(self):
        self.listener.stop()
        if self.use_camera:
            self.camera.stop()
        self.rtde_c.stopScript()


if __name__ == "__main__":
    robot_ip = "192.169.0.10"
    data_dir = "./collected_data"
    task_name = input("Enter task name: ")
    instruction = input("Enter task instruction: ")

    collector = UR5DataCollector(robot_ip, data_dir, task_name, instruction, use_camera=True)

    try:
        collector.run()
    finally:
        collector.close()