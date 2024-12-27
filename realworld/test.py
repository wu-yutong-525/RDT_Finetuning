from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from pynput import keyboard
import time
from math import pi
import numpy as np
import copy
import cv2
import pickle

from realenv import UR5RealEnv
from utils import precise_sleep, precise_wait

class Service(object):
    def __init__(self, ip, acc=0.5, vel=0.5, frequency=125):
        self.env = UR5RealEnv(robot_ip='192.169.0.10')
        self.env.start()
        self.rtde_c = self.env.rtde_c
        self.rtde_r = self.env.rtde_r
        # self.rtde_c = RTDEControlInterface(ip)
        # self.rtde_r = RTDEReceiveInterface(ip)
        self.acc = acc
        self.vel = vel
        self.dt = 1. / frequency
        self.lookahead_time = 0.1
        self.gain = 500
        # defined in joint space
        # self.init_pose = [0, -pi/2, -pi/2, -pi/2, pi/2, 0]
        self.init_pose = [1.5, -pi/2, -pi/2, -pi/2, pi/2, 0]
        self.listener = keyboard.Listener(on_press=lambda key: Service._on_press(self, key))

    def loop(self):
        print("Starting loop...")
        st_time = time.time()
        self.listener.start()
        while True:
            if time.time() - st_time > 1:
                st_time = time.time()
                p_l = self.rtde_r.getActualTCPPose()
                p_j = self.rtde_r.getActualQ()
                # print("actual Q: ", self.rtde_r.getActualQ())
                # print("===================rc=========")
                # print("Current tool pose:", p_l)
                # print("Current joint pos:", p_j)
                # print("============================")

    def close(self):
        self.env.stop()
        self.listener.stop()
        # self.rtde_c.stopScript()

    def _translate(self, vec):
        tcp = self.rtde_r.getActualTCPPose()
        # t_start = self.rtde_c.initPeriod()
        tcp [:3] = [a + b for (a, b) in zip(tcp, vec)]
        self.rtde_c.servoL(tcp, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)
        # self.rtde_c.servoL(tcp, self.vel, self.acc, 2, self.lookahead_time, self.gain)
        # self.rtde_c.waitPeriod(t_start)

    def _on_press(self, key):
        try:
            key = key.char
        except:
            key = key
        finally:
            if key == 'w':
                print('\nW pressed (Up)')
                self._translate((0, 0, 0.01))
            elif key == 's':
                print('\nS pressed (Down)')
                self._translate((0, 0, -0.01))
            elif key == 'a':
                print('\nA pressed (Left)')
                # self._translate((0, 0.01, 0))

                # self.rtde_c.speedJ([0.1, 0, 0, 0, 0, 0], 1.2, 0.008)

                p = self.rtde_r.getActualQ()
                p = np.append(p, 0)
                p[0] += 0.2
                
                self.env.act(p)
            elif key == 'd':
                print('\nD pressed (Right)')
                # self._translate((0, -0.01, 0))

                # self.rtde_c.speedJ([-0.1, 0, 0, 0, 0, 0], 1.2, 0.008)

                p = self.rtde_r.getActualQ()
                p = np.append(p, 0)
                p[0] -= 0.2

                self.env.act(p)
            elif key == 'q':
                print('\nQ pressed (Front)')
                self._translate((0.01, 0, 0))
            elif key == 'e':
                print('\nE pressed (Back)')
                self._translate((-0.01, 0, 0))
            elif key == 'z':
                print('\nZ pressed (Gripper open)')
                pos = self.env.gripper.get_current_position()
                pos -= 5
                self.env.gripper.move(pos, 100, 0)
            elif key == 'x':
                print('\nX pressed (Gripper close)')
                pos = self.env.gripper.get_current_position()
                pos += 5
                self.env.gripper.move(pos, 100, 0)
            # elif key == 'r':
            #     print('\nR pressed (Reset)')
            #     self.rtde_c.servoStop()
            #     self.rtde_c.moveJ(self.init_pose)
            elif key == '1':
                print('\n1 pressed (Test single)')
                # 2s
                tcp = self.rtde_r.getActualTCPPose()
                st = time.time()
                for _ in range(125):
                    t_start = self.rtde_c.initPeriod()
                    tcp[2] += 0.001
                    self.rtde_c.servoL(tcp, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)
                    self.rtde_c.waitPeriod(t_start)
                print(time.time() - st)  # about 1.61s
                self.rtde_c.servoStop()
            elif key == '2':
                print('\n2 pressed (Imitate environment)')
                self.rtde_c.servoStop()
                q = self.rtde_r.getActualQ()
                rec_q = copy.deepcopy(q)
                frequency = 10
                dt = 1. / frequency
                for _ in range(20):
                    # get observation
                    time.sleep(0.005)
                    # inference
                    time.sleep(0.01)
                    q[2] += 0.02  # output of policy
                    # act
                    st_time = time.monotonic()
                    self.rtde_c.moveJ(q, 1.0, 0.5, True)
                    # self.rtde_c.servoJ(q, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)
                    precise_sleep(0.088)
                    # q = self.rtde_r.getActualQ()
                    # self.rtde_c.servoJ(q, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)
                    # self.rtde_c.servoStop()
                    print("before stop", time.monotonic() - st_time)
                    self.rtde_c.stopJ(0.5, False)
                    # self.rtde_c.moveJ(q, 1.0, 1.4, True)
                    # self.rtde_c.stopJ(1.4, True)
                    print("after stop:", time.monotonic() - st_time)
                print("*****", self.rtde_r.getActualQ()[2], self.rtde_r.getActualQ()[2] - rec_q[2])
            elif key == '3':
                print('\n3 pressed (Sim2real align test)')
                self.rtde_c.servoStop()

                action = self.rtde_r.getActualQ()
                action.append(255)
                for i in range(20):
                    t_start = time.time()
                    print(f"=====New cycle: {i}, time: {t_start:.2f}=====")
                    obs = self.env.step(action)
                    cv2.imshow("obs:color", obs['color'])
                    cv2.waitKey(1)
                    # simulate inference
                    precise_sleep(0.1, time_func=time.time)
                    action[0] += 0.01
                    t_end = time.time()
                    print(f"End of a cycle, time: {t_end:.2f}, duration: {t_end - t_start:.2f}")
                cv2.destroyAllWindows()
            elif key == 'p':
                print('\nP pressed (Calibrate)')
                self.rtde_c.servoStop()

                # def get_action(action):
                #     for _ in range(20):
                #         action = [a + 0.02 for a in action]
                #         yield action
                #     for _ in range(30):
                #         action = [a - 0.02 for a in action]
                #         yield action
                #     for _ in range(60):
                #         yield action
                #     for _ in range(10):
                #         action = [a + 0.2 for a in action]
                #         yield action
                #     for _ in range(10):
                #         action = [a - 0.2 for a in action]
                #         yield action
                #     for _ in range(60):
                #         yield action
                #     for _ in range(15):
                #         action = [a + 0.05 for a in action]
                #         yield action
                #     for _ in range(15):
                #         action = [a - 0.05 for a in action]
                #         yield action
                #     for _ in range(60):
                #         yield action
                
                # def get_action(st_action):
                #     with open('./actions.pickle', 'rb') as f:
                #         actions = pickle.load(f)
                #         for action in actions:
                #             yield action + st_action

                def get_action():
                    with open('./actions.pickle', 'rb') as f:
                        actions = pickle.load(f)
                        for action in actions:
                            yield action

                steps = []
                qpos = [[], [], [], [], [], []]
                delta_qpos = [[], [], [], [], [], []]
                delta_act = [[], [], [], [], [], []]
                
                st_action = self.rtde_r.getActualQ()
                st_action.append(0)
                for step, act in enumerate(get_action()):
                    # get obs -> inference -> act -> wait
                    st_time = time.monotonic()
                    print(f"=====New cycle: {step}, time: {st_time}=====")

                    # obs
                    # obs = self.env.get_obs()
                    print("Obs latency: ", time.monotonic() - st_time)
                    # simulate inference
                    precise_sleep(0.01, time_func=time.monotonic)
                    print("Inference latency: ", time.monotonic() - st_time)

                    steps.append(step)
                    q = self.rtde_r.getActualQ()
                    for i in range(len(qpos)):
                        qpos[i].append(q[i])
                        delta_qpos[i].append(q[i] - st_action[i])
                        # delta_act[i].append(act[i] - st_action[i])
                        delta_act[i].append(act[i])
                    
                    # act
                    self.env.act(act)
                    print("Action: ", act)
                    # cv2.imshow("obs:color", obs['color'])
                    # cv2.waitKey(1)
                    print("Act latency: ", time.monotonic() - st_time)

                    # wait
                    precise_wait(st_time + 0.08, time_func=time.monotonic)
                    
                    ed_time = time.monotonic()
                    print(f"End cycle: {step}, dur: {ed_time - st_time}=====")
                
                self.env.emergency_stop()
                
                with open('real_qpos.pickle', 'wb') as f:
                    pickle.dump(delta_qpos, f)
                with open('real_act.pickle', 'wb') as f:
                    pickle.dump(delta_act, f)

                def plot(xs, ys, str_x, str_y, title):
                    import matplotlib.pyplot as plt
                    _, ax = plt.subplots()
                    colors = ['red', 'green', 'blue', 'yellow', 'orange']
                    # for i in range(len(xs)):
                    ax.plot(xs, ys, color=colors[0])
                    ax.set_xlabel(str_x)
                    ax.set_ylabel(str_y)
                    ax.set_title(title)
                    # ax.legend()
                    plt.savefig(f'../assets/{title}.png')
                    # plt.show()

                
                for i in range(len(qpos)):
                    plot(steps, qpos[i], 'steps', f'real_joint{i}', f'real_joint{i}')


if __name__ == "__main__":

    service = Service(ip='192.169.0.10')
    try:
        service.loop()
    finally:
        service.close()