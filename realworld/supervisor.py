from pynput import keyboard

class Supervisor(object):
    def __init__(self, env):
        self.listener = keyboard.Listener(
            lambda key: Supervisor._on_press(key, env))

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    @staticmethod
    def _on_press(key, env):
        try:
            key = key.char
        except:
            key = key
        finally:
            if key == keyboard.Key.esc:
                print("ESC Pressed: Supervisor stop environment.")
                env.emergency_stop()
            elif key == 'r':
                print("R Pressed: Supervisor restart environment.")
                env.reset()
            elif key == 'g':
                print("G Pressed: Grasp!")
                env.grasp()
            elif key == 'l':
                print("L Pressed: Gripper locked.")
                env.lock_gripper()
            elif key == 'o':
                print("O Pressed: Gripper opened.")
                env.open_gripper()
        