import pyrealsense2 as rs
import numpy as np


class RealSense(object):
    """Class for a single realsense camera.
    """
    def __init__(self, fps=30, width=640, height=480):
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, fps)
        self.pipeline = rs.pipeline()
        
    def start(self):
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def stop(self):
        self.pipeline.stop()

    def get_frame(self):
        """Return a BGR frame and corresponding timestamp. 
        """
        while True:
            frameset = self.pipeline.wait_for_frames()
            aligned_frameset = self.align.process(frameset)
            timestamp = aligned_frameset.get_timestamp() / 1000  # convert ms to s
            color_frame = aligned_frameset.get_color_frame()
            depth_frame = aligned_frameset.get_depth_frame()
            if not depth_frame or not color_frame:
                print(depth_frame, color_frame)
                continue
            else:
                break

        # TODO
            
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        clipping_distance_in_meters = 2
        clipping_distance = clipping_distance_in_meters / depth_scale

        depth = np.asanyarray(depth_frame.get_data())[..., None]

        depth[depth >= clipping_distance] = clipping_distance
        depth[depth <= 0] = clipping_distance

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype(np.uint8)

        return {
            'color': np.asanyarray(color_frame.get_data()),
            'depth': depth,
            'timestamp': timestamp
        }

