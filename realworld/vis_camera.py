from realsense import RealSense
import cv2
import numpy as np
import torch
from mjenvs.embedding.dpt import DptProcess
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading





camera = RealSense(width=640, height=480)
camera.start()

img_size = 224

def transform(img):
    orig_h, orig_w, _ = img.shape
    orig_ratio = orig_w / orig_h
    target_ratio = 1
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
    
    crop_img = cv2.resize(crop_img, (img_size, img_size))
    crop_img = np.rot90(crop_img, k=-1, axes=(0, 1))
    return crop_img


dpt_depth_np = np.zeros((img_size, img_size, 1), dtype=np.uint8)


if __name__ == '__main__':

    try:

        def plot_histogram():
            fig, ax = plt.subplots()
            x = np.arange(256)  # Assuming depth values range from 0 to 255
            hist, = ax.plot(x, np.zeros(256))
            ax.set_title('Depth Image Histogram')
            ax.set_xlabel('Depth Value')
            ax.set_ylabel('Frequency')

            def update(frame):
                # Calculate the histogram
                global dpt_depth_np
                if dpt_depth_np is not None:
                    print("===", dpt_depth_np.mean(), dpt_depth_np.shape)
                    hist_data = cv2.calcHist([dpt_depth_np], [0], None, [256], [0, 256])
                    print(hist_data)
                    hist.set_ydata(hist_data.flatten())
                    return hist,

            # Create an animation
            ani = FuncAnimation(fig, update, frames=None, blit=False, interval=20)

            plt.show()

        # Start the histogram plotting in a new thread
        histogram_thread = threading.Thread(target=plot_histogram)
        histogram_thread.start()
        
        dpt = DptProcess((3, img_size, img_size), dpt_size=518, model_type='vitl').cuda()
        while True:
            data = camera.get_frame()
            
            img = data['color']
            depth = data['depth']

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            depth = depth.astype(np.uint8)

            # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            print(depth.min(), depth.max(), depth.shape, img.shape)

            new_img = transform(img)
            new_depth = transform(depth)

            print(new_img.shape)
            
            dpt_depth = dpt(torch.as_tensor(new_img.transpose(2, 0, 1).copy(), device='cuda'))
            out_min = dpt_depth.amin((-2, -1), keepdim=True)
            out_max = dpt_depth.amax((-2, -1), keepdim=True)
            dpt_depth = (dpt_depth - out_min) / (out_max - out_min) * 255.
            dpt_depth_np = dpt_depth.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            # depth += torch.from_numpy(np.random.normal(0, 2.55, depth.shape)).to(depth.device)
            # depth = torch.clamp(depth, 0, 255)
            print(dpt_depth.shape)
            cv2.imshow("dpt", dpt_depth_np)

            cv2.imshow("transformed_depth", new_depth)
            # cv2.imshow("orig_depth", depth)
            cv2.imshow("transformed_rgb", new_img)
            cv2.imshow("orig_rgb", img)
            cv2.waitKey(1)

            
    finally:
        camera.stop()