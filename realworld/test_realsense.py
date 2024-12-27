from realsense import RealSense
import cv2
import numpy as np

camera = RealSense(width=640, height=480)
camera.start()

width = height = 640

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
    crop_img = np.rot90(crop_img, k=-1, axes=(0, 1))
    return crop_img

try:
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
        # new_depth = cv2.resize(new_depth, (128, 128))
        
        cv2.imshow("transformed_depth", new_depth)
        # cv2.imshow("orig_depth", depth)
        cv2.imshow("transformed_rgb", new_img)
        cv2.imshow("orig_rgb", img)
        cv2.waitKey(1)
finally:
    camera.stop()