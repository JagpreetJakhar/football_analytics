import cv2 as cv
import numpy as np
def generate_frames(video):
    reader = cv.VideoCapture(video)
    frames=[]
    while True:
        ret,frame = reader.read()
        if not ret:
            break
        frames.append(frame)
    reader.release()
    return frames

def generate_video(frames,dest):
    fmt = cv.VideoWriter_fourcc(*'mp4v')
    #default 25fps
    height, width = frames[0].shape[:2]
    writer = cv.VideoWriter(dest, fmt, 25.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    
def crop_image(image: np.ndarray, bbox: np.ndarray):
    xyxy = np.round(bbox).astype(int)
    x1, y1, x2, y2 = xyxy
    cropped_img = image[y1:y2, x1:x2]
    torso_cropped_img = cropped_img[0:int(cropped_img.shape[0]/2),:]
    return torso_cropped_img