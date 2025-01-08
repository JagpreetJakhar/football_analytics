import numpy as np
import cv2 as cv
colors = [
        (255, 191, 0),  # Sky Blue (#00BFFF in RGB)
        (211, 20, 255),  # Deep Pink (#FF1493 in RGB)
        (0, 215, 255)  ]
def draw_ellipse(frame, box,class_id):
    x1= int(box[0])
    x2= int(box[2])
    y1= int(box[1])
    y2= int(box[3])
    width = x2-x1
    center = int((x1+x2)/2),int(y2)
      # Gold (#FFD700 in RGB)
    color = colors[class_id]
    frame = cv.ellipse(
        frame,
        center=center,
        axes=(int(width), int(0.30 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=200,
        color=color,
        thickness=2,
        lineType=cv.LINE_AA,
    )
    return frame
    
def draw_triangle(frame,box):
    y = int(box[1])
    color = (0, 215, 255)  # Gold in BGR
    b = 25  
    h = 21  
    x_center = int((box[0] + box[2]) / 2)

    vertices = np.array([
        [x_center, y],              # Bottom center
        [x_center - b // 2, y - h], # Top left
        [x_center + b // 2, y - h]  # Top right
    ], dtype=np.int32)

    frame = cv.fillPoly(frame, [vertices], color=color)
    return frame
def annotate(frame,box,class_id,tracker_id):
    x1, y2 = int(box[0]), int(box[3])  
    text = f"#{tracker_id}"

    # Define font, scale, and thickness
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (0, 0, 0)
    padding = 5  # Add padding

    # Calculate text size to center it horizontally
    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
    text_width = text_size[0] + 2 * padding  # Adding padding to the width
    text_height = text_size[1] + 2 * padding  # Adding padding to the height

 
    text_x = int(x1 - text_width // 2 + (box[2] - box[0]) // 2)
    text_y = int(y2 + text_height + padding)  


    frame = cv.rectangle(frame, (text_x - padding, text_y - text_height), (text_x + text_width - padding, text_y), color=colors[class_id], thickness=-1)

    # Put the text on the frame
    frame = cv.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv.LINE_AA,
    )
    return frame