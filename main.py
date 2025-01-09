#!#!/usr/bin/python
import argparse
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import ultralytics
import cv2 as cv
from utils import generate_video,generate_frames,crop_image
from inference import infer
from visualisations import draw_triangle,draw_ellipse,annotate
from interpolation import ball_interpolation
def main():
    print("Starting...")
    #ultralytics.checks()
    
    parser = argparse.ArgumentParser(description="Take Input Video for Inference and Output Video Result")
    parser.add_argument('-i', '--input', required=True, help="Path to the input video file")
    parser.add_argument('-o', '--output', required=True, help="Path to the output video file")
    parser.add_argument('-r', '--result', required=True, help="Name of Results CSV file")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    RESULT_NAME = args.result
    
    frames = generate_frames(input_path)
    if frames:
        NO_OF_FRAMES=len(frames)
        print(f"Number of frames generated: {NO_OF_FRAMES}")
    else:
        print("No frames generated from the video. Please check the input path.")
    result_df = infer(NO_OF_FRAMES,frames,RESULT_NAME)
    result_df = ball_interpolation(result_df)
    result_df.to_parquet(f'results/{RESULT_NAME}.parquet', index=False)
    print(f'Results saved to results/{RESULT_NAME}.parquet')
    print("===================")
    out_frames=[]
    for i in range(NO_OF_FRAMES):
        frame = frames[i].copy()
        for _, row in result_df[result_df['frame']==i].iterrows():
    # Extract the bounding box coordinates, class, and confidence score
            box = row['box']
            class_id = row['class_id']
            track_id = row['tracker_id']
            if class_id>-1:
               frame= draw_ellipse(frame,box,class_id)
            if class_id==-1:
                frame=draw_triangle(frame,box)
            if track_id>-1:
               frame= annotate(frame,box,class_id,track_id)
        #cv.imwrite(f'output/manframes/out_{i:04}.jpg',frame)
        out_frames.append(frame)
    #Save Results

    generate_video(out_frames, output_path)
    print(f"Video has been saved to {output_path}")
    
    
    
if __name__=='__main__':
    main()