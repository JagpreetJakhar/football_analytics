import supervision as sv
from ultralytics import YOLO
import numpy as np
import pandas as pd
from classifier import TeamClassifier,goalkeeper
from utils import crop_image
BALL_ID=0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

model= YOLO("models/best.pt")
tracker = sv.ByteTrack()
tracker.reset()
team_classifier = TeamClassifier(device="cuda")
def infer(NO_OF_FRAMES,frames,RESULT_NAME):
# Process the input video
    predictions=[]
    data=[]
    team_classifier.fit(frames)
    crops = []
    for i in range(1,NO_OF_FRAMES,20):
        result = model.predict(frames[i],save=False,show_labels=False,line_width=1,show_boxes=False,verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [crop_image(frames[i], xyxy) for xyxy in players_detections.xyxy]
        crops += players_crops
    team_classifier.fit(crops) 
    print('=====Team Classifier Done=====') 
    for i in range(NO_OF_FRAMES):
        predictions += model.predict(frames[i],save=False,show_labels=False,line_width=1,show_boxes=False,verbose=False)
        detections = sv.Detections.from_ultralytics(predictions[i])
        
        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)
        
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]
        
        players_crops = [crop_image(frames[i], xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        goalkeepers_detections.class_id = goalkeeper(
            players_detections, goalkeepers_detections)

        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections])

        all_detections.class_id = all_detections.class_id.astype(int)
        for t in all_detections:
            data.append({'frame':i,'box':t[0].tolist(),'confidence':t[2],'class_id':t[3],'class_name':t[5]['class_name'],'tracker_id':t[4]})
        
        for t in ball_detections:
            data.append({'frame':i,'box':t[0].tolist(),'confidence':t[2],'class_id':-1,'class_name':'ball','tracker_id':-1})
    all_detections_df = pd.DataFrame(data) 

    return all_detections_df