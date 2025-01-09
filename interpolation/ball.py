import pandas as pd
import numpy as np

def ball_interpolation(df):
    min_frame = df['frame'].min()
    max_frame = df['frame'].max()
    all_frames = pd.DataFrame({'frame': range(min_frame, max_frame + 1)})
    
    frames_with_ball = df[df['class_name'] == 'ball']['frame']
    frames_without_ball = all_frames[~all_frames['frame'].isin(frames_with_ball)]
    
    missing_ball_rows = frames_without_ball.copy()

    # Assign a list of NaNs to each row of 'box'
    missing_ball_rows['box'] = [[np.nan, np.nan, np.nan, np.nan] for _ in range(len(missing_ball_rows))]

    # Set default values for other columns
    missing_ball_rows['confidence'] = 0 
    missing_ball_rows['class_id'] = -1
    missing_ball_rows['class_name'] = 'ball'
    missing_ball_rows['tracker_id'] = -1
    
    df_with_missing = pd.concat([df, missing_ball_rows], ignore_index=True)
    df_with_missing = df_with_missing.sort_values(by=['frame']).reset_index(drop=True)
    
    box_coords = ['x_min', 'y_min', 'x_max', 'y_max']
    df_with_missing[box_coords] = pd.DataFrame(df_with_missing['box'].tolist(), index=df_with_missing.index)
    
    df_with_missing.loc[df_with_missing['class_name'] == 'ball', box_coords] = (
    df_with_missing[df_with_missing['class_name'] == 'ball'][box_coords]
    .interpolate(method='linear'))
    
    df_with_missing['box'] = df_with_missing[box_coords].apply(lambda row: row.tolist(), axis=1)
    df_with_missing = df_with_missing.drop(columns=box_coords)
    
    
    return df_with_missing