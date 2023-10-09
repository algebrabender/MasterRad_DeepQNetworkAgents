import numpy as np
import cv2

def preprocess_frame(screen, exclude, output):
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) 
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = cv2.resize(screen, (output, output), interpolation = cv2.INTER_AREA)
    
    return screen

def stack_frame(stacked_frames, frame, is_new):
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame
    
    return stacked_frames

