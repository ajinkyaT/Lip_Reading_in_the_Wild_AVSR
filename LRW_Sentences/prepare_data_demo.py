#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys
import dlib
import skvideo.io


FPS = 25
FRAME_ROWS = 112
FRAME_COLS = 112
NFRAMES = 5 # size of input volume of frames
MARGIN = NFRAMES/2
COLORS = 1 # grayscale
CHANNELS = COLORS*NFRAMES
SEQ_LEN = 250
SAMPLE_LEN = SEQ_LEN-2*MARGIN
DEFAULT_FACE_DIM = 292


mouth_destination_path = os.path.dirname('demo_data'+'/' + 'mouth')
if not os.path.exists(mouth_destination_path):
    os.makedirs(mouth_destination_path)
    
def process_video(input_video_path):
    temp_frames = np.zeros((SEQ_LEN*CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
    inputparameters = {}
    outputparameters = {}
    reader = skvideo.io.FFmpegReader(input_video_path,
                    inputdict=inputparameters,
                    outputdict=outputparameters)
    writer = skvideo.io.FFmpegWriter('demo_data/lip_area.mp4')
    video_shape = reader.getShape()
    (num_frames, h, w, c) = video_shape
    print(num_frames, h, w, c)
    predictor_path = 'dlib/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    activation = []
    max_counter = 250
    total_num_frames = int(video_shape[0])
    num_frames = min(total_num_frames,max_counter)
    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Required parameters for mouth extraction.
    width_crop_max = 0
    height_crop_max = 0

    for i in np.arange(SEQ_LEN):
        for frame in reader.nextFrame():
            if counter > num_frames:
                break
            
                    # Detection of the frame
            detections = detector(frame, 1)
        
            # 20 mark for mouth
            marks = np.zeros((2, 20))
        
            # All unnormalized face features.    
            if len(detections) > 0:
                for k, d in enumerate(detections):
        
                    # Shape of the face.
                    shape = predictor(frame, d)
        
                    co = 0
                    # Specific for the mouth.
                    for ii in range(48, 68):
                        """
                        This for loop is going over all mouth-related features.
                        X and Y coordinates are extracted and stored separately.
                        """
                        X = shape.part(ii)
                        A = (X.x, X.y)
                        marks[0, co] = X.x
                        marks[1, co] = X.y
                        co += 1
        
                    # Get the extreme points(top-left & bottom-right)
                    X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                        int(np.amax(marks, axis=1)[0]),
                                                        int(np.amax(marks, axis=1)[1])]
        
                    # Find the center of the mouth.
                    X_center = (X_left + X_right) / 2.0
                    Y_center = (Y_left + Y_right) / 2.0
        
                    # Make a boarder for cropping.
                    border = 30
                    X_left_new = X_left - border
                    Y_left_new = Y_left - border
                    X_right_new = X_right + border
                    Y_right_new = Y_right + border
        
                    # Width and height for cropping(before and after considering the border).
                    width_new = X_right_new - X_left_new
                    height_new = Y_right_new - Y_left_new
                    width_current = X_right - X_left
                    height_current = Y_right - Y_left
        
                    # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
                    if width_crop_max == 0 and height_crop_max == 0:
                        width_crop_max = width_new
                        height_crop_max = height_new
                    else:
                        width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                        height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)
        
                    # # # Uncomment if the lip area is desired to be rectangular # # # #
                    #########################################################
                    # Find the cropping points(top-left and bottom-right).
                    X_left_crop = int(X_center - width_crop_max / 2.0)
                    X_right_crop = int(X_center + width_crop_max / 2.0)
                    Y_left_crop = int(Y_center - height_crop_max / 2.0)
                    Y_right_crop = int(Y_center + height_crop_max / 2.0)
                    #########################################################
                    if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                        mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]
        
                        # Save the mouth area.
                        mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                        cv2.imwrite(mouth_destination_path + '/' + 'frame' + '_' + str(counter) + '.png', mouth_gray)
        
                        print("The cropped mouth is detected ...")
                        activation.append(1)
                    else:
                        cv2.putText(frame, 'The full mouth is not detectable. ', (30, 30), font, 1, (0, 255, 255), 2)
                        print("The full mouth is not detectable. ...")
                        activation.append(0)
            else:
                cv2.putText(frame, 'Mouth is not detectable. ', (30, 30), font, 1, (0, 0, 255), 2)
                print("Mouth is not detectable. ...")
                activation.append(0)
            
            if activation[counter] == 1:
                # Demonstration of face.
                cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)
    
            # cv2.imshow('frame', frame)
            print('frame number %d of %d' % (counter, num_frames))
        
            # write the output frame to file
            print("writing frame %d with activation %d" % (counter + 1, activation[counter]))
            try:
                writer.writeFrame(frame)
            except:
                pass
            counter += 1
    writer.close()
    

#		if len(faces)==0 or len(faces)>1:
#			print ('Face detection error in %s frame: %d'%(vf, i))							
#			face = frame[199:-85,214:-214] # hard-coded face location
#		else:
#			for (x,y,w,h) in faces:
#				face = frame[y:y+DEFAULT_FACE_DIM,x:x+DEFAULT_FACE_DIM]
#		face = cv2.resize(face,(FRAME_COLS,FRAME_ROWS))
#		face = np.expand_dims(face,axis=2)
#		face = face.transpose(2,0,1)
#		temp_frames[i*COLORS:i*COLORS+COLORS,:,:] = face

if __name__ == '__main__':
    input_video_path = 'data/00005_25_fps.mp4'
    process_video(input_video_path)