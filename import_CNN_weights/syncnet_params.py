import os

# SYNCNET - PARAMS

#############################################################
# PARAMS
#############################################################

# SYNCNET directory
if 'SYNCNET_DIR' not in dir():
    SYNCNET_DIR = os.path.dirname(os.path.realpath(__file__))

SYNCNET_WEIGHTS_FILE_V4 = os.path.join(SYNCNET_DIR, 'syncnet-weights/lipsync_v4_73.mat')

SYNCNET_WEIGHTS_FILE_V7 = os.path.join(SYNCNET_DIR, 'syncnet-weights/lipsync_v7_73.mat')

# IMAGE_DATA FORMAT = {'channels_first', 'channels_last'}
IMAGE_DATA_FORMAT = 'channels_last'

#############################################################
# CONSTANTS
#############################################################

MOUTH_H = 112

MOUTH_W = 112

FACE_H = 224

FACE_W = 224

SYNCNET_VIDEO_CHANNELS = 5

SYNCNET_MFCC_CHANNELS = 12

AUDIO_TIME_STEPS = 20
