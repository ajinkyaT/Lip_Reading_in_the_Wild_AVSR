from syncnet_functions import *


version = 'v4'
# # Multi-view
# version = 'v7'

# Mode = {'audio', 'lip', 'both'}

# mode = 'both'
mode = 'lip'
syncnet_lip_model = load_pretrained_syncnet_model(version=version, mode=mode, verbose=True)

syncnet_lip_model.save('syncnet_lip_model.h5')