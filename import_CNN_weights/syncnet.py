from syncnet_functions import *

# Version = {'v4', 'v7'}
# Frontal
version = 'v4'
# # Multi-view
# version = 'v7'

# Mode = {'audio', 'lip', 'both'}

# mode = 'both'
mode = 'both'
syncnet_audio_model, syncnet_lip_model = load_pretrained_syncnet_model(version=version, mode=mode, verbose=False)

# # mode = 'audio'
# mode = 'audio'
# syncnet_audio_model = load_pretrained_syncnet_model(version=version, mode=mode, verbose=False)

# # mode = 'lip'
# mode = 'lip'
# syncnet_lip_model = load_pretrained_syncnet_model(version=version, mode=mode, verbose=False)
