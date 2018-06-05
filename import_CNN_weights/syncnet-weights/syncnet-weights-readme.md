# syncnet-weights

Place pre-trained SyncNet weight files here.

## [2017-10-11]

Currently, pre-trained weights are available on the [VGG webpage](http://www.robots.ox.ac.uk/~vgg/software/lipsync/) for SyncNet. The weight files are:

- "syncnet_v4.mat"

- "syncnet_v7.mat"

_"v4" corresponds to model trained on frontal faces, while "v7" is the one trained on multi-view faces._

But these weights are in .mat format, they have not been saved via Matlab 7.3 version or later. Importing them into python is difficult. Weights saved via Matlab 7.3 version are required.

To be able to use the above files in Keras, load these into Matlab 7.3 (or higher) and resave them. (Make sure to update the file names in _syncnet\_params.py_ in the parent directory)

Or contact me for the resaved weights, I'm available at vikram.voleti@gmail.com.
