import numpy as np
from skimage import transform

# We want to greyscale img, crop image(get rid of the roof since it contains no useful information) and normalize pixels
def preprocess(frame):
    # greyscale
    frame = np.mean(frame, -1)

    # crop
    frame = frame[30:-10, 30:-30]

    # normalize
    frame = frame / 255.0

    # resize
    frame = transform.resize(frame, [84,84])

    return frame

# to give machine some sense of motion, we want to feed net with not a single img, but instead 4 stacked images
def stack_images(stacked_frames, cur_frame):
    # preprocess frame
    frame = preprocess(frame)

    # append to deque
    stacked_frames.append(frame)

    # build frame, that stacked from 4 other stack_images
    stacked_frame = np.stack(stacked_frames, axis = 2)

    return stacked_frame
