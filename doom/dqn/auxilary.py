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

# Memory to enable our agent not only learn from current s, a,
# but every iteration take sample from memory (s, a, r, s')
class Memory():

    def __init__(self, max_size):
        self.buf = deque(maxlen = max_size)

    def add(self, exp):
        self.buf.append(exp)

    def sample(self, batch_size):
        buf_size = len(self.buf)
        indx = np.random.choice(np.arange(buf_size),
                                size = batch_size,
                                    replace = False)
        return [self.buf[i] for i in indx]
