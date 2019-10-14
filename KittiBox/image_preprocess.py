import scipy as scp
import scipy.misc
import cv2
import numpy as np 
# current shape 1280 720
# desired shape 1248 384

# image = scp.misc.imread('./frames/test/frame0000.jpg')
# print(image.shape)
image = scp.misc.imresize(image, (384,
                                        682),
                                interp='cubic')
# scp.misc.imsave('./frames/testout/frame0000.jpg', image)
# print(image.shape)
# 299 * 2
image = np.pad(image, [(0, 0), (299, 299), (0, 0)], 'constant', constant_values = [(0,0),(0,0),(0,0)])
# print(image1.shape)
# scp.misc.imsave('./frames/testout/frame0001.jpg', image1)
