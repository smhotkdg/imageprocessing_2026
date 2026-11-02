import pixels_and_slicing as p
import numpy as np

tiny = np.zeros((6,8), np.uint8)
print('tiny.shape = ', tiny.shape, ' (높이, 너비)')
tiny[2,5] = 255
p.show_grid(tiny, 'tiny[2,5] = 255')