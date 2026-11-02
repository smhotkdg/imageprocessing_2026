import pixels_and_slicing as p
import numpy as np

tiny = np.zeros((6,8), np.uint8)
print('tiny.shape = ', tiny.shape, ' (높이, 너비)')
tiny[4,1] = 200
tiny[0,6] = 100

p1 = np.zeros((6, 8), np.uint8)
print("test")
p1[4,1] = 200
p1[0,6] = 100


expected = np.zeros((6, 8), np.uint8); expected[4, 1] = 200; expected[0, 6] = 100
if not p1.any():
    p.check('문제 1', None)
else:
    p.check('문제 1', p.same(p1, expected), hint='(x, y)를 배열에서는 [y, x]로 씁니다.')
    p.show_grid(p1, 'my answer')

p2 = np.zeros((6, 8), np.uint8)
print("test")
p2[3,:] = 50
p2[:,-1] = 255
p.show_grid(p2)

