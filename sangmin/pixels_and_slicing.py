import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
print('NumPy :', np.__version__)
print('OpenCV:', cv2.__version__)

# tiny = np.zeros((6, 8), np.uint8)
# print('tiny.shape =', tiny.shape, ' (높이, 너비)')
# tiny[2, 5] = 255          # [행 y=2, 열 x=5]
# tiny[4, 1] = 200
# tiny[0, 1] = 100
# show_grid(tiny, 'tiny[2, 5] = 255')

def check(name, ok, hint=''):
    # 문제 결과를 알려 주는 도우미: None이면 미작성, True면 통과
    if ok is None:
        print(f'[미작성] {name}: TODO를 채운 뒤 이 셀을 다시 실행하세요.')
    elif ok:
        print(f'[통과] {name}')
    else:
        print(f'[다시 확인] {name}  {hint}')


def same(a, b):
    # 두 배열이 모양·값까지 같은지 (None이면 None)
    if a is None:
        return None
    a = np.asarray(a)
    return a.shape == np.shape(b) and np.array_equal(a, b)


def show(images, titles=None, cols=None, size=3.2, cmap='gray'):
    # 여러 장을 한 줄로 보여 준다. 흑백은 0~255 고정, 컬러는 RGB로 가정
    if not isinstance(images, (list, tuple)):
        images, titles = [images], [titles] if titles else None
    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis('off')
        if i >= n:
            continue
        im = images[i]
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=255 if im.dtype == np.uint8 else None)
        else:
            ax.imshow(im)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def show_grid(img, title=None, size=None):
    # 아주 작은 흑백 이미지를 칸과 숫자, 행·열 번호와 함께 보여 준다
    h, w = img.shape
    size = size or (0.55 * w + 1, 0.55 * h + 0.6)
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    for (yy, xx), v in np.ndenumerate(img):
        ax.text(xx, yy, str(int(v)), ha='center', va='center', fontsize=9,
                color='black' if v > 127 else 'white')
    ax.set_xticks(range(w)); ax.set_yticks(range(h))
    ax.set_xlabel('x = column index'); ax.set_ylabel('y = row index')
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    for k in range(w + 1):
        ax.axvline(k - 0.5, color='0.6', lw=0.6)
    for k in range(h + 1):
        ax.axhline(k - 0.5, color='0.6', lw=0.6)
    if title:
        ax.set_title(title, pad=28)
    plt.tight_layout()
    plt.show()

def crop(img, x, y, w, h):
    return img[y:y+h, x:x+w]

astronaut = data.astronaut()
astronaut_YCrCb = cv2.cvtColor(astronaut, cv2.COLOR_RGB2YCrCb)

face_1 = (200,120)
face_2 = (225, 90)
face_3 = (220, 170)
background = (100, 100)
cloth = (200, 350)

face_1_reversed = tuple(reversed(face_1))
face_2_reversed = tuple(reversed(face_2))
face_3_reversed = tuple(reversed(face_3))
background_reversed = tuple(reversed(background))
cloth_reversed = tuple(reversed(cloth))


## get color
face_1_rgb = astronaut[face_1_reversed]
face_2_rgb = astronaut[face_2_reversed]
face_3_rgb = astronaut[face_3_reversed]
background_rgb = astronaut[background_reversed]
cloth_rgb = astronaut[cloth_reversed]

face_1_YCrCb = astronaut_YCrCb[face_1_reversed]
face_2_YCrCb = astronaut_YCrCb[face_2_reversed]
face_3_YCrCb = astronaut_YCrCb[face_3_reversed]
background_YCrCb = astronaut_YCrCb[background_reversed]
cloth_YCrCb = astronaut_YCrCb[cloth_reversed]

print("face_1_rgb : ", face_1_rgb, ", face_1_YCrCb : ", face_1_YCrCb)
print("face_2_rgb : ", face_2_rgb, ", face_2_YCrCb : ", face_2_YCrCb)
print("face_3_rgb : ", face_3_rgb, ", face_3_YCrCb : ", face_3_YCrCb)
print("background_rgb : ", background_rgb, ", background_YCrCb : ", background_YCrCb)
print("cloth_rgb : ", cloth_rgb, ", cloth_YCrCb : ", cloth_YCrCb)

## plot image

circle_radius = 5
circle_color = (0,0,255)
circle_thickness = 2

processed_image = cv2.circle(astronaut, face_1, circle_radius, circle_color, circle_thickness)
processed_image = cv2.circle(processed_image, face_2, circle_radius, circle_color, circle_thickness)
processed_image = cv2.circle(processed_image, face_3, circle_radius, circle_color, circle_thickness)
processed_image = cv2.circle(processed_image, background, circle_radius, circle_color, circle_thickness)
processed_image = cv2.circle(processed_image, cloth, circle_radius, circle_color, circle_thickness)

show(processed_image)
