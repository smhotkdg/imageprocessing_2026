
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

plt.rcParams.update({
    'figure.dpi': 110,
    'font.size': 10
})

print('NumPy :', np.__version__)
print('OpenCV:', cv2.__version__)


# =========================================================
# 1. Check Functions
# =========================================================

def check(name, ok, hint=''):
    if ok is None:
        print(f'[Not completed] {name}')
    elif ok:
        print(f'[Passed] {name}')
    else:
        print(f'[Check again] {name} {hint}')


def same(a, b):
    if a is None:
        return None

    a = np.asarray(a)
    return a.shape == np.shape(b) and np.array_equal(a, b)


# =========================================================
# 2. Image Display Functions
# =========================================================

def show(images, titles=None, cols=None, size=3.2, cmap='gray'):

    if not isinstance(images, (list, tuple)):
        images = [images]
        titles = [titles] if titles else None

    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(size * cols, size * rows),
        squeeze=False
    )

    for i, ax in enumerate(axes.ravel()):
        ax.axis('off')

        if i >= n:
            continue

        im = images[i]

        if im.ndim == 2:
            ax.imshow(
                im,
                cmap=cmap,
                vmin=0,
                vmax=255 if im.dtype == np.uint8 else None
            )
        else:
            ax.imshow(im)

        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def show_grid(img, title=None, size=None):

    h, w = img.shape

    size = size or (0.55 * w + 1, 0.55 * h + 0.6)

    fig, ax = plt.subplots(figsize=size)
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)

    for (yy, xx), v in np.ndenumerate(img):
        ax.text(
            xx, yy, str(int(v)),
            ha='center',
            va='center',
            fontsize=9,
            color='black' if v > 127 else 'white'
        )

    ax.set_xticks(range(w))
    ax.set_yticks(range(h))

    ax.set_xlabel('x = column index')
    ax.set_ylabel('y = row index')

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    for k in range(w + 1):
        ax.axvline(k - 0.5, color='0.6', lw=0.6)

    for k in range(h + 1):
        ax.axhline(k - 0.5, color='0.6', lw=0.6)

    if title:
        ax.set_title(title, pad=28)

    plt.tight_layout()
    plt.show()

astro = data.astronaut()      # 우주인 사진 (RGB, 512x512)
cat = data.chelsea()          # 고양이 사진 (RGB, 300x451)
print('astro', astro.shape, '| cat', cat.shape)
astro_temp = astro.copy()

cv2.rectangle(
    astro_temp,
    (210, 100),     
    (250, 140),   
    (0, 255, 0),   
    2              
)


astro = data.astronaut()
cat = data.chelsea()

print('astro', astro.shape, '| cat', cat.shape)

astro_temp = astro.copy()

ycc = cv2.cvtColor(astro, cv2.COLOR_RGB2YCrCb)

#얼굴
#x1, y1 = 190, 145
#x2, y2 = 205, 165

#헬멧
x1, y1 = 380, 430
x2, y2 = 410, 460

cv2.rectangle(
    astro_temp,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

roi_rgb = astro[y1:y2+1, x1:x2+1]
roi_ycc = ycc[y1:y2+1, x1:x2+1]

print('RGB MIN:', roi_rgb.min(axis=(0, 1)))
print('RGB MAX:', roi_rgb.max(axis=(0, 1)))

print('YCrCb MIN:', roi_ycc.min(axis=(0, 1)))
print('YCrCb MAX:', roi_ycc.max(axis=(0, 1)))

show([astro_temp, cat], ['astronaut', 'chelsea (cat)'], size=4)

print('RGB [x=225, y=130]:', astro[130, 225])
print('YCrCb [x=225, y=130]:', ycc[130, 225])

Y, Cr, Cb = [ycc[..., i].astype(float) for i in range(3)]

book_mask = (
    (Cr >= 133) & (Cr <= 173) &
    (Cb >= 77) & (Cb <= 127)
)

show(book_mask.astype(np.uint8) * 255, 'book mask', size=4)

roi_ycc = ycc[y1:y2+1, x1:x2+1]

Cr = roi_ycc[:, :, 1]
Cb = roi_ycc[:, :, 2]

print('최소 Cr:', Cr.min())
print('최대 Cr:', Cr.max())
print('최소 Cb:', Cb.min())
print('최대 Cb:', Cb.max())

Y, Cr, Cb = cv2.split(ycc)

book_mask = (
    (Cr >= roi_ycc[:, :, 1].min()) &
    (Cr <= roi_ycc[:, :, 1].max()) &
    (Cb >= roi_ycc[:, :, 2].min()) &
    (Cb <= roi_ycc[:, :, 2].max())
)

show(book_mask.astype(np.uint8) * 255, 'skin mask', size=4)

