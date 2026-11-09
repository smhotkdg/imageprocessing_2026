
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


# =========================================================
# 3. Tiny Image Example
# =========================================================

# tiny = np.zeros((6, 8), np.uint8)
# tiny[2, 5] = 255
# tiny[4, 1] = 200
# tiny[0, 1] = 100
# show_grid(tiny, 'Tiny Image')


# =========================================================
# 4. Load Camera Image
# =========================================================

cam = data.camera()

H, W = cam.shape

print(
    'shape:', cam.shape,
    '| dtype:', cam.dtype,
    '| min:', cam.min(),
    '| max:', cam.max()
)

print('cam[100, 200] =', cam[100, 200])
print('cam[200, 100] =', cam[200, 100])


# =========================================================
# 5. Pixel Coordinate Visualization
# =========================================================

fig, ax = plt.subplots(figsize=(5, 5))

ax.imshow(cam, cmap='gray', vmin=0, vmax=255)

ax.plot(
    200, 100, 'o',
    color='red',
    ms=9,
    label='cam[100, 200]'
)

ax.plot(
    100, 200, 'o',
    color='dodgerblue',
    ms=9,
    label='cam[200, 100]'
)

ax.legend(loc='lower left', fontsize=8)
ax.set_title('Same Numbers, Different Pixels')

plt.tight_layout()
plt.show()


# =========================================================
# 6. Row Brightness Analysis
# =========================================================

y_row = 200

row = cam[y_row, :].copy()
row.sort()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))

ax[0].imshow(cam, cmap='gray', vmin=0, vmax=255)
ax[0].axhline(y_row, color='red')
ax[0].set_title(f'Row y = {y_row}')
ax[0].axis('off')

ax[1].plot(row)

ax[1].set(
    xlabel='Sorted Pixel Index',
    ylabel='Brightness',
    title='Sorted Row Brightness',
    ylim=(0, 260)
)

plt.tight_layout()
plt.show()

print('Row length:', row.shape)
print('Mean brightness:', round(float(row.mean()), 1))
print('Median brightness:', np.median(row))


# =========================================================
# 7. Crop Function
# =========================================================

def crop(img, x, y, w, h):

    y_end = min(y + h, img.shape[0])
    x_end = min(x + w, img.shape[1])

    out = np.zeros(
        (y_end - y, x_end - x) + img.shape[2:],
        dtype=img.dtype
    )

    for yy in range(y, y_end):
        for xx in range(x, x_end):
            out[yy - y, xx - x] = img[yy, xx]

    return out


# =========================================================
# 8. Crop Test
# =========================================================

tests = [
    (150, 60, 130, 140),
    (0, 0, 10, 20),
    (500, 400, 12, 112),
    (250, 130, 80, 60)
]

bad = [
    t for t in tests
    if not same(
        crop(cam, *t),
        cam[t[1]:t[1]+t[3], t[0]:t[0]+t[2]]
    )
]

check('Problem 7', not bad)

show(
    [cam] + [crop(cam, *t) for t in tests],
    ['Camera'] + [f'Crop {t}' for t in tests]
)


# =========================================================
# 9. Building Crop
# =========================================================

bx, by, bw, bh = 340, 110, 172, 140

building = crop(cam, bx, by, bw, bh)

show(
    [cam, building],
    ['Camera', 'Building Crop'],
    size=4
)


# =========================================================
# 10. Load Astronaut and Cat
# =========================================================

astro = data.astronaut()
cat = data.chelsea()

print('Astronaut shape:', astro.shape)
print('Cat shape:', cat.shape)

show(
    [astro, cat],
    ['Astronaut', 'Chelsea (Cat)'],
    cols=2,
    size=4
)


# =========================================================
# 11. Astronaut Sampling Points
# =========================================================

# Coordinates: (x, y)

points = {
    'Cheek': (200, 140),
    'Forehead': (220, 85),
    'Chin': (225, 175),
    'Background': (340, 60),
    'Spacesuit': (120, 320)
}


# =========================================================
# 12. Pixel Analysis Function
# =========================================================

def analyze_pixel(img, x, y, window=5):

    # Selected pixel RGB
    rgb = img[y, x].astype(int)

    # Brightness
    brightness = (
        0.2126 * rgb[0] +
        0.7152 * rgb[1] +
        0.0722 * rgb[2]
    )

    # HEX color
    hex_color = '#{:02X}{:02X}{:02X}'.format(*rgb)

    # 5x5 neighborhood
    half = window // 2

    region = img[
        max(0, y-half):min(img.shape[0], y+half+1),
        max(0, x-half):min(img.shape[1], x+half+1)
    ]

    mean_rgb = region.mean(axis=(0, 1))

    mean_brightness = (
        0.2126 * mean_rgb[0] +
        0.7152 * mean_rgb[1] +
        0.0722 * mean_rgb[2]
    )

    return (
        rgb,
        brightness,
        hex_color,
        mean_rgb,
        mean_brightness
    )


# =========================================================
# 13. Show Sampling Points on Astronaut
# =========================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(astro)

for name, (x, y) in points.items():

    ax.scatter(
        x, y,
        s=90,
        c='yellow',
        edgecolors='black',
        linewidths=1.5
    )

    ax.annotate(
        name,
        (x, y),
        xytext=(12, -12),
        textcoords='offset points',
        color='yellow',
        fontsize=10,
        fontweight='bold',
        bbox=dict(
            facecolor='black',
            alpha=0.6
        )
    )

ax.set_title('Astronaut - Pixel Sampling Points')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

plt.tight_layout()
plt.show()


# =========================================================
# 14. RGB and Brightness Results
# =========================================================

results = {}

for name, (x, y) in points.items():

    rgb, brightness, hex_color, mean_rgb, mean_brightness = \
        analyze_pixel(astro, x, y)

    results[name] = {
        'rgb': rgb,
        'brightness': brightness,
        'hex': hex_color,
        'mean_rgb': mean_rgb,
        'mean_brightness': mean_brightness
    }

    print(f'\n[{name}]')
    print(f'Coordinate: ({x}, {y})')
    print(f'RGB: {rgb.tolist()}')
    print(f'Brightness: {brightness:.2f} / 255')
    print(f'HEX: {hex_color}')
    print(f'5x5 Mean RGB: {mean_rgb.round(2)}')
    print(f'5x5 Mean Brightness: {mean_brightness:.2f} / 255')


# =========================================================
# 15. Sampled Color Comparison
# =========================================================

fig, axes = plt.subplots(
    1,
    len(points),
    figsize=(15, 3)
)

for ax, (name, result) in zip(axes, results.items()):

    color = result['rgb'] / 255.0

    patch = np.ones((80, 80, 3), dtype=float)
    patch[:] = color

    ax.imshow(patch)

    ax.set_title(
        f"{name}\n"
        f"RGB: {result['rgb'].tolist()}\n"
        f"Brightness: {result['brightness']:.1f}",
        fontsize=9
    )

    ax.axis('off')

plt.suptitle('Astronaut - Sampled Pixel Colors')
plt.tight_layout()
plt.show()


# =========================================================
# 16. Brightness Comparison Graph
# =========================================================

names = list(results.keys())

brightness_values = [
    results[name]['brightness']
    for name in names
]

plt.figure(figsize=(10, 4))

bars = plt.bar(
    names,
    brightness_values,
    color=[
        results[name]['rgb'] / 255.0
        for name in names
    ],
    edgecolor='black'
)

for bar, value in zip(bars, brightness_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 3,
        f'{value:.1f}',
        ha='center'
    )

plt.ylabel('Brightness (0-255)')
plt.ylim(0, 270)
plt.title('Brightness Comparison by Region')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()
