import numpy as np

w = [255, 255, 255]
c = [0, 0, 0]
h = [50, 50, 50]
g = [90, 90, 90]
f = [139, 69, 19]
a = [255, 232, 122]
b = [255, 210, 100]
d = [205, 133, 63]
e = [255, 165, 0]
i = [255, 255, 153]
j = [255, 0, 0]

# fmt: off
# pylint: disable = line-too-long
pikachu_image: np.ndarray = np.array([
    [w, w, w, w, c, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w,    w, w, w, w, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, c, h, c, w, w, w, w,   w, w, w, w, w, w, w, w, w, w,    w, w, w, w, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, c, h, c, w, w, w, w,   w, w, w, w, w, w, w, w, w, w,    w, w, w, w, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, c, h, h, c, w, w, w,   w, w, w, w, w, w, w, w, w, w,    w, w, w, w, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, h, h, h, c, w, w, w,   w, w, w, w, w, w, w, w, w, w,    w, w, w, w, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, h, h, h, f, w, w, w,   w, w, w, w, w, w, w, w, w, w,    w, w, w, w, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, h, g, b, f, w, w, w,   w, w, w, w, w, w, w, w, w, w,    w, w, w, w, h, h, h, h, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, g, b, b, b, f, w, w,   w, w, w, w, w, w, w, w, w, w,    w, h, h, h, g, g, g, g, h, c,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, b, b, b, b, f, w, w,   w, w, w, w, w, w, w, w, w, d,    d, b, a, a, g, g, g, g, g, c,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, b, b, b, b, f, w, w,   d, d, d, d, w, w, w, d, d, a,    a, a, a, a, g, g, g, g, c, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, f, b, b, b, d, d, e,   i, i, i, a, a, e, d, a, a, a,    a, a, a, g, g, g, g, c, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, f, b, b, e, e, i, i,   i, i, i, i, a, a, a, a, a, a,    a, a, a, g, g, c, c, w, w, w,   w, w, w, w, d, d, w, w, w, w],
    [w, w, w, w, f, e, a, i, i, i,   i, i, i, a, a, a, a, a, a, a,    a, a, c, c, c, w, w, w, w, w,   w, w, w, d, a, a, d, w, w, w],
    [w, w, w, w, d, a, a, a, a, a,   a, a, a, b, c, c, b, a, a, e,    d, c, w, w, w, w, w, w, w, w,   w, w, d, a, a, a, a, d, w, w],
    [w, w, w, f, e, c, b, a, a, a,   a, a, a, g, w, c, h, a, a, a,    c, w, w, w, w, w, w, w, w, w,   w, e, a, a, a, a, a, d, w, w],
    [w, w, w, f, c, w, b, a, a, a,   a, a, a, c, c, g, h, a, a, a,    c, w, w, w, w, w, w, w, w, w,   e, a, a, a, a, a, a, a, d, w],
    [w, w, w, f, g, c, a, a, f, e,   a, a, a, b, c, c, b, a, a, a,    d, w, w, w, w, w, w, w, e, e,   a, a, a, a, a, a, a, a, d, w],
    [w, w, w, d, h, d, a, a, a, a,   a, a, a, a, a, a, a, j, j, a,    e, f, w, w, w, w, w, d, a, a,   a, a, a, a, a, a, a, a, d, w],
    [w, w, f, a, a, a, a, a, f, d,   e, a, a, a, a, a, j, j, j, j,    b, c, w, w, w, w, d, a, a, a,   a, a, a, a, a, a, a, a, a, d],
    [w, w, f, j, b, a, e, f, d, b,   b, d, d, e, a, b, j, j, j, j,    b, c, w, w, w, d, a, a, a, a,   a, a, a, a, a, a, a, a, a, d],
    [w, w, f, j, b, b, b, b, b, b,   b, b, b, b, b, b, j, j, j, j,    b, c, w, w, w, d, b, a, a, a,   a, a, a, a, a, a, a, a, a, d],
    [w, w, f, b, b, b, b, b, b, b,   b, b, b, b, b, b, b, b, j, j,    b, c, w, w, d, b, b, b, b, a,   a, a, a, a, a, a, a, a, c, w],
    [w, w, w, d, b, b, b, b, b, b,   b, b, b, b, b, b, b, b, b, b,    b, c, w, d, b, b, b, b, b, b,   b, a, a, a, a, a, c, c, w, w],
    [w, w, w, w, c, b, b, b, b, b,   b, b, b, b, b, b, b, b, b, b,    b, f, w, d, b, b, b, b, b, b,   b, b, b, a, c, c, w, w, w, w],
    [w, w, w, w, c, b, b, b, b, b,   b, b, b, b, b, b, b, b, b, b,    b, a, c, d, f, b, b, b, b, b,   b, b, c, c, w, w, w, w, w, w],
    [w, w, w, w, c, a, b, b, b, b,   b, b, b, b, b, b, b, b, b, b,    a, a, c, w, w, f, b, b, b, b,   c, c, w, w, w, w, w, w, w, w],
    [w, w, w, w, c, a, a, a, b, b,   b, b, b, b, b, b, b, b, b, a,    a, a, c, w, w, w, f, b, b, d,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, w, c, a, a, a, a, a,   b, b, b, b, b, e, b, a, a, a,    a, a, c, w, w, w, w, f, b, b,   f, w, w, w, w, w, w, w, w, w],
    [w, w, w, w, c, a, a, d, a, a,   a, a, a, a, e, a, a, a, a, a,    a, a, f, w, w, w, c, e, b, b,   f, w, w, w, w, w, w, w, w, w],
    [w, w, w, c, a, a, a, e, a, a,   a, a, a, a, e, a, a, a, a, a,    a, a, a, c, w, c, e, b, b, b,   b, f, w, w, w, w, w, w, w, w],
    [w, w, w, c, a, a, a, a, d, a,   a, a, a, d, a, a, a, a, a, b,    a, a, a, c, f, e, e, e, b, e,   f, c, w, w, w, w, w, w, w, w],
    [w, w, w, c, e, a, a, a, d, a,   a, a, a, d, a, a, a, a, b, e,    a, a, b, c, e, e, e, e, f, f,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, c, d, a, a, a, e, a,   a, a, d, a, a, a, a, a, d, e,    a, a, b, c, c, e, e, c, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, c, d, a, a, a, a, d,   a, a, c, e, a, a, a, d, a, a,    a, b, b, b, c, c, d, d, c, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, a, a, d, a, e, d, a,   a, a, a, c, a, e, d, a, a, a,    a, b, b, b, c, w, c, d, d, c,   w, w, w, w, w, w, w, w, w, w],
    [w, w, d, a, a, a, d, d, a, a,   a, a, a, a, d, d, a, a, a, a,    b, b, b, b, c, c, d, d, d, c,   w, w, w, w, w, w, w, w, w, w],
    [w, c, b, a, a, a, a, a, a, a,   a, a, a, a, a, a, a, a, b, a,    b, b, b, b, f, f, d, c, c, w,   w, w, w, w, w, w, w, w, w, w],
    [w, c, b, b, a, a, a, a, a, a,   a, a, a, a, a, a, a, b, a, b,    a, b, b, b, b, c, c, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, c, b, b, b, b, b, a, a, a,   a, a, a, a, b, b, b, a, b, a,    b, b, b, b, b, c, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, d, b, b, b, b, b, b, b,   b, b, b, b, b, b, b, b, b, b,    b, b, b, b, f, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, b, b, b, b, b, b, b,   b, b, b, b, b, b, b, b, b, b,    b, b, b, e, c, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, w, c, c, b, b, b, e, e,   c, c, c, c, f, b, b, b, b, b,    b, b, e, c, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, f, e, d, d, d, d, c,   w, w, w, w, w, c, f, d, d, e,    e, d, c, w, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, c, a, d, a, e, f, c, c, w,   w, w, w, w, w, w, w, w, f, b,    b, b, d, c, w, w, w, w, w, w,   w, w, w, w, w, w, w, w, w, w],
    [w, w, c, c, c, c, w, w, w, w,   w, w, w, w, w, w, w, w, w, w,    c, c, c, d, a, c, w, w, w, w,   w, w, w, w, w, w, w, w, w, w]],
    dtype=np.uint8,
)
# fmt: on
