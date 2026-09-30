import numpy as np
from PIL import Image

IMAGE_SIZE = (64, 64)

def extract_features(image):
    """Extract a compact HOG-style feature vector from a hand image."""
    image = image.convert("L").resize(IMAGE_SIZE)
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    gx = np.zeros_like(pixels)
    gy = np.zeros_like(pixels)

    gx[:, 1:-1] = pixels[:, 2:] - pixels[:, :-2]
    gy[1:-1, :] = pixels[2:, :] - pixels[:-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180

    cell = 8
    bins = 9
    features = []

    for y in range(0, 64, cell):
        for x in range(0, 64, cell):
            mag = magnitude[y:y+cell, x:x+cell]
            ang = orientation[y:y+cell, x:x+cell]

            hist = np.zeros(bins)

            for m, a in zip(mag.ravel(), ang.ravel()):
                b = min(int(a // 20), bins - 1)
                hist[b] += m

            hist /= np.linalg.norm(hist) + 1e-8
            features.extend(hist)

    return np.asarray(features, dtype=np.float32)
