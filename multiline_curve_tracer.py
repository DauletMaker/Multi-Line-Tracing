import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology

def get_neighbors(y, x, shape):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors

def compute_segment_direction(segment):
    if len(segment) < 2:
        return None
    y0, x0 = segment[0]
    y1, x1 = segment[-1]
    v = np.array([y1 - y0, x1 - x0])
    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    return v / norm

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def trace_direction_match(start, skeleton, junctions_set, window=15):
    visited = np.zeros_like(skeleton, dtype=bool)
    path = [start]
    visited[start] = True
    current = start

    while True:
        neighbors = [
            (ny, nx)
            for dy in [-1, 0, 1]
            for dx in [-1, 0, 1]
            if (dy != 0 or dx != 0)
            and 0 <= (ny := current[0] + dy) < skeleton.shape[0]
            and 0 <= (nx := current[1] + dx) < skeleton.shape[1]
            and skeleton[ny, nx]
            and not visited[ny, nx]
        ]

        if not neighbors:
            break

        prev_segment = path[-window:] if len(path) >= 2 else path
        incoming_dir = compute_segment_direction(prev_segment)
        if incoming_dir is None:
            next_pixel = neighbors[0]
        else:
            candidates = []
            for n in neighbors:
                direction = compute_segment_direction([current, n])
                if direction is not None:
                    angle = angle_between(incoming_dir, direction)
                    candidates.append((angle, n))
            if candidates:
                candidates.sort()
                _, next_pixel = candidates[0]
            else:
                break

        visited[next_pixel] = True
        path.append(next_pixel)
        current = next_pixel

        if current in junctions_set:
            continue

    return path

def find_grouped_junctions(skeleton, radius=3):
    points = np.argwhere(skeleton)
    junctions = []
    for y, x in points:
        neighbors = [n for n in get_neighbors(y, x, skeleton.shape) if skeleton[n]]
        if len(neighbors) > 2:
            junctions.append((y, x))

    grouped = []
    visited = set()
    for jy, jx in junctions:
        if (jy, jx) in visited:
            continue
        group = [(jy, jx)]
        for oy, ox in junctions:
            if (oy, ox) in visited:
                continue
            if np.linalg.norm([jy - oy, jx - ox]) <= radius:
                group.append((oy, ox))
                visited.add((oy, ox))
        gy, gx = np.mean(group, axis=0).astype(int)
        grouped.append((gy, gx))
    return set(grouped)


# Load and preprocess image
image_path = "proto6 3.png"  # Replace with your image path
image = io.imread(image_path)
if image.shape[-1] == 4:
    image = image[:, :, :3]

gray = color.rgb2gray(image)
blurred = filters.gaussian(gray, sigma=1)
binary = blurred < filters.threshold_otsu(blurred)
skeleton = morphology.skeletonize(binary)

# Identify endpoints
line_pixels = np.argwhere(skeleton)
from_endpoints = [
    tuple(pt) for pt in line_pixels
    if sum(skeleton[ny, nx] > 0 for ny, nx in get_neighbors(pt[0], pt[1], skeleton.shape)) == 1
]

# Detect junctions and initialize
grouped_junctions = find_grouped_junctions(skeleton, radius=3)
visited_global = np.zeros_like(skeleton, dtype=bool)
final_image = np.ones((*skeleton.shape, 3), dtype=np.uint8) * 255

# Color palette
color_palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 128, 128)
]

# Trace and color each line
color_idx = 0
for ep in from_endpoints:
    if visited_global[ep]:
        continue
    traced_path = trace_direction_match(ep, skeleton, grouped_junctions, window=15)
    for y, x in traced_path:
        final_image[y, x] = color_palette[color_idx % len(color_palette)]
        visited_global[y, x] = True
    color_idx += 1

for y, x in line_pixels:
    if not visited_global[y, x]:
        final_image[y, x] = (0, 0, 255)

# Mark junctions in purple
for jy, jx in grouped_junctions:
    for dy in range(-2, 7):
        for dx in range(-2, 7):
            ny, nx = jy + dy, jx + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                final_image[ny, nx] = (255, 0, 255)

plt.figure(figsize=(10, 10))
plt.imshow(final_image)
plt.axis("off")
plt.title("Multi-Line Tracing with Colored Paths and Junctions")
plt.show()
