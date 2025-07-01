import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, color, filters, morphology
from skimage.measure import label as cc_label
from scipy.ndimage import label as ndi_label
from skimage.draw import line
from collections import defaultdict

# --- Helpers ---

def neighbors8(y, x, shape):
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0:
                continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                yield ny, nx

def make_border_order(W):
    B, idx = W+2, 0
    bo = {}
    for x in range(B):
        bo[(0, x)] = idx; idx+=1
    for y in range(1, B-1):
        bo[(y, B-1)] = idx; idx+=1
    for x in range(B-1, -1, -1):
        bo[(B-1, x)] = idx; idx+=1
    for y in range(B-2, 0, -1):
        bo[(y, 0)] = idx; idx+=1
    return bo

# --- 1) load & skeletonize ---
img = io.imread('/Users/dauletmukhanov/Desktop/KMG/proto11.png')
# handle grayscale vs. RGB(A)
if img.ndim == 2:
    gray = img
else:
    # drop alpha if present
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    gray = color.rgb2gray(img)

# binarize and skeletonize
blur = filters.gaussian(gray, 1)
bw   = blur < filters.threshold_otsu(blur)
skel = morphology.skeletonize(bw)

# --- 2) detect centroids (exact 4‐arm gating) ---
def detect_centroids(skel, W=31):
    half = W//2
    # rough candidates: skeleton pixels with ≥3 neighbors
    cands = [(y,x) for y,x in np.argwhere(skel)
             if sum(skel[ny,nx] for ny,nx in neighbors8(y,x,skel.shape)) >= 3]
    mask = np.zeros_like(skel, bool)
    mask[tuple(zip(*cands))] = True
    mask = morphology.closing(mask, morphology.square(5))
    lbl, n = ndi_label(mask, structure=np.ones((3,3)))
    raw = [np.array(np.where(lbl==i)).mean(axis=1) for i in range(1, n+1)]
    bo = make_border_order(W)

    good = []
    for yc, xc in raw:
        yi, xi = int(round(yc)), int(round(xc))
        outer = skel[yi-half-1:yi+half+2, xi-half-1:xi+half+2]
        bm = np.zeros_like(outer, bool)
        bm[0,:] = bm[-1,:] = bm[:,0] = bm[:,-1] = True
        lblb, num = cc_label(outer & bm, connectivity=2, return_num=True)

        comps = []
        for comp in range(1, num+1):
            ys, xs = np.where(lblb==comp)
            ids = [bo[(y,x)] for y,x in zip(ys, xs)]
            comps.append((comp, np.median(ids)))
        comps.sort(key=lambda t: t[1])

        merged, grp, last = [], [comps[0][0]], comps[0][1]
        for comp, m in comps[1:]:
            if m - last < 1:
                grp.append(comp)
            else:
                merged.append(grp)
                grp = [comp]
            last = m
        merged.append(grp)

        if len(merged) == 4:
            good.append((yc, xc))

    # cluster nearby centroids
    parent = list(range(len(good)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(good)):
        for j in range(i+1, len(good)):
            if np.linalg.norm(np.array(good[i]) - np.array(good[j])) < 15:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(len(good)):
        clusters[find(i)].append(i)

    cents = [ np.mean([good[i] for i in grp], axis=0)
              for grp in clusters.values() ]
    return sorted(cents, key=lambda c: c[0])

cents = detect_centroids(skel, W=31)

# --- 3) carve & label segments ---
W, half = 31, 31//2
carved = skel.copy()
for yc, xc in cents:
    yi, xi = map(int, map(round, (yc, xc)))
    carved[yi-half:yi+half+1, xi-half:xi+half+1] = False

seg_lbl, _ = ndi_label(carved, structure=np.ones((3,3)))
segments = {i: np.argwhere(seg_lbl==i)
            for i in np.unique(seg_lbl) if i>0}

# --- 4) union‐find full curves ---
parent = list(range(len(segments)+1))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for sid, pts in segments.items():
    for y, x in pts:
        for ny, nx in neighbors8(y, x, skel.shape):
            t = seg_lbl[ny, nx]
            if t>0 and t!=sid:
                union(sid, t)

curves = defaultdict(list)
for sid in segments:
    curves[find(sid)].append(sid)

# --- 5) universal seeding & coloring ---
start_y   = {r: min(p[0] for s in segs for p in segments[s])
             for r, segs in curves.items()}
global_min = min(start_y.values())
starters   = [r for r, y0 in start_y.items() if abs(y0-global_min)<=1]
starters.sort(key=lambda r: min(p[1] for s in curves[r] for p in segments[s]))

import matplotlib.pyplot as mplplt
cmap    = mplplt.get_cmap('tab20', len(starters))
palette = [cmap(i)[:3] for i in range(len(starters))]

segment_color = {}
for i, r in enumerate(starters):
    for sid in curves[r]:
        segment_color[sid] = palette[i]

# --- 6) detect arms & record coords ---
bo = make_border_order(W)
intersection_arms = {}
for k, (yc, xc) in enumerate(cents, start=1):
    yi, xi = int(round(yc)), int(round(xc))
    outer = skel[yi-half-1:yi+half+2, xi-half-1:xi+half+2]
    bm = np.zeros_like(outer, bool)
    bm[0,:] = bm[-1,:] = bm[:,0] = bm[:,-1] = True
    lblb, nb = cc_label(outer & bm, connectivity=2, return_num=True)

    comps = []
    for comp in range(1, nb+1):
        ys, xs = np.where(lblb==comp)
        ids = [bo[(y,x)] for y,x in zip(ys, xs)]
        comps.append((comp, np.median(ids)))
    comps.sort(key=lambda t: t[1])

    merged, grp, last = [], [comps[0][0]], comps[0][1]
    for comp, m in comps[1:]:
        if m - last < 1:
            grp.append(comp)
        else:
            merged.append(grp)
            grp = [comp]
        last = m
    merged.append(grp)

    arms = []
    for group in merged:
        ys, xs = np.where(lblb==group[0])
        ry, rx = int(ys.mean()), int(xs.mean())
        sy, sx = yi-half-1 + ry, xi-half-1 + rx
        sid = next((seg_lbl[ny,nx]
                   for ny,nx in neighbors8(sy,sx,skel.shape)
                   if seg_lbl[ny,nx]>0), None)
        ang = (np.arctan2(sx-xi, -(sy-yi)) + 2*np.pi) % (2*np.pi)
        arms.append((ang, sid, (sy, sx)))

    arms.sort(key=lambda t:t[0])
    intersection_arms[k] = [(sid, coord) for _, sid, coord in arms]

# --- 7) junction‐based recoloring ---
for idx in sorted(intersection_arms, key=lambda i: cents[i-1][0]):
    arms = intersection_arms[idx]
    pre = [(i, segment_color[sid]) for i, (sid, _) in enumerate(arms)
           if sid in segment_color]
    if len(pre) == 2:
        for i, col in pre:
            opp_sid, _ = arms[(i+2)%4]
            if opp_sid not in segment_color:
                segment_color[opp_sid] = col

# --- 8) CONNECT THROUGH THE WINDOWS ---
canvas = np.ones((*skel.shape, 3))
# paint all segments
for sid, pts in segments.items():
    col = segment_color.get(sid, (1,1,1))
    for y, x in pts:
        canvas[y, x] = col

# draw straight connectors
for idx in intersection_arms:
    arms = intersection_arms[idx]
    for start in (0, 1):
        sid1, (y1, x1) = arms[start]
        sid2, (y2, x2) = arms[start+2]
        col = segment_color.get(sid1)
        if col is None or sid2 not in segment_color:
            continue
        rr, cc = line(y1, x1, y2, x2)
        canvas[rr, cc] = col

# --- 9) show final ---
plt.figure(figsize=(6,10))
plt.imshow(canvas)
plt.axis('off')
plt.title('Curves Connected Through Windows')
plt.show()
