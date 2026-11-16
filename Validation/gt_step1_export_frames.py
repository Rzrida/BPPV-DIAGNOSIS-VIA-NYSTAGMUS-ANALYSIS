# ===== GT STEP 1 (local / VS Code): crop the eye video + export ~50 frames to hand-label =====
# Install once:  pip install opencv-python numpy pandas PyWavelets
# Run:           python gt_step1_export_frames_local.py
# Outputs (next to this script):
#   cropped_eye.mp4        the cropped video (same crop method as Cell 1 of the notebook)
#   label_frames/          50 clean PNG frames, shuffled, no overlays   -> input of step 2
#   label_index.csv        file -> frame number + category              -> needed for step 3
#   crop_info.json         crop box etc.
#   crop_preview.png       CHECK THIS: pupil must be inside the green box in every frame
import json, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import pywt

VIDEO = r"/Users/ridazahra/Desktop/FYP/Validation/pupil_x_y/video.mp4"          # <-- put your video path here

# ---- crop settings: keep IDENTICAL to Cell 1 of the notebook so pixel coordinates match pupil_x / pupil_y ----
CROP_SCALE = 2.5
MIN_CROP_FRAC = 0.4
N_SAMPLES = 60
WAVELET, LEVELS = "db2", 3
ANALYSIS_W = 256

# ---- labelling set ----
N_LABEL = 50
MIN_GAP = 5            # chosen frames are at least this many frames apart (avoids near-duplicate frames)
SEED = 0               # fixed -> re-running gives the same frames (don't change settings after you start labelling)

OUT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CROPPED = OUT / "cropped_eye.mp4"
LABEL_DIR = OUT / "label_frames"


# =====================  1) open video (ffmpeg fallback for odd codecs)  =====================
def _opens(p):
    c = cv2.VideoCapture(str(p)); ok = c.isOpened() and c.read()[0]; c.release(); return ok

SRC_VIDEO = VIDEO
if not _opens(SRC_VIDEO):
    try:
        import imageio_ffmpeg
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"], check=True)
        import imageio_ffmpeg
    _ff = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    _conv = OUT / "video_converted.mp4"
    subprocess.run([_ff, "-y", "-i", str(VIDEO), "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(_conv)],
                   capture_output=True)
    SRC_VIDEO = str(_conv)
    if not _opens(SRC_VIDEO):
        raise RuntimeError(f"Cannot open video: {VIDEO}")

cap = cv2.VideoCapture(str(SRC_VIDEO))
W, H = int(cap.get(3)), int(cap.get(4))
FPS_IN = cap.get(cv2.CAP_PROP_FPS) or 30.0
N_IN = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
want = set(np.linspace(0, max(N_IN - 1, 0), N_SAMPLES).astype(int).tolist())
samples, sample_idx, i = [], [], 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i in want: samples.append(f); sample_idx.append(i)
    i += 1
cap.release(); N_IN = i
print(f"Video: {W}x{H}, {FPS_IN:.1f} fps, {N_IN} frames; analysing {len(samples)} sample frames")


# =====================  2) wavelet eye finder (same as Cell 1)  =====================
AW = min(ANALYSIS_W, W); AH = int(round(H * AW / W)); SC = W / AW

def eye_score_map(bgr):
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (AW, AH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    coeffs = pywt.wavedec2(g, WAVELET, level=LEVELS)
    energy = np.zeros((AH, AW), np.float32)
    for (cH, cV, cD) in coeffs[1:]:
        e = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2).astype(np.float32)
        energy += cv2.resize(e, (AW, AH), interpolation=cv2.INTER_LINEAR)
    energy = cv2.GaussianBlur(energy, (0, 0), max(AH, AW) / 40)
    energy /= energy.max() + 1e-9
    a = cv2.resize(coeffs[0].astype(np.float32), (AW, AH), interpolation=cv2.INTER_LINEAR)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    s = (1.0 - a) * energy
    m = int(0.05 * min(AH, AW))
    s[:m, :] = 0; s[-m:, :] = 0; s[:, :m] = 0; s[:, -m:] = 0
    return s / (s.max() + 1e-9)

heat_map = cv2.GaussianBlur(np.mean([eye_score_map(f) for f in samples], axis=0), (0, 0), max(AH, AW) / 50)
py, px = np.unravel_index(np.argmax(heat_map), heat_map.shape)
mask = (heat_map >= 0.5 * heat_map.max()).astype(np.uint8)
_, lab = cv2.connectedComponents(mask)
ys, xs = np.nonzero(lab == lab[py, px])
ew = max(xs.max() - xs.min() + 1, (ys.max() - ys.min() + 1) * 4 / 3)
cx, cy, ew = px * SC, py * SC, ew * SC
print(f"Wavelet: eye centre ~ ({cx:.0f}, {cy:.0f}) px   eye width ~ {ew:.0f} px")

k_max = int(min(W // 8, H // 6))
k = int(np.clip(round(CROP_SCALE * ew / 8), max(1, int(MIN_CROP_FRAC * k_max)), k_max))
CW, CH = 8 * k, 6 * k
X0 = int(np.clip(round(cx - CW / 2), 0, W - CW))
Y0 = int(np.clip(round(cy - CH / 2), 0, H - CH))
print(f"Crop box (x0, y0, w, h) = ({X0}, {Y0}, {CW}, {CH}) of the {W}x{H} frame")

# preview with crop box on 3 sample frames
sel3 = np.linspace(0, len(samples) - 1, min(3, len(samples))).astype(int)
tiles = []
for s_i in sel3:
    fr = samples[s_i].copy()
    cv2.rectangle(fr, (X0, Y0), (X0 + CW, Y0 + CH), (0, 255, 0), max(2, W // 200))
    tiles.append(cv2.resize(fr, (int(W * 300 / H), 300)))
cv2.imwrite(str(OUT / "crop_preview.png"), np.hstack(tiles))
print(f"Saved {OUT / 'crop_preview.png'}  -> check the pupil is inside the green box in ALL frames "
      f"(if not, raise CROP_SCALE and rerun)")


# =====================  3) write cropped video + cheap per-frame proxies for frame selection  =====================
# Proxies are NOT the model's output (that would need EllSeg); they only help pick diverse frames:
#   cx_proxy  : x of the centroid of the darkest ~3% of pixels (roughly where the pupil is, left <-> right)
#   motion    : mean abs difference to the previous frame (quick phases / motion blur)
#   dark_min  : intensity of the darkest blurred spot (high = pupil hidden/covered -> blink candidate)
cap = cv2.VideoCapture(str(SRC_VIDEO))
wr = cv2.VideoWriter(str(CROPPED), cv2.VideoWriter_fourcc(*"mp4v"), FPS_IN, (CW, CH))
SW, SH = 104, 78
cx_proxy, motion, dark_min = [], [], []
prev, n_w = None, 0
while True:
    ok, f = cap.read()
    if not ok: break
    c = f[Y0:Y0 + CH, X0:X0 + CW]
    wr.write(c); n_w += 1
    g = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (SW, SH), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    thr = np.percentile(g, 3)
    yy, xx = np.nonzero(g <= thr)
    cx_proxy.append(xx.mean() * CW / SW)
    dark_min.append(g.min())
    motion.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
    prev = g
cap.release(); wr.release()
cx_proxy, motion, dark_min = map(np.array, (cx_proxy, motion, dark_min))
n_fr = n_w
print(f"Cropped video: {CROPPED}  ({n_fr} frames, {CW}x{CH})")


# =====================  4) pick frames (hard cases + random), at least MIN_GAP apart  =====================
rng = np.random.default_rng(SEED)
pick = {}

def take(order, n, cat):
    got = 0
    for i in order:
        i = int(i)
        if got >= n: break
        if all(abs(i - j) >= MIN_GAP for j in pick):
            pick[i] = cat; got += 1

take(np.argsort(cx_proxy), 8, "extreme_left")
take(np.argsort(-cx_proxy), 8, "extreme_right")
take(np.argsort(-motion), 8, "fast_motion")
take(np.argsort(-dark_min), 8, "possible_blink")
take(rng.permutation(n_fr), N_LABEL - len(pick), "random")
print(f"Selected {len(pick)} frames")


# =====================  5) export clean PNGs from the CROPPED video (shuffled, category hidden)  =====================
shutil.rmtree(LABEL_DIR, ignore_errors=True)
LABEL_DIR.mkdir(parents=True)
sel = list(pick)
rng.shuffle(sel)
name_of = {f: f"{k:03d}.png" for k, f in enumerate(sel)}

cap = cv2.VideoCapture(str(CROPPED))
rows, i = [], 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if i in name_of:
        cv2.imwrite(str(LABEL_DIR / name_of[i]), fr)
        rows.append(dict(file=name_of[i], frame=i, category=pick[i]))
    i += 1
cap.release()

pd.DataFrame(rows).sort_values("file").to_csv(OUT / "label_index.csv", index=False)
json.dump(dict(video=str(VIDEO), width=W, height=H, fps=FPS_IN, n_frames=n_fr,
               crop_x0=X0, crop_y0=Y0, crop_w=CW, crop_h=CH, crop_scale=CROP_SCALE),
          open(OUT / "crop_info.json", "w"), indent=2)

print(f"\nExported {len(rows)} frames -> {LABEL_DIR}")
print(pd.DataFrame(rows).category.value_counts().to_string())
print("\nNext: run gt_step2_label_local.py from this same folder.")