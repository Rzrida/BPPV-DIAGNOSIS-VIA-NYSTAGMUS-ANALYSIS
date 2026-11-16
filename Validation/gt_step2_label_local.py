# ===== GT STEP 2: click the pupil centre on each frame (run LOCALLY on your PC) =====
# Setup:  pip install opencv-python
#         unzip label_frames.zip so you have a folder "label_frames/" next to this script
# Run:    python gt_step2_label_local.py
#
# Controls:  left-click = place centre (click again to move it)
#            Enter/Space = accept    s = skip (pupil not visible / fully closed)    q = quit (progress is saved, rerun to resume)
#
# Rule: mark the centre of the WHOLE pupil as you judge it, even if the lid covers part of it.
import cv2, glob, os, csv

ZOOM = 3                       # lower to 2 if the window is bigger than your screen
FOLDER, OUT = "label_frames", "labels.csv"

done = set()
if os.path.exists(OUT):
    done = {r["file"] for r in csv.DictReader(open(OUT))}
else:
    open(OUT, "w").write("file,x,y\n")

files = [f for f in sorted(glob.glob(os.path.join(FOLDER, "*.png"))) if os.path.basename(f) not in done]
pt = []

def on_click(ev, x, y, flags, param):
    if ev == cv2.EVENT_LBUTTONDOWN:   # back to original-pixel coords (pixel centres at integers)
        pt[:] = [(x + 0.5) / ZOOM - 0.5, (y + 0.5) / ZOOM - 0.5]

cv2.namedWindow("label")
cv2.setMouseCallback("label", on_click)

for k, f in enumerate(files):
    name = os.path.basename(f)
    img = cv2.imread(f)
    big = cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    pt.clear()
    while True:
        view = big.copy()
        if pt:
            cx, cy = int(round((pt[0] + 0.5) * ZOOM - 0.5)), int(round((pt[1] + 0.5) * ZOOM - 0.5))
            cv2.drawMarker(view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.putText(view, f"{k + 1}/{len(files)}  click=centre  Enter=accept  s=skip  q=quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("label", view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and pt:
            open(OUT, "a").write(f"{name},{pt[0]:.2f},{pt[1]:.2f}\n")
            break
        if key == ord("s"):
            open(OUT, "a").write(f"{name},nan,nan\n")
            break
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit

cv2.destroyAllWindows()
print("Done ->", OUT)
