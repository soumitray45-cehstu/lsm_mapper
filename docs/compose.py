"""Compose the walkthrough video (1920x1080, 30 fps) from captured GUI frames."""
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

FR = "/home/lsm/demo/frames"
OUT = "/home/lsm/demo/video"
os.makedirs(OUT, exist_ok=True)
W, H = 1920, 1080
BG = (24, 26, 30)
F_BIG = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
F_MID = ImageFont.truetype("DejaVuSans.ttf", 32)
F_CAP = ImageFont.truetype("DejaVuSans.ttf", 30)
F_SMALL = ImageFont.truetype("DejaVuSans.ttf", 24)


def card(lines, path):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 330
    for txt, font, col in lines:
        tw = d.textlength(txt, font=font)
        d.text(((W - tw) / 2, y), txt, font=font, fill=col)
        y += font.size + 28
    im.save(path)


def framed(src, caption, path):
    im = Image.new("RGB", (W, H), BG)
    s = Image.open(src).convert("RGB")
    scale = min(1600 / s.width, 940 / s.height)
    s = s.resize((int(s.width * scale), int(s.height * scale)), Image.LANCZOS)
    x = (W - s.width) // 2
    im.paste(s, (x, 28))
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 92, W, H], fill=(14, 15, 18))
    tw = d.textlength(caption, font=F_CAP)
    d.text(((W - tw) / 2, H - 70), caption, font=F_CAP, fill=(235, 235, 235))
    im.save(path)


shots = json.load(open(os.path.join(FR, "shots.json")))
seq = []
card([("LSM Mapper 2.0", F_BIG, (255, 255, 255)),
      ("Landslide susceptibility mapping without a GIS", F_MID, (200, 200, 200)),
      ("Walkthrough of all stages on a sample dataset (Chattogram, Bangladesh)", F_SMALL, (160, 160, 160)),
      ("Supplementary video S1", F_SMALL, (120, 160, 220))], f"{OUT}/c000.png")
seq.append((f"{OUT}/c000.png", 4.0))
for i, s in enumerate(shots):
    p = f"{OUT}/f{i:03d}.png"
    framed(s["file"], s["caption"], p)
    seq.append((p, s["sec"]))
extra = [("/home/lsm/demo/map.png", "Interactive web map: one switchable layer per method (OpenStreetMap basemap appears when online)", 5.0),
         ("/home/lsm/demo/outputs.png", "Outputs: GeoTIFFs, tables, figures, models and a reproducible run report", 6.0)]
for j, (src, cap, sec) in enumerate(extra):
    if os.path.exists(src):
        p = f"{OUT}/e{j}.png"
        framed(src, cap, p)
        seq.append((p, sec))
card([("Free and open source (MIT License)", F_MID, (255, 255, 255)),
      ("github.com/soumitray45-cehstu/lsm_mapper", F_MID, (120, 170, 230)),
      ("Windows executable on the Releases page  ·  archived on Zenodo", F_SMALL, (180, 180, 180))],
     f"{OUT}/c999.png")
seq.append((f"{OUT}/c999.png", 5.0))
with open(f"{OUT}/list.txt", "w") as f:
    for p, sec in seq:
        f.write(f"file '{p}'\nduration {sec}\n")
    f.write(f"file '{seq[-1][0]}'\n")
tmp = f"{OUT}/LSM_Mapper_walkthrough.tmp.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{OUT}/list.txt",
                "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-movflags", "+faststart", tmp], check=True)
os.replace(tmp, f"{OUT}/LSM_Mapper_walkthrough.mp4")
print("total s", sum(s for _, s in seq))
