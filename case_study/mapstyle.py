"""Map panels in the style of Figure 1: geographic graticule with degree-minute
labels on all sides, dashed grid, compass-rose north arrow, alternating
black/white scale bar in km, serif font."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon, Rectangle
from matplotlib import patheffects as pe
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "savefig.dpi": 300,
                     "mathtext.fontset": "stix"})
EXT = (91.30, 92.75, 20.70, 23.75)          # lon_min, lon_max, lat_min, lat_max
LAT0 = 22.2
DISTRICTS = [("Khagrachari", 91.93, 23.22), ("Chattogram", 91.70, 22.66), ("Rangamati", 92.33, 22.33),
             ("Bandarban", 92.30, 21.95), ("Cox's Bazar", 92.06, 21.30)]


def read_geo(path, res=0.0006, categorical=True):
    """Warp a raster to WGS 84 for display."""
    rs = Resampling.nearest if categorical else Resampling.bilinear
    with rasterio.open(path) as s:
        with WarpedVRT(s, crs="EPSG:4326", resampling=rs) as v:
            b = v.bounds
            w = int((b.right - b.left) / res)
            h = int((b.top - b.bottom) / res)
            a = v.read(1, out_shape=(h, w), resampling=Resampling.mode if categorical else Resampling.average)
            nd = v.nodata
    a = a.astype(float)
    if nd is not None and not (isinstance(nd, float) and math.isnan(nd)):
        a[a == nd] = np.nan
    return a, (b.left, b.right, b.bottom, b.top)


def dm(v, hemi):
    d = int(math.floor(abs(v) + 1e-9))
    m = int(round((abs(v) - d) * 60))
    if m == 60:
        d, m = d + 1, 0
    return f"{d}°{m}′{hemi}"


def frame(ax, left=True, right=True, top=True, bottom=True, dlon=0.5, dlat=0.5, fs=6.5):
    lo, hi, la, ha = EXT
    ax.set_xlim(lo, hi)
    ax.set_ylim(la, ha)
    ax.set_aspect(1 / math.cos(math.radians(LAT0)))
    xs = np.arange(math.ceil(lo / dlon) * dlon, hi, dlon)
    ys = np.arange(math.ceil(la / dlat) * dlat, ha, dlat)
    for x in xs:
        ax.axvline(x, color="#9a9a9a", lw=0.4, ls=(0, (4, 3)), zorder=0.5)
    for y in ys:
        ax.axhline(y, color="#9a9a9a", lw=0.4, ls=(0, (4, 3)), zorder=0.5)
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xticklabels([dm(x, "E") for x in xs], fontsize=fs)
    ax.set_yticklabels([dm(y, "N") for y in ys], fontsize=fs, rotation=90, va="center")
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=1.5,
                   left=left, labelleft=left, right=right, labelright=right,
                   top=top, labeltop=top, bottom=bottom, labelbottom=bottom)
    for t in ax.get_yticklabels():
        t.set_rotation(90)
    for s in ax.spines.values():
        s.set_linewidth(0.9)


def compass(ax, x, y, size=0.13, fs=6):
    """Eight-point compass rose (axes-fraction centre, size in lon degrees)."""
    lo, hi, la, ha = EXT
    cx, cy = lo + x * (hi - lo), la + y * (ha - la)
    k = 1 / math.cos(math.radians(LAT0))           # stretch lat so it looks round
    r, ri = size, size * 0.22
    for i in range(8):
        ang = math.radians(90 - 45 * i)
        rr = r if i % 2 == 0 else r * 0.6
        tip = (cx + rr * math.cos(ang), cy + rr * math.sin(ang) * k)
        a1, a2 = ang + math.radians(22.5), ang - math.radians(22.5)
        p1 = (cx + ri * math.cos(a1), cy + ri * math.sin(a1) * k)
        p2 = (cx + ri * math.cos(a2), cy + ri * math.sin(a2) * k)
        ax.add_patch(Polygon([(cx, cy), p1, tip], closed=True, fc="black", ec="black", lw=0.4, zorder=6))
        ax.add_patch(Polygon([(cx, cy), p2, tip], closed=True, fc="white", ec="black", lw=0.4, zorder=6))
    for lab, ang in (("N", 90), ("E", 0), ("S", 270), ("W", 180)):
        a = math.radians(ang)
        ax.text(cx + (r + 0.045) * math.cos(a), cy + (r + 0.045) * math.sin(a) * k, lab,
                ha="center", va="center", fontsize=fs, zorder=7)


def scalebar(ax, x, y, ticks=(0, 25, 50, 100), fs=5.5, h=0.022):
    """Alternating black/white bar; first segment split into quarters as in Figure 1."""
    lo, hi, la, ha = EXT
    x0, y0 = lo + x * (hi - lo), la + y * (ha - la)
    deg_per_km = 1 / (111.32 * math.cos(math.radians(y0)))
    edges = list(ticks)
    segs = []
    q = edges[1] / 2
    segs += [(0, q), (q, edges[1])]
    segs += list(zip(edges[1:-1], edges[2:]))
    for i, (a, b) in enumerate(segs):
        ax.add_patch(Rectangle((x0 + a * deg_per_km, y0), (b - a) * deg_per_km, h,
                               fc="black" if i % 2 == 0 else "white", ec="black", lw=0.5, zorder=6))
    for t in ticks:
        ax.text(x0 + t * deg_per_km, y0 + h + 0.03, str(t), ha="center", va="bottom", fontsize=fs, zorder=7)
    ax.text(x0 + ticks[-1] * deg_per_km + 0.03, y0 + h / 2, "Km", ha="left", va="center", fontsize=fs, zorder=7)


def outline(ax, mask_path):
    a, ext = read_geo(mask_path, res=0.003, categorical=True)
    m = np.nan_to_num(a) > 0
    ax.contour(np.linspace(ext[0], ext[1], m.shape[1]), np.linspace(ext[3], ext[2], m.shape[0]), m.astype(float),
               levels=[0.5], colors="black", linewidths=0.45, zorder=4)


def labels(ax, fs=6):
    for name, x, y in DISTRICTS:
        ax.text(x, y, name, ha="center", va="center", fontsize=fs, weight="bold", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    ax.annotate("Kaptai\nLake", xy=(92.17, 22.78), xytext=(92.52, 22.92), fontsize=5.5, style="italic",
                color="#1f4e9a", ha="center", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#1f4e9a"),
                path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
