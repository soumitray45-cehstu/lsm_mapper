# PyInstaller spec: pyinstaller packaging/LSM_Mapper.spec --noconfirm
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hidden = [], [], []
for pkg in ("rasterio", "pyogrio", "pyproj", "shapely", "geopandas", "xgboost",
            "jenkspy", "folium", "branca", "numba", "llvmlite"):
    d, b, h = collect_all(pkg, filter_submodules=lambda n: ".tests" not in n and ".testing" not in n)
    datas += d; binaries += b; hidden += h
hidden += collect_submodules("sklearn", filter=lambda n: ".tests" not in n) + collect_submodules("lsm_mapper")
hidden += ["scipy.special._cdflib", "PIL.Image"]

a = Analysis(["../main.py"], pathex=[".."], binaries=binaries, datas=datas,
             hiddenimports=hidden, excludes=["tkinter", "IPython", "pytest", "shap", "torch"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LSM_Mapper",
          console=False, icon=None)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LSM_Mapper_cli", console=True)
coll = COLLECT(exe, cli, a.binaries, a.datas, name="LSM_Mapper")
