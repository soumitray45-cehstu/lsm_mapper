"""LSM Mapper launcher.

    LSM_Mapper.exe                    -> desktop interface
    LSM_Mapper.exe settings.json      -> interface with saved settings loaded
    LSM_Mapper.exe --run settings.json -> batch run without the interface
"""
import os
import sys

if getattr(sys, "frozen", False):          # PyInstaller bundle: point GDAL/PROJ at bundled data
    base = sys._MEIPASS
    for var, sub in (("GDAL_DATA", "rasterio/gdal_data"), ("PROJ_DATA", "rasterio/proj_data"),
                     ("PROJ_LIB", "rasterio/proj_data")):
        p = os.path.join(base, sub)
        if os.path.isdir(p):
            os.environ[var] = p
    import tempfile
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "lsm_mapper_numba"))


def _frozen_parallel():
    """In the frozen executable, run joblib/scikit-learn parallel work in
    threads: loky worker processes would re-launch the executable itself."""
    if getattr(sys, "frozen", False):
        try:
            import joblib
            joblib.parallel_config(backend="threading")
        except Exception:
            os.environ["JOBLIB_MULTIPROCESSING"] = "0"


def _main():
    _frozen_parallel()
    if len(sys.argv) > 1 and sys.argv[1] in ("--run", "--version", "--selftest"):
        if sys.argv[1] == "--version":
            from lsm_mapper import __version__
            print(__version__)
            return 0
        if sys.argv[1] == "--selftest":
            from lsm_mapper.selftest import main as st
            return st()
        from lsm_mapper.cli import main as cli
        return cli(sys.argv[2:])
    from lsm_mapper.gui.logger import setup_crash_logger
    setup_crash_logger()
    from lsm_mapper.gui.app import main as gui
    return gui()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(_main())
