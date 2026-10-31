"""Command-line runner: the same engine as the GUI, driven by a JSON file.

    LSM_Mapper.exe --run config.json
    python -m lsm_mapper.cli config.json
"""
import argparse
import sys

from lsm_mapper.core.pipeline import Run, RunConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description="LSM Mapper batch run")
    ap.add_argument("config", help="run configuration (JSON)")
    ap.add_argument("--no-maps", action="store_true", help="skip map prediction")
    a = ap.parse_args(argv)
    cfg = RunConfig.load(a.config)
    if a.no_maps:
        cfg.make_maps = False
    Run(cfg, log=lambda s: print(s, flush=True)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
