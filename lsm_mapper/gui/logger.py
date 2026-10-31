import sys
import os
import traceback
from datetime import datetime

def setup_crash_logger():
    log_path = os.path.join(
        os.path.dirname(sys.executable)
        if getattr(sys, 'frozen', False)
        else os.path.dirname(os.path.abspath(__file__)),
        "crash_log.txt")

    def handle_exception(exc_type, exc_value, exc_tb):
        with open(log_path, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"CRASH: {datetime.now()}\n")
            f.write("".join(traceback.format_exception(
                exc_type, exc_value, exc_tb)))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = handle_exception
    return log_path