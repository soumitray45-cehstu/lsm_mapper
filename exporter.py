import os
import shutil

def export_results(lsm_tif, report_text, auc, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, "accuracy_report.txt")
    with open(report_path, "w") as f:
        f.write(f"AUC-ROC Score: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
    print(f"Report saved to {report_path}")

    lsm_copy = os.path.join(out_dir, "LSM_output.tif")
    shutil.copy(lsm_tif, lsm_copy)
    print(f"LSM raster copied to {lsm_copy}")