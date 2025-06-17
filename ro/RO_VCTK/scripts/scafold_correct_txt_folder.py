import os
import shutil

# ==== CONFIG ====
root_dir = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\train"
txt_root = os.path.join(root_dir, "txt")
# ================

# 1. Iterăm prin toate fișierele .txt din txt_root
for file in os.listdir(txt_root):
    if not file.endswith(".txt"):
        continue

    file_id = os.path.splitext(file)[0]  # Ex: SPK01_male_cv_ro_00001
    speaker = "_".join(file_id.split("_")[:4])  # SPKxx_male_cv_ro

    src_path = os.path.join(txt_root, file)
    dest_dir = os.path.join(txt_root, speaker)
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, file)
    shutil.move(src_path, dest_path)
    print(f"Mutat: {file} -> {dest_path}")
