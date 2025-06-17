# import os

# # Paths
# root_dir = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\dev"
# txt_dir = os.path.join(root_dir, "txt")
# wav_dir = os.path.join(root_dir, "wavs")
# output_path = os.path.join(root_dir, "dev.txt")

# # Construim liniile pentru train.txt
# lines = []

# for file in sorted(os.listdir(txt_dir)):
#     if not file.endswith(".txt"):
#         continue

#     file_id = os.path.splitext(file)[0]  # SPKxx_...
#     speaker_name = "_".join(file_id.split("_")[:4])  # SPK01_male_cv_ro (4 bucăți)
    
#     # Căi relative
#     wav_rel_path = f"wavs/{speaker_name}/{file_id}.wav"
#     transcript_path = os.path.join(txt_dir, file)

#     # Citim textul
#     with open(transcript_path, "r", encoding="utf-8") as f:
#         text = f.read().strip()

#     # Adăugăm linia
#     lines.append(f"{wav_rel_path}|{text}|{speaker_name}")

# # Scriem fișierul train.txt
# with open(output_path, "w", encoding="utf-8") as f_out:
#     f_out.write("\n".join(lines))

# print(f"train.txt generat cu {len(lines)} linii.")


import os

# Setări
wav_folder = "wav48_silence_trimmed"  # ← modifică în "wavs_train" pentru train.txt
wav_subfolder = "wavs"
root_dir = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\train"
txt_dir = os.path.join(root_dir, "txt_train")
wav_dir = os.path.join(root_dir, "wav48_silence_trimmed")
output_path = os.path.join(root_dir, "metadata_final.txt")

# Construim liniile
lines = []

for file in sorted(os.listdir(txt_dir)):
    if not file.endswith(".txt"):
        continue

    file_id = os.path.splitext(file)[0]  # ex: SPK18_female_rss_ro_00123.txt
    speaker_name = "_".join(file_id.split("_")[:4])

    # ← calea relativă corectă spre fișierul audio
    wav_rel_path = f"{wav_folder}/{speaker_name}/{file_id}.flac"
    transcript_path = os.path.join(txt_dir, file)

    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    lines.append(f"{wav_rel_path}|{text}|{speaker_name}")

# Scriem fișierul de ieșire
with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write("\n".join(lines))

print(f"{os.path.basename(output_path)} generat cu {len(lines)} linii.")
