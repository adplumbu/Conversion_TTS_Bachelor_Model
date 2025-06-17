import os
import librosa
import soundfile as sf

# ==== CONFIG ====
target_sr = 22050
folder = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\wavs_train\SPK18_female_rss_ro"  # ← înlocuiește cu folderul tău
# ===============

count = 0
for file in os.listdir(folder):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(folder, file)

    try:
        audio, sr = librosa.load(path, sr=None)

        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sf.write(path, audio_resampled, target_sr)
            print(f"{file}: {sr} Hz -> {target_sr} Hz")
            count += 1
        else:
            print(f"{file}: deja la {target_sr} Hz")

    except Exception as e:
        print(f"Eroare la {file}: {str(e)}")

print(f"\nTotal fisiere resamplate: {count}")
