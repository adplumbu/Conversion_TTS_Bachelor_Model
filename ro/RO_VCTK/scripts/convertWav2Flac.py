import os
from pydub import AudioSegment

# === CONFIG ===
input_root = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\train\wav48_silence_trimmed"               # WAV source
output_root = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\train\wav48_silence_trimmed_flac"        # FLAC dest

converted_count = 0

for speaker_folder in os.listdir(input_root):
    speaker_input_path = os.path.join(input_root, speaker_folder)
    speaker_output_path = os.path.join(output_root, speaker_folder)

    if not os.path.isdir(speaker_input_path):
        continue

    os.makedirs(speaker_output_path, exist_ok=True)

    for file in os.listdir(speaker_input_path):
        if file.endswith(".wav") and "_mic1" not in file:
            wav_path = os.path.join(speaker_input_path, file)
            base_name = os.path.splitext(file)[0]
            new_filename = base_name + "_mic1.flac"
            flac_path = os.path.join(speaker_output_path, new_filename)

            try:
                audio = AudioSegment.from_wav(wav_path)
                audio.export(flac_path, format="flac")
                print(f"{file} -> {new_filename}")
                converted_count += 1
            except Exception as e:
                print(f"Eroare la {file}: {e}")

print(f"\nTotal fișiere convertite: {converted_count}")
