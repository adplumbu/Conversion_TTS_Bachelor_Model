# import os
# import sys
# from TTS.utils.synthesizer import Synthesizer

# # Setează codificarea corectă pentru consolă și I/O
# if sys.stdout.encoding != 'utf-8':
#     sys.stdout.reconfigure(encoding='utf-8')

# # === CONFIG ===
# MODEL_PATH = "/root/Conversion_TTS_Bachelor_Model/model/vits_romanian_full_ds_run-June-17-2025_08+17PM-58125cf9/best_model.pth"
# CONFIG_PATH = "/root/Conversion_TTS_Bachelor_Model/model/vits_romanian_full_ds_run-June-17-2025_08+17PM-58125cf9/config.json"
# OUTPUT_PATH_WAV = "/root/Conversion_TTS_Bachelor_Model/synth_output/audio"
# OUTPUT_PATH_TEXT = "/root/Conversion_TTS_Bachelor_Model/synth_output/transcripts"
# SPEAKER_NAME = "VCTK_SPK01_male_cv_ro"  # <<--- modifică aici speaker-ul dorit

# # Creează directoarele dacă nu există
# os.makedirs(OUTPUT_PATH_WAV, exist_ok=True)
# os.makedirs(OUTPUT_PATH_TEXT, exist_ok=True)

# # Inițializează sintetizatorul
# synthesizer = Synthesizer(
#     model_path=MODEL_PATH,
#     config_path=CONFIG_PATH,
#     speakers_file_path=None,
#     use_cuda=True  # Setează True pentru GPU
# )

# # === TEXTE DE TESTAT ===
# test_texts = [
#     "Acum noi vorbim doar din auzite, din ce am citit pe internet.",
#     "Frâna bruscă a dus la răsturnarea mașinii în afara părţii carosabile.",
#     "El a adăugat că nu știe exact când se vor termina lucrările."
# ]

# # Normalizează textul

# def normalize_romanian_text(text):
#     replacements = {
#         'ş': 'ș', 'Ş': 'Ș',
#         'ţ': 'ț', 'Ţ': 'Ț',
#         '\u0219': 'ș', '\u0218': 'Ș',
#         '\u021B': 'ț', '\u021A': 'Ț',
#     }
#     for old, new in replacements.items():
#         text = text.replace(old, new)
#     return text

# # === SINTEZĂ AUDIO ===
# for i, text in enumerate(test_texts):
#     normalized_text = normalize_romanian_text(text)
#     print(f"[INFO] Sintetizez: {normalized_text}")

#     try:
#         wav = synthesizer.tts(normalized_text, speaker_name=SPEAKER_NAME)
#         wav_path = os.path.join(OUTPUT_PATH_WAV, f"test_{i+1}.wav")
#         txt_path = os.path.join(OUTPUT_PATH_TEXT, f"test_{i+1}.txt")

#         synthesizer.save_wav(wav, wav_path)
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write(normalized_text)

#         print(f"[SALVAT] {wav_path} & {txt_path}")

#     except Exception as e:
#         print(f"[EROARE] La textul {i+1}: {e}")

# print("\n[SUCCESS] Sinteza a fost completă!")

import os
import sys
from TTS.utils.synthesizer import Synthesizer

# Asigură-te că terminalul folosește UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# === Setări principale ===
MODEL_PATH = r"F:\LICENTA2025\BachelorWorkspace\tts_vits_romanian\model\runpod_training\vits_romanian_full_ds_run-June-17-2025_08+17PM-58125cf9\best_model_31152.pth"
CONFIG_PATH = r"F:\LICENTA2025\BachelorWorkspace\tts_vits_romanian\model\runpod_training\vits_romanian_full_ds_run-June-17-2025_08+17PM-58125cf9\config.json"
OUTPUT_PATH_WAV = r"F:\LICENTA2025\BachelorWorkspace\tts_vits_romanian\model\synthesis_output"
SPEAKER_NAME = "VCTK_SPK01_male_cv_ro"  # înlocuiește cu speakerul dorit

# Creează folderul de output dacă nu există
os.makedirs(OUTPUT_PATH_WAV, exist_ok=True)

# Inițializează sintetizatorul
synthesizer = Synthesizer(
    MODEL_PATH,
    CONFIG_PATH,
    use_cuda=False,  # setează pe True dacă rulezi pe GPU
tts_speakers_file=r"F:\LICENTA2025\BachelorWorkspace\tts_vits_romanian\model\runpod_training\vits_romanian_full_ds_run-June-17-2025_08+17PM-58125cf9\speakers.pth"
)

# Lista de propoziții de test
test_sentences = [
    "Totuşi, pare să fie atras de lumina reflectoarelor.",
    "Ulterior, a devenit profesor de instrumente tradiţionale.",
    "O altă regulă este că trebuie să descrii o scenă din natură.",
    "Frâna bruscă a dus la răsturnarea maşinii.",
    "Nu prea avea timp pentru el însuși.",
]

# Generează audio pentru fiecare propoziție
for i, text in enumerate(test_sentences, start=1):
    try:
        print(f"[{i}] Sintetizez: {text}")
        outputs = synthesizer.tts(text, speaker_name=SPEAKER_NAME)

        wav_path = os.path.join(OUTPUT_PATH_WAV, f"test_output_{i}.wav")
        synthesizer.save_wav(outputs, wav_path)
        print(f"    -> Audio salvat la: {wav_path}")

    except Exception as e:
        print(f"    [Eroare] {e}")
        continue

print("Proces de sintetizare finalizat.")
