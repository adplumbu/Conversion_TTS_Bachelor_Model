import os
import shutil

# Paths originale
transcript_dir = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\txt\SPK18_female_rss_ro_transcripts"
audio_dir = r"F:\LICENTA2025\BachelorWorkspace\dataset\cv-corpus-21.0-2025-03-14-ro.tar\cv-corpus-21.0-2025-03-14-ro\cv-corpus-21.0-2025-03-14\ro\RO_VCTK\wavs\SPK18_female_rss_ro"

# Path-uri noi (copii redenumite)
new_transcript_dir = os.path.join(os.path.dirname(transcript_dir), "SPK18_female_rss_ro_transcripts_renamed")
new_audio_dir = os.path.join(os.path.dirname(audio_dir), "SPK18_female_rss_ro_audio_renamed")

# Creare directoare noi dacă nu există
os.makedirs(new_transcript_dir, exist_ok=True)
os.makedirs(new_audio_dir, exist_ok=True)

# Listă fișiere
transcripts = sorted([f for f in os.listdir(transcript_dir) if f.endswith(".txt")])
audios = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

# Verificare egalitate
if len(transcripts) != len(audios):
    raise ValueError(f"Different number of transcript and audio files: {len(transcripts)} vs {len(audios)}")

# Copiere cu redenumire
for idx, (txt_file, wav_file) in enumerate(zip(transcripts, audios), start=1):
    id_str = f"{idx:05d}"
    base_name = f"SPK18_female_rss_ro_{id_str}"

    # Paths vechi
    old_txt = os.path.join(transcript_dir, txt_file)
    old_wav = os.path.join(audio_dir, wav_file)

    # Paths noi
    new_txt = os.path.join(new_transcript_dir, base_name + ".txt")
    new_wav = os.path.join(new_audio_dir, base_name + ".wav")

    # Copiere cu pastrare metadate
    shutil.copy2(old_txt, new_txt)
    shutil.copy2(old_wav, new_wav)

print("Copiere si redenumire completa in foldere noi.")
