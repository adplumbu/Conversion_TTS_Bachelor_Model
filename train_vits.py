import sys
import io
import multiprocessing
import librosa
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
# from TTS.tts.configs.shared_configs import CharactersConfig
# from pathlib import Path


# Mută tot codul principal într-o funcție
def main():
    # output_path = os.path.dirname(os.path.abspath(__file__))
    output_path=r"/root/Conversion_TTS_Bachelor_Model/model"
    dataset_config = BaseDatasetConfig(
        formatter="vctk",
        language="ro",
        meta_file_train="metadata_final_fixed.csv",
        ignored_speakers=[
            "VCTK_SPK01_male_cv_ro",
            "VCTK_SPK06_male_cv_ro",
            "VCTK_SPK16_female_cv_ro"
        ],
        
        #meta_file_val="dev.csv",

    path=r"/root/Conversion_TTS_Bachelor_Model/ro/RO_VCTK/train"
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    vits_args = VitsArgs(use_speaker_embedding=True)

    config = VitsConfig(
        model_args=vits_args,
        audio=audio_config,
        model= "vits",
        run_name="vits_romanian_full_ds_run_characters",
        project_name="VITS Romanian Training CV-RSS",
        run_description="""
            - Training of VITS model in Romanian using Common Voice and Romanian Speech Synthesis Corpusesbatch_size
        """,
        dashboard_logger="tensorboard",
        batch_size=64,
        eval_batch_size=32,
        eval_split_max_size=1200,
        batch_group_size=64,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        epochs=350,
        print_step=50,
        plot_step=100,
        save_step=5000,
        log_model_step=1000,
        save_n_checkpoints= 2,
        save_checkpoints=True,
        run_eval=True,
        test_delay_epochs=-1,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        phonemizer="espeak",
        phoneme_language="ro",
        phoneme_cache_path=r"/root/Conversion_TTS_Bachelor_Model/ds_phonemes",
        compute_input_seq_cache=True,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=True,
        characters=CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
        pad= "<PAD>",
        eos= "<EOS>",
        bos= "<BOS>",
        blank= "<BLNK>",
        characters="AĂÂBCDEFGHIÎJKLMNOPQRSȘTȚUVWXYZaăâbcdefghiîjklmnopqrsșştțţuvwxyzüá",
        punctuations= '"\'?!., ',
        phonemes= "",
        is_unique=True,
        is_sorted=True,
        ),
        test_sentences=[
    [
        "Mica afacere a tatălui meu rămâne mică.",
        "VCTK_SPK01_male_cv_ro",
        None,
        "ro",
    ],
    [
        "Planificăm, de asemenea, să urmăm o abordare paralelă.",
        "VCTK_SPK08_female_cv_ro",
        None,
        "ro",
    ],
    [
        "Acum, negociază pentru achiziționarea unei a doua mori, afacerile luând amploare.",
        "VCTK_SPK15_male_cv_ro",
        None,
        "ro",
    ],
    [
        "Astăzi nu mai putem schimba acest fapt.",
        "VCTK_SPK18_female_rss_ro",
        None,
        "ro",
    ],
    [
        "Nu este o rușine să copiezi.",
        "VCTK_SPK17_male_cv_ro",
        None,
        "ro",
    ]
]
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # # DEBUG: verificăm ce linie din metadata provoacă eroarea de encoding
    # metadata_path = os.path.join(dataset_config.path, dataset_config.meta_file_train)
    # print(f"Verificăm encoding-ul în fișierul: {metadata_path}")
    # with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
    #     for i, line in enumerate(f):
    #         print(f"{i}: {line.strip()}")

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size = 0.1,
        #eval_split_size=config.eval_split_size,
    )

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.model_args.num_speakers = speaker_manager.num_speakers

    ###########Verify Dataset###########
    # După încărcarea datelor, afișează statistici
    print(f"Date incarcate: {len(train_samples)} esantioane pentru antrenare, {len(eval_samples)} pentru evaluare.")

    # Verifică durata totală a audio
    total_duration = 0
    for sample in train_samples + eval_samples:
        try:
            audio_path = os.path.join(dataset_config.path, sample["audio_file"])
            y, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
            total_duration += len(y) / sr
        except Exception as e:
            print(f"Eroare la procesarea {sample['audio_file']}: {str(e)}")

    print(f"Durata totala a audio: {total_duration/60:.2f} minute")

    # Verifică primele 3 fișiere pentru detalii
    for i, sample in enumerate(train_samples[:3]):
        try:
            audio_path = os.path.join(dataset_config.path, sample["audio_file"])
            y, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
            print(f"Fisier {i+1}: {sample['audio_file']}, Durata: {len(y)/sr:.2f}s, Sample rate: {sr}")
        except Exception as e:
            print(f"Eroare la fisierul {sample['audio_file']}: {str(e)}")

    # Verifică primele 3 transcripturi
    for i, sample in enumerate(train_samples[:3]):

            print(f"Transcript {i+1}: {sample['text']}")

    # pretrained_checkpoint = "path/catre/modelul_preantrenat.pth"

    ###########Verify Dataset###########

    model = Vits(config, ap, tokenizer, speaker_manager)


    trainer = Trainer(
        TrainerArgs(gpu=0),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        #dashboard_logger="TensorboardLogger"
    )

    trainer.fit()

# Acest bloc este cheia pentru a rezolva eroarea de multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
