import os
import torch
import librosa
import sys
import io
import multiprocessing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from TTS.TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from torch.utils.tensorboard import SummaryWriter


# Adaugam o clasa de callback pentru dezghetarea graduala a straturilor
class UnfreezeCallback:
    def __init__(self, model, unfreeze_epoch=5):
        self.model = model
        self.unfreeze_epoch = unfreeze_epoch
        
    def on_epoch_end(self, trainer):
        if trainer.epoch == self.unfreeze_epoch:
            print(f"Epoch {trainer.epoch}: Dezghetarea straturilor de codare...")
            for param in self.model.text_encoder.parameters():
                param.requires_grad = True
            
            # Recalculeaza si afiseaza numarul parametrilor antrenabili
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Numar parametri antrenabili dupa dezghetare: {trainable_params:,}")

def freeze_encoder_layers(model, freeze=True):
    """Ingheata sau dezgheata straturile de codare text"""
    for name, param in model.text_encoder.named_parameters():
        param.requires_grad = not freeze
    
    # Calculeaza numarul de parametri antrenabili
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Straturi encoder {'inghetate' if freeze else 'dezghetate'}")
    print(f"Numar parametri antrenabili: {trainable_params:,}")

def main():
    # Seteaza calea catre modelul pre-antrenat
    # IMPORTANT: Inlocuieste cu calea reala catre modelul tau pre-antrenat in engleza
    pretrained_checkpoint = "path/to/english_vits_model.pth"
    
    output_path = os.path.dirname(os.path.abspath(__file__))

    # Configuratie set de date
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=r"F:\LICENTA2025\BachelorWorkspace\dataset\training"   
    )

    # Configuratie audio
    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    # Configuratie model pentru fine-tuning
    config = VitsConfig(
        audio=audio_config,
        run_name="vits_romanian_finetuned",
        run_description="Fine-tuning model VITS pre-antrenat pentru limba romana",
        batch_size=16,  # Batch mai mic pentru fine-tuning
        eval_batch_size=8,
        batch_group_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        epochs=50,  # Mai putine epoci pentru fine-tuning
        print_step=25,
        plot_step=100,
        save_step=250,  # Salveaza mai des pentru a urmari progresul
        save_n_checkpoints=10,
        run_eval=True,
        test_delay_epochs=1,
        text_cleaner="basic_cleaners",
        use_phonemes=False,  # Poti incerca si True cu phonemizer pentru romana
        compute_input_seq_cache=True,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=True,
        # Setam vocabularul pentru limba romana fara diacritice
        characters={
            "pad": "_",
            "eos": "~",
            "bos": "^",
            "characters": "abcdefghijklmnopqrstuvwxyz -'?!.,",
            "punctuations": "'?!.,",
            "phonemes": ""
        },
        test_sentences=[
            'Nu este treaba lor ce constitutie avem.',
            'Ea era tot timpul pe minge.',
            'Nicoara crede ca acest concurs va avea succes.',
            'Afganistanul va fi reprezentat la adunarea generala de ministrul de externe, a declarat un responsabil al misiunii.',
            'Evenimentul are ca scop facilitarea schimbului de idei privind viitorul securitatii energetice in aceste regiuni.',
            'La serviciu vin dimineata iar acasa ajung seara.'
        ],
        # Parametri optimizati pentru fine-tuning
        lr=1e-5,  # Rata de invatare mai mica pentru fine-tuning
        lr_scheduler="ExponentialLR",
        lr_scheduler_params={"gamma": 0.998},
        optimizer="AdamW",
        optimizer_params={"weight_decay": 0.01},
        scheduler_after_epoch=True,
        warmup_steps=1000,  # Crestere graduala a ratei de invatare
        grad_clip=1.0,  # Limitarea gradientilor pentru stabilitate
    )

    # Initializarea procesorului audio si a tokenizer-ului
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Incarcarea seturilor de date
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.1
    )

    # Verificare set de date
    print(f"Date incarcate: {len(train_samples)} esantioane pentru antrenare, {len(eval_samples)} pentru evaluare.")

    # Verifica durata totala audio
    total_duration = 0
    for sample in train_samples + eval_samples:
        try:
            audio_path = os.path.join(dataset_config.path, sample["audio_file"])
            y, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
            total_duration += len(y) / sr
        except Exception as e:
            print(f"Eroare la procesarea {sample['audio_file']}: {str(e)}")

    print(f"Durata totala audio: {total_duration/60:.2f} minute")

    # Verifica primele 3 fisiere pentru detalii
    for i, sample in enumerate(train_samples[:3]):
        try:
            audio_path = os.path.join(dataset_config.path, sample["audio_file"])
            y, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
            print(f"Fisier {i+1}: {sample['audio_file']}, Durata: {len(y)/sr:.2f}s, Sample rate: {sr}")
        except Exception as e:
            print(f"Eroare la fisierul {sample['audio_file']}: {str(e)}")

    # Verifica primele 3 transcripturi
    for i, sample in enumerate(train_samples[:3]):
        print(f"Transcript {i+1}: {sample['text']}")

    # Initializeaza modelul
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    # Imprima informatii despre tokenizer
    print(f"Tokenizer: {len(tokenizer.characters)} caractere")
    print(f"Caractere in tokenizer: {tokenizer.characters}")
    
    # Inghetam straturile de codare text la inceput
    freeze_encoder_layers(model, freeze=True)
    
    # Incarca modelul pre-antrenat daca exista
    if os.path.exists(pretrained_checkpoint):
        print(f"Incarcare model pre-antrenat din: {pretrained_checkpoint}")
        try:
            checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
            
            # Determina cheia corecta pentru starea modelului
            model_state_key = "model"
            if "model" not in checkpoint:
                if isinstance(checkpoint, dict) and any(k.startswith("text_encoder") for k in checkpoint.keys()):
                    model_state_key = ""  # Starea modelului este chiar checkpoint-ul
                else:
                    # Verificam alte chei posibile
                    for key in ["model_state", "state_dict", "generator"]:
                        if key in checkpoint:
                            model_state_key = key
                            break
            
            # Obtine dictionarul starii modelului
            if model_state_key:
                pretrained_dict = checkpoint[model_state_key]
            else:
                pretrained_dict = checkpoint
            
            # Filtreaza straturile care nu se potrivesc din cauza diferentelor de vocabular
            model_dict = model.state_dict()
            filtered_pretrained_dict = {}
            
            for k, v in pretrained_dict.items():
                # Ignora straturile de embedding care vor avea dimensiuni diferite 
                # din cauza diferentelor de vocabular intre engleza si romana
                if "text_encoder.embedding" in k:
                    print(f"Ignoram stratul: {k} (dimensiuni incompatibile din cauza vocabularului diferit)")
                    continue
                
                # Verifica daca dimensiunile se potrivesc
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_pretrained_dict[k] = v
                elif k in model_dict:
                    print(f"Ignoram stratul: {k} (dimensiuni incompatibile: {v.shape} vs {model_dict[k].shape})")
                else:
                    print(f"Stratul {k} nu exista in modelul nou")
            
            # Actualizeaza starea modelului cu parametrii pre-antrenati
            model_dict.update(filtered_pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            print(f"Model pre-antrenat incarcat cu succes! ({len(filtered_pretrained_dict)}/{len(pretrained_dict)} straturi)")
        except Exception as e:
            print(f"Eroare la incarcarea modelului pre-antrenat: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"ATENTIE: Fisierul de model pre-antrenat nu a fost gasit: {pretrained_checkpoint}")
        print("Se incepe antrenamentul de la zero!")

    # Configurarea si initializarea trainerului
    trainer = Trainer(
        TrainerArgs(gpu=None),  # Seteaza GPU-ul aici daca ai unul
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Adauga callback-ul pentru dezghetarea graduala
    unfreeze_callback = UnfreezeCallback(model, unfreeze_epoch=5)
    trainer.add_callback("on_epoch_end", unfreeze_callback.on_epoch_end)
    
    # Incepe antrenamentul
    trainer.fit()

# Ruleaza functia principala
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()