import os
import argparse
import torch
import numpy as np
from scipy.io.wavfile import write
from utils import get_hparams_from_file, load_filepaths_and_text
from models import SynthesizerTrn
from text import text_to_sequence

# Fungsi untuk memuat model
def load_model(checkpoint_path, config_path):
    hps = get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    utils.load_checkpoint(checkpoint_path, net_g, None)
    return net_g, hps

# Fungsi untuk melakukan inference
def synthesize_audio(text, net_g, hps, output_path):
    stn_tst = text_to_sequence(text, hps.data.text_cleaners)
    stn_tst = torch.LongTensor(stn_tst).unsqueeze(0).cuda()
    stn_tst_lengths = torch.LongTensor([stn_tst.size(1)]).cuda()

    with torch.no_grad():
        audio = net_g.infer(
            stn_tst,
            stn_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1
        )[0][0, 0].data.cpu().float().numpy()

    # Simpan audio ke file .wav
    write(output_path, hps.data.sampling_rate, (audio * 32767).astype(np.int16))
    print(f"Saved synthesized audio to {output_path}")

# Fungsi utama untuk melakukan inference pada seluruh file teks
def inference_and_save(filelist, checkpoint_path, config_path, output_folder):
    # Load model dan hyperparameters
    net_g, hps = load_model(checkpoint_path, config_path)

    # Pastikan folder output ada
    os.makedirs(output_folder, exist_ok=True)

    # Load teks dari filelist
    filepaths_and_text = load_filepaths_and_text(filelist)

    # Lakukan inference untuk setiap teks dan simpan hasilnya
    for idx, (filepath, text) in enumerate(filepaths_and_text):
        output_path = os.path.join(output_folder, f"output_{idx + 1}.wav")
        synthesize_audio(text, net_g, hps, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform TTS inference and save audio")
    parser.add_argument("--filelist", type=str, required=True, help="Path to text filelist")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config JSON")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save synthesized audio")

    args = parser.parse_args()

    inference_and_save(
        filelist=args.filelist,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_folder=args.output_folder
    )
