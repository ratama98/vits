import os
import argparse
import torch
import numpy as np
from scipy.io.wavfile import write
from utils import get_hparams_from_file, load_filepaths_and_text
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def load_model(checkpoint_path, config_path):
    hps = get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    utils.load_checkpoint(checkpoint_path, net_g, None)
    return net_g, hps

def synthesize_audio(text, net_g, hps, output_path):
    cleaned_sequence = []
    for symbol in text:
        try:
            # Tambahkan simbol jika valid
            symbol_id = _symbol_to_id[symbol]
            cleaned_sequence.append(symbol_id)
        except KeyError:
            print(f"[WARNING] Ignoring unrecognized symbol: {symbol}")

    # Konversi cleaned_sequence ke tensor
    stn_tst = torch.LongTensor(cleaned_sequence).unsqueeze(0).cuda()
    stn_tst_lengths = torch.LongTensor([stn_tst.size(1)]).cuda()

    with torch.no_grad():
        audio = net_g.infer(
            stn_tst,
            stn_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1
        )[0][0, 0].data.cpu().float().numpy()

    write(output_path, hps.data.sampling_rate, (audio * 32767).astype(np.int16))
    print(f"Saved synthesized audio to {output_path}")

def inference_and_save(filelist, checkpoint_path, config_path, output_folder):
    net_g, hps = load_model(checkpoint_path, config_path)

    os.makedirs(output_folder, exist_ok=True)

    filepaths_and_text = load_filepaths_and_text(filelist)

    for filepath, text in filepaths_and_text:
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_folder, filename)

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
