import argparse
import librosa
import numpy as np
from utils import load_filepaths_and_text

def calculate_mcd13(ref_audio_path, syn_audio_path, sr=22050, n_mfcc=13):
    ref_audio, _ = librosa.load(ref_audio_path, sr=sr)
    syn_audio, _ = librosa.load(syn_audio_path, sr=sr)

    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
    syn_mfcc = librosa.feature.mfcc(y=syn_audio, sr=sr, n_mfcc=n_mfcc)

    min_length = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    ref_mfcc, syn_mfcc = ref_mfcc[:, :min_length], syn_mfcc[:, :min_length]

    diff = ref_mfcc - syn_mfcc
    K = 10 / np.log(10) * np.sqrt(2)  
    mcd13 = K * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))

    return mcd13

def evaluate_mcd13(filelist, ref_folder, syn_folder, sr=22050, n_mfcc=13):
    filepaths_and_text = load_filepaths_and_text(filelist)

    mcd_scores = []
    for entry in filepaths_and_text:
        audio_id = entry[0].split('/')[-1]  
        ref_audio_path = f"{ref_folder}/{audio_id}"
        syn_audio_path = f"{syn_folder}/{audio_id}"

        try:
            mcd_score = calculate_mcd13(ref_audio_path, syn_audio_path, sr=sr, n_mfcc=n_mfcc)
            mcd_scores.append(mcd_score)
            print(f"{audio_id}: MCD13 = {mcd_score:.2f} dB")
        except Exception as e:
            print(f"Error processing {audio_id}: {e}")

    avg_mcd13 = np.mean(mcd_scores)
    print(f"\nAverage MCD13: {avg_mcd13:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MCD13 for TTS model")
    parser.add_argument("--filelist", type=str, required=True, help="Path to test filelist (e.g., test.txt)")
    parser.add_argument("--ref_folder", type=str, required=True, help="Folder containing reference audio files")
    parser.add_argument("--syn_folder", type=str, required=True, help="Folder containing synthesized audio files")
    parser.add_argument("--sr", type=int, default=22050, help="Sampling rate for audio files")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCCs to extract")

    args = parser.parse_args()

    evaluate_mcd13(
        filelist=args.filelist,
        ref_folder=args.ref_folder,
        syn_folder=args.syn_folder,
        sr=args.sr,
        n_mfcc=args.n_mfcc,
    )
