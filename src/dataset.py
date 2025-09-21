import torch
import torchaudio
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from pathlib import Path

import librosa

from datasets import load_dataset
from typing import Union

from tokenizer import Tokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AudioException(Exception):
    pass


class AudioProcessing:
    def load_wav(self, audio_path: Union[str, Path]):

        raise NotImplementedError("Subclass must implement method load_wav")

    def amplitude_to_db(self, x, min_db=-100):
        clip_val = 10 ** (min_db / 20)
        return 20 * torch.log10(torch.clamp(x, min=clip_val))

    def db_to_amplitude(self, x):
        return 10 ** (x / 20)

    def normalize(self, x, min_db=-100, max_abs_val=4):

        x = (x - min_db) / -min_db
        x = 2 * max_abs_val * x - max_abs_val
        x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
        return x

    def denormalize(self, x, min_db=-100, max_abs_val=4):

        x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
        x = (x + max_abs_val) / (2 * max_abs_val)
        x = x * -min_db + min_db

        return x


class AudioMelConversions(AudioProcessing):
    def __init__(
        self,
        num_mels=80,
        sampling_rate=22050,
        n_fft=1024,
        window_size=1024,
        hop_size=256,
        fmin=0,
        fmax=8000,
        center=False,
        min_db=-100,
        max_scaled_abs=4,
    ):

        super(AudioMelConversions, self).__init__()

        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def load_wav(self, audio_path):

        if not Path(audio_path).exists or audio_path == None:
            raise AudioException("Path audio doesnt exist Please set the rigth path ")

        audio_path = Path(audio_path)

        if not audio_path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
            raise AudioException(f"Unsupported audio format: {audio_path.suffix}")

        try:

            audio, sr = torchaudio.load(audio_path)

            if sr != self.sampling_rate:

                audio = torchaudio.functional.resample(
                    waveform=audio, orig_freq=sr, new_freq=self.sampling_rate
                )

                sr = self.sampling_rate

            return audio.squeeze(0), sr

        except Exception as e:

            raise AudioException(f"Failed to load audio file {audio_path}: {str(e)}")

    def _get_spec2mel_proj(self):

        mel = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        return torch.from_numpy(mel)

    def audio2mel(self, audio, do_norm=False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(
            input=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=torch.hann_window(self.window_size).to(audio.device),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spectrogram = torch.abs(spectrogram)

        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)

        mel = self.amplitude_to_db(mel, self.min_db)

        if do_norm:

            mel = self.normalize(
                mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs
            )

        return mel

    def mel2audio(self, mel, do_denorm=False, griffin_lim_iters=60):

        if do_denorm:

            mel = self.denormalize(
                mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs
            )

        mel = self.db_to_amplitude(mel)

        spectrogram = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()

        audio = librosa.griffinlim(
            S=spectrogram,
            n_iter=griffin_lim_iters,
            hop_length=self.hop_size,
            win_length=self.window_size,
            n_fft=self.n_fft,
            window="hann",
        )

        audio *= 32767 / max(0.01, np.max(np.abs(audio)))

        audio = audio.astype(np.int16)

        return audio


def build_padding_mask(lengths):

    B = lengths.size(0)
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i] :] = 1

    return mask.bool()


class TTSDataset(Dataset):
    def __init__(
        self,
        name_or_path,
        sample_rate=22050,
        n_fft=1024,
        window_size=1024,
        hop_size=256,
        fmin=0,
        fmax=8000,
        num_mels=80,
        center=False,
        normalized=False,
        min_db=-100,
        max_scaled_abs=4,
        split="train",
    ):

        self.dataset = load_dataset(path=name_or_path, split=split)
        self.tokenizer = Tokenizer(path_or_name=name_or_path, sampling_rate=sample_rate)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels
        self.center = center
        self.normalized = normalized
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        # self.transcript_lengths = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]

        self.audio_proc = AudioMelConversions(
            num_mels=self.num_mels,
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            window_size=self.win_size,
            hop_size=self.hop_size,
            fmin=self.fmin,
            fmax=self.fmax,
            center=self.center,
            min_db=self.min_db,
            max_scaled_abs=self.max_scaled_abs,
        )

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        audio = self.dataset[idx]["audio"]["array"]

        transcript = self.dataset[idx]["transcription"]

        transcript_ids = self.tokenizer.encode(transcript)

        mel = self.audio_proc.audio2mel(audio, do_norm=True)

        return transcript, mel.squeeze(0), transcript_ids.squeeze(0)


if __name__ == "__main__":

    # audiotts = AudioMelConversions(
    #     sampling_rate=16000
    # )

    # audio, sr = audiotts.load_wav(audio_path="asr000.wav")

    # dataset = load_dataset("abdouaziiz/alffa" , split="train+validation+test")

    # audio=dataset[0]["audio"]["array"]

    # mel = audiotts.audio2mel(audio=audio)

    # print(mel.shape)

    ds = TTSDataset(name_or_path="abdouaziiz/alffa", split="train+validation+test")

    from torch.utils.data import DataLoader

    dl = DataLoader(dataset=ds, batch_size=1)

    data = next(iter(dl))

    print(data)
