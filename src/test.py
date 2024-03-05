import torch
import torchaudio
import IPython

from torchaudio import transforms
from IPython.display import Audio
import matplotlib.pyplot as plt


waveform, sample_rate = torchaudio.load("audio.wav", normalize=True)
transform = transforms.MelSpectrogram(sample_rate)
mel_specgram = transform(waveform) 

plt.figure()
plt.imshow(mel_specgram.log2()[0,:,:].numpy(), cmap='gray')
plt.show()


