import torch.nn as nn
import torchaudio.functional as F
from torch import Tensor
from typing import Optional
import torch
import math
import matplotlib.pyplot as plt


class Griffilin(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        specgram: Tensor,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: float,
        n_iter: int,
        momentum: float,
        length: Optional[int],
        rand_init: bool,
    ):

        return F.griffinlim(
            specgram,
            window,
            n_fft,
            hop_length,
            win_length,
            power,
            n_iter,
            momentum,
            length,
            rand_init,
        )


""" 
if __name__=="__main__":

    grid = Griffilin()
    specgram = torch.randn(1, 1025, 400)
    window = torch.hann_window(400)
    n_fft = 1024
    hop_length = 256
    win_length = 400
    power = 2
    n_iter = 32
    momentum = 0.99
    length = 400
    rand_init = True

    waveform = grid(specgram, window, n_fft, hop_length, win_length, power, n_iter, momentum, length, rand_init)

    print(waveform.shape)

    plt.plot(waveform.squeeze().numpy()) """

