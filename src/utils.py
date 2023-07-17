import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class Griffilin(nn.Module):
    """Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

    Args:
        specgram (Tensor): A magnitude-only STFT spectrogram of dimension `(..., freq, frames)`
            where freq is ``n_fft // 2 + 1``.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins
        hop_length (int): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        win_length (int): Window size. (Default: ``n_fft``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
        n_iter (int): Number of iteration for phase recovery process.
        momentum (float): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge.
        length (int or None): Array length of the expected output.
        rand_init (bool): Initializes phase randomly if True, to zero otherwise.

    Returns:
        Tensor: waveform of `(..., time)`, where time equals the ``length`` parameter if given.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
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
