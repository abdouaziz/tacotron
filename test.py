# test the tacotron model 
from src import Tacotron



tacotron= Tacotron(
    n_vocab=32_000
)
print(tacotron())