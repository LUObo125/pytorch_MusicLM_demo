
import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from musiclm_pytorch.trainer import MuLaNTrainer

#from sklearn import preprocessing
device = 'cuda'
from torch.utils.data import Dataset
class TextAudioDataset(Dataset):
    def __init__(self, wavs, texts):
        super().__init__()
        self.wavs = wavs
        self.texts = texts

    def __len__(self):
        if len(self.wavs) != len(self.texts):
            return -1
        else:
            return len(self.wavs)

    def __getitem__(self, idx):
        return self.wavs[idx], self.texts[idx]
########################
#MuLaN
########################

wavs = torch.load('wavs_small.pt')
texts = torch.load('texts_small.pt')

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
).cuda()

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
).cuda()

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
).cuda()

trainer = MuLaNTrainer(
    mulan = mulan,
    dataset = TextAudioDataset(wavs, texts),
    num_train_steps = 50000,
    batch_size = 4,
    save_model_every = 10,
    force_clear_prev_results = False,
)

trainer.to('cuda')
trainer.train()