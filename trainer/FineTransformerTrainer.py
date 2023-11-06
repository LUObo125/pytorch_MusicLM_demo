import torch
from audiolm_pytorch import SoundStream, FineTransformer, FineTransformerTrainer

from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from musiclm_pytorch import MuLaNEmbedQuantizer

device = 'cuda'

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
# setup the quantizer with the namespaced conditioning embeddings, unique per quantizer as well as namespace (per transformer)
MuLaN.load(mulan, 'F:/Gitstore/musiclm/results/mulan.120.pt')
quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)


soundstream = SoundStream.init_and_load_from('F:/Gitstore/musiclm/results/soundstream.30.pt')


fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 1024,
    depth = 6,
    flash_attn = True,
    audio_text_condition = True 
)

trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    codec = soundstream,
    audio_conditioner = quantizer,   # pass in the MulanEmbedQuantizer instance above
    folder ='F:\Gitstore\musiclm\musiclmpytorch\musicdata',
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1_000_000
).cuda()

trainer.train()