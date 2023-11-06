import torch
from audiolm_pytorch import HubertWithKmeans, SoundStream, CoarseTransformer, CoarseTransformerTrainer
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
MuLaN.load(mulan, 'path/to/mulan/checkpoint.pt')
quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)

# now say you want the conditioning embeddings for semantic transformer

wav2vec = HubertWithKmeans(
    checkpoint_path = 'path/to/hubert_base_ls960.pt',
    kmeans_path = 'path/to/hubert_base_ls960_L9_km500.bin'
)


soundstream = SoundStream.init_and_load_from('path/to/soundstream/checkpoint.pt')

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 1024,
    depth = 6,
    flash_attn = True,
    has_condition = True,
    audio_text_condition = True 
)

trainer = CoarseTransformerTrainer(
    transformer = coarse_transformer,
    codec = soundstream,
    wav2vec = wav2vec,
    audio_conditioner = quantizer,   # pass in the MulanEmbedQuantizer instance above
    folder ='/path/to/audio/files',
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1_000_000
).cuda()

trainer.train()