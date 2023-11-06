import argparse
from musiclm_pytorch import MusicLM
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer
from audiolm_pytorch import SoundStream, CoarseTransformer
from audiolm_pytorch import FineTransformer
from audiolm_pytorch import AudioLM

from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from musiclm_pytorch import MuLaNEmbedQuantizer

import torchaudio

def create_Mulan():
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
    ).cuda()

    return mulan, quantizer

device = 'cuda'
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', help='prompts to generate audio for', type=str, nargs='+')
    args = parser.parse_args()


    # now say you want the conditioning embeddings for semantic transformer

    mulan, quantizer = create_Mulan()

    wav2vec = HubertWithKmeans(
        checkpoint_path = 'path/to/hubert_base_ls960.pt',
        kmeans_path = 'path/to/hubert_base_ls960_L9_km500.bin'    
    ).cuda()



    soundstream = SoundStream.init_and_load_from('path/to/soundstream/checkpoint.pt').cuda()


    semantic_transformer = SemanticTransformer(
        num_semantic_tokens = wav2vec.codebook_size,
        dim = 1024,
        depth = 6,
        audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
    ).cuda()
    SemanticTransformer.load(semantic_transformer,'path/to/Semantic/checkpoint.pt')


    coarse_transformer = CoarseTransformer(
        num_semantic_tokens = wav2vec.codebook_size,
        codebook_size = 1024,
        num_coarse_quantizers = 3,
        dim = 1024,
        depth = 6,
        flash_attn = True,
        has_condition = True,
        audio_text_condition = True 
    ).cuda()
    CoarseTransformer.load(coarse_transformer,'path/to/coarse/checkpoint.pt')


    fine_transformer = FineTransformer(
        num_coarse_quantizers = 3,
        num_fine_quantizers = 5,
        codebook_size = 1024,
        dim = 1024,
        depth = 6,
        flash_attn = True,
        audio_text_condition = True 
    ).cuda()
    FineTransformer.load(fine_transformer,'path/to/fine/checkpoint.pt')


    audio_lm = AudioLM(
        wav2vec = wav2vec,
        codec = soundstream,
        semantic_transformer = semantic_transformer,
        coarse_transformer = coarse_transformer,
        fine_transformer = fine_transformer
    ).cuda()

    musiclm = MusicLM(
        audio_lm = audio_lm,                 # `AudioLM` 
        mulan_embed_quantizer = quantizer    # the `MuLaNEmbedQuantizer` from above
    ).cuda()

    music = musiclm(args.prompt, num_samples = 4) # sample 4 and pick the top match with mulan
    music = music.cpu()
    torchaudio.save( f'{args.prompt[:35]}_top_match_{4}.wav', music, 44100)