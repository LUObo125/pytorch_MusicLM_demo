# imports
from audiolm_pytorch import SoundStreamTrainer
from audiolm_pytorch import MusicLMSoundStream

# define dataset paths
dataset_folder = "/path/to/audio/files"

soundstream = MusicLMSoundStream().cuda()

trainer = SoundStreamTrainer(
    soundstream,
    folder = dataset_folder,
    batch_size = 4,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length = 320 * 32,
    save_results_every = 100,
    save_model_every = 1000,
    num_train_steps = 10_000_000
)


trainer.train()