import torch
import os
import torchaudio
from datasets import load_dataset
#from sklearn import preprocessing
from x_clip.tokenizer import tokenizer

############################
#dataset
############################

dir_path = '/path/to/audio/files'
file_ls = os.listdir(dir_path)

# if you want use other dataset, please check the format fist, may can't be used here 
dataset = load_dataset("google/MusicCaps")

# get a ton of <sound, text> pairs and train

def load_and_process_audio(file_path, target_length):
    waveform, _ = torchaudio.load(file_path)
    # Trim or pad the audio to a specified length
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    waveform = waveform.mean(0)
    return waveform

def load_text(filename, dataset):
    index = dataset['train']['ytid'].index(filename)
    text = dataset['train']['aspect_list'][index]
    text = tokenizer.tokenize([text])
    if text.shape[1] < 144:
        padding = 144 - text.shape[1]
        text = torch.nn.functional.pad(text, (0, padding))
    text = text.reshape(-1)
    return text

audio_tensors = []
text_tensors = []

min_length = min([torchaudio.load(os.path.join(dir_path, file))[0].shape[1] for file in file_ls])

for filename_with_extension in file_ls :

    filename = os.path.splitext(filename_with_extension)[0]  # 去掉扩展名
    audio_file_path = os.path.join(dir_path, filename_with_extension)
    
    audio = load_and_process_audio(audio_file_path, min_length)
    text = torch.tensor(load_text(filename, dataset))
    
    audio_tensors.append(audio)
    text_tensors.append(text)

# 将列表转换为张量
wavs = torch.stack(audio_tensors)
texts = torch.stack(text_tensors)

torch.save(wavs,'wavs.pt')
torch.save(texts,'texts.pt')