import torch
import os
import torchaudio
from datasets import load_dataset
#from sklearn import preprocessing
from x_clip.tokenizer import tokenizer

############################
#dataset
############################

dir_path = 'F:\Gitstore\musiclm\musiclmpytorch\musicdata'
file_ls = os.listdir('F:\Gitstore\musiclm\musiclmpytorch\musicdata')

dataset = load_dataset("google/MusicCaps")

#le = preprocessing.LabelEncoder()
# get a ton of <sound, text> pairs and train

def load_and_process_audio(file_path, target_length):
    waveform, _ = torchaudio.load(file_path)
    # 将音频裁剪或填充到指定的长度
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    waveform = waveform.mean(0)
    return waveform

# 加载文字描述
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

min_length = 88200

num_total = 500
num = 0
# 遍历文件列表并加载音频和文本数据
for filename_with_extension in file_ls :

    filename = os.path.splitext(filename_with_extension)[0]  # 去掉扩展名
    audio_file_path = os.path.join(dir_path, filename_with_extension)
    
    audio = load_and_process_audio(audio_file_path, min_length)
    text = torch.tensor(load_text(filename, dataset))
    
    audio_tensors.append(audio)
    text_tensors.append(text)
    num += 1
    if num >= num_total:
        break

# 将列表转换为张量
wavs = torch.stack(audio_tensors)
texts = torch.stack(text_tensors)

torch.save(wavs,'wavs_small.pt')
torch.save(texts,'texts_small.pt')