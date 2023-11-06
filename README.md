# pytorch_MusicLM_demo

A Demo for [musiclm-pytorch](https://github.com/lucidrains/musiclm-pytorch) comes with executable training files.

# Usage 

## Install
First install MusicLM and AudioLM<br>
```bash
$ pip install musiclm-pytorch
```
```bash
$ pip install audiolm-pytorch
```
then run two setup file

## Dataset
I use [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) to train the model, so fist try download the datasets.
You will get a lot of .WAV format audio, PLEASE remember to change the path in trainer/datasetGeneration.py

## Taining
Before training, please download the hubert checkpoints, which can be downloaded at
<https://github.com/facebookresearch/fairseq/tree/main/examples/hubert>
Check all tainers that use this part and change the path in the trainer file.

then follow the Demo.ipynb to train the model.

# Citation 
```bibtex
@inproceedings{Agostinelli2023MusicLMGM,
    title     = {MusicLM: Generating Music From Text},
    author    = {Andrea Agostinelli and Timo I. Denk and Zal{\'a}n Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matthew Sharifi and Neil Zeghidour and C. Frank},
    year      = {2023}
}
```

```bibtex
@article{Huang2022MuLanAJ,
  title   = {MuLan: A Joint Embedding of Music Audio and Natural Language},
  author  = {Qingqing Huang and Aren Jansen and Joonseok Lee and Ravi Ganti and Judith Yue Li and Daniel P. W. Ellis},
  journal = {ArXiv},
  year    = {2022},
  volume  = {abs/2208.12415}
}
```
