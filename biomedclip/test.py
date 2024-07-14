import torch
import open_clip_custom
from torch import nn
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomedclip = open_clip.create_model('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224').to('cuda')
biomedclip.eval()

biomedclip_hf_api = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]
texts = tokenizer([template + l for l in labels]).to('cuda')

out = biomedclip.text.transformer(input_ids=texts,)
