import torch
from torch import nn
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8


tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomedclip, preprocess = create_model_from_pretrained(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
biomedclip = biomedclip.cuda()
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
