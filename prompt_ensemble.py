import torch
from tqdm import tqdm

from CLIP.tokenizer import tokenize
import numpy as np


# def encode_text_with_prompt_ensemble(model, obj, device):
#     prompt_normal = [
#         "{}",
#         "flawless {}",
#         "perfect {}",
#         "unblemished {}",
#         "{} without flaw",
#         "{} without defect",
#         "{} without damage",
#     ]
#     prompt_abnormal = [
#         "damaged {}",
#         "broken {}",
#         "{} with flaw",
#         "{} with defect",
#         "{} with damage",
#     ]
#     prompt_state = [prompt_normal, prompt_abnormal]

#     prompt_templates = [
#         "a bad {domain} photo of a {state}.",
#         "a low resolution {domain} photo of the {state}.",
#         "a bad {domain} photo of the {state}.",
#         "a cropped {domain} photo of the {state}.",
#         "a bright {domain} photo of a {state}.",
#         "a dark {domain} photo of the {state}.",
#         "a {domain} photo of my {state}.",
#         "a {domain} photo of the cool {state}.",
#         "a close-up {domain} photo of a {state}.",
#         "a black and white {domain} photo of the {state}.",
#         "a bright {domain} photo of the {state}.",
#         "a cropped {domain} photo of a {state}.",
#         "a jpeg corrupted {domain} photo of a {state}.",
#         "a blurry {domain} photo of the {state}.",
#         "a {domain} photo of the {state}.",
#         "a good {domain} photo of the {state}.",
#         "a {domain} photo of one {state}.",
#         "a close-up {domain} photo of the {state}.",
#         "a {domain} photo of a {state}.",
#         "a low resolution {domain} photo of a {state}.",
#         "a {domain} photo of a large {state}.",
#         "a blurry {domain} photo of a {state}.",
#         "a jpeg corrupted {domain} photo of the {state}.",
#         "a good {domain} photo of a {state}.",
#         "a {domain} photo of the small {state}.",
#         "a {domain} photo of the large {state}.",
#         "a black and white {domain} photo of a {state}.",
#         "a dark {domain} photo of a {state}.",
#         "a {domain} photo of a cool {state}.",
#         "a {domain} photo of a small {state}.",
#     ]

#     text_features = []
#     class_embeddings_list = []

#     # Calculate average length of prompted_sentence
#     lengths = []
#     for i in range(len(prompt_state)):
#         prompted_state = [state.format(obj) for state in prompt_state[i]]
#         prompted_sentence = []
#         for s in prompted_state:
#             for template in prompt_templates:
#                 prompted_sentence.append(template.format(domain="medical", state=s))
#         lengths.append(len(prompted_sentence))

#     average_length = int(np.mean(lengths))
#     fixed_length = average_length + 10  # Add a buffer to account for variations

#     for i in tqdm(range(len(prompt_state)), desc=f"Embedding text {obj}"):
#         prompted_state = [state.format(obj) for state in prompt_state[i]]
#         prompted_sentence = []
#         for s in prompted_state:
#             for template in prompt_templates:
#                 prompted_sentence.append(template.format(domain="medical", state=s))

#         # print(
#         #     f"Iteration {i}: Original number of sentences in prompted_sentence = {len(prompted_sentence)}"
#         # )

#         # Ensure prompted_sentence has a fixed length
#         if len(prompted_sentence) > fixed_length:
#             prompted_sentence = prompted_sentence[:fixed_length]
#         elif len(prompted_sentence) < fixed_length:
#             prompted_sentence += [""] * (fixed_length - len(prompted_sentence))

#         # print(
#         #     f"Iteration {i}: Adjusted number of sentences in prompted_sentence = {len(prompted_sentence)}"
#         # )

#         prompted_sentence = tokenize(prompted_sentence).to(device)
#         class_embeddings = model.encode_text(prompted_sentence)
#         # print(f"Iteration {i}: Class embeddings shape: {class_embeddings.shape}")

#         class_embedding = class_embeddings.mean(dim=0)
#         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#         class_embedding /= class_embedding.norm()
#         text_features.append(class_embedding)
#         class_embeddings_list.append(class_embeddings)

#     text_features = torch.stack(text_features, dim=1).to(device)
#     class_embeddings_list = torch.stack(class_embeddings_list, dim=2).to(device)

#     text_features = text_features.cuda()
#     return text_features, class_embeddings_list


def encode_text_with_prompt_ensemble(model, obj, device):
    prompt_normal = [
        "{}",
        "flawless {}",
        "perfect {}",
        "unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without damage",
    ]
    prompt_abnormal = [
        "damaged {}",
        "broken {}",
        "{} with flaw",
        "{} with defect",
        "{} with damage",
    ]
    prompt_state = [prompt_normal, prompt_abnormal]

    prompt_templates = [
        "a bad {domain} photo of a {state}.",
        "a low resolution {domain} photo of the {state}.",
        "a bad {domain} photo of the {state}.",
        "a cropped {domain} photo of the {state}.",
        "a bright {domain} photo of a {state}.",
        "a dark {domain} photo of the {state}.",
        "a {domain} photo of my {state}.",
        "a {domain} photo of the cool {state}.",
        "a close-up {domain} photo of a {state}.",
        "a black and white {domain} photo of the {state}.",
        "a bright {domain} photo of the {state}.",
        "a cropped {domain} photo of a {state}.",
        "a jpeg corrupted {domain} photo of a {state}.",
        "a blurry {domain} photo of the {state}.",
        "a {domain} photo of the {state}.",
        "a good {domain} photo of the {state}.",
        "a {domain} photo of one {state}.",
        "a close-up {domain} photo of the {state}.",
        "a {domain} photo of a {state}.",
        "a low resolution {domain} photo of a {state}.",
        "a {domain} photo of a large {state}.",
        "a blurry {domain} photo of a {state}.",
        "a jpeg corrupted {domain} photo of the {state}.",
        "a good {domain} photo of a {state}.",
        "a {domain} photo of the small {state}.",
        "a {domain} photo of the large {state}.",
        "a black and white {domain} photo of a {state}.",
        "a dark {domain} photo of a {state}.",
        "a {domain} photo of a cool {state}.",
        "a {domain} photo of a small {state}.",
    ]

    text_features = []

    for i in tqdm(range(len(prompt_state)), desc=f"Embedding text {obj}"):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(domain="medical", state=s))
                
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)

        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device)
    text_features = text_features.cuda()
    return text_features
