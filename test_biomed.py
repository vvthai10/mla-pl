from biomedclip.learner.config import biomedclip_hf_api
from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
)  # works on open-clip-torch>=2.23.0, timm>=0.9.8

biomedclip, preprocess = create_model_from_pretrained(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
biomedclip = biomedclip.cuda()
biomedclip.eval()

# Inspect the text model attributes
# print(dir(biomedclip.text))

# # Check for specific attributes
# print(hasattr(biomedclip.text.transformer, "embeddings"))  # Check if embeddings exist
# print(
#     hasattr(
#         biomedclip.text.transformer.embeddings,
#         "position_embeddings",
#     )
# )

# print(hasattr(biomedclip.text, "transformer"))
# print(
#     hasattr(biomedclip.text.transformer.embeddings, "LayerNorm")
# )  # Check if LayerNorm exists

# print(biomedclip.text.proj)
print(dir(biomedclip))
# print(biomedclip.text.transformer.width)
print(biomedclip.text.transformer)
print(biomedclip.text.transformer.embeddings.position_embeddings.weight.shape)
print(biomedclip.text.transformer.embeddings.LayerNorm)
# print(dir(biomedclip.text.transformer))
