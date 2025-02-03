import torch

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

MODEL_PATH = './local_models/bert-large-uncased-whole-word-masking-finetuned-squad'

EMBEDDING_MODEL_PATH = './local_models/all-MiniLM-L6-v2'

FILE_PATH = './corpus.json'