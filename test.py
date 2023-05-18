# import pdb
#
# from transformers import AutoTokenizer, AutoModelForMaskedLM
#
# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
# model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
#
# pdb.set_trace()
# # prepare input
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
#
# # forward pass
# output = model(**encoded_input)

'''

import pdb

import torch
# Load the model in fairseq
from fairseq.models.roberta import XLMRModel
xlmr = XLMRModel.from_pretrained('/path/to/xlmr.large', checkpoint_file='model.pt')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)

pdb.set_trace()

en_tokens = xlmr.encode('Hello world!')
assert en_tokens.tolist() == [0, 35378,  8999, 38, 2]
xlmr.decode(en_tokens)  # 'Hello world!'

# Extract the last layer's features
last_layer_features = xlmr.extract_features(en_tokens)
assert last_layer_features.size() == torch.Size([1, 6, 1024])

'''

# SentenceTransformer
import pdb

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

pdb.set_trace()

#Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence']

#Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)
