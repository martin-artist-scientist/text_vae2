#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import string
from collections import Counter


# ### Directories and text loading
# Initially we will set the main directories and some variables regarding the characteristics of our texts.
# We set the maximum sequence length to 25, the maximun number of words in our vocabulary to 12000 and we will use 300-dimensional embeddings. Finally we load our texts from a csv. The text file is the train file of the Quora Kaggle challenge containing around 808000 sentences.

# In[2]:


# Define the number of rows to load (set to None to load all rows)
num_rows = None  # Replace with the desired number of rows to load, or set to None for all rows

# Load the data
if num_rows is None:
    data = pd.read_csv('test_small.csv')
else:
    data = pd.read_csv('test_small.csv', nrows=num_rows)

data.head()


# ### Text Preprocessing
# To preprocess the text we will use the tokenizer and the text_to_sequences function from Keras
# 

# In[3]:


from transformers import BertTokenizer

def preprocess_text(text):
    if isinstance(text, str):
        # Lowercase the text
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', '', text)
    else:
        text = ''
    return text

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

data['question1'] = data['question1'].apply(lambda x: tokenizer.encode_plus(preprocess_text(x), truncation=True, max_length=128, padding='max_length'))
data['question2'] = data['question2'].apply(lambda x: tokenizer.encode_plus(preprocess_text(x), truncation=True, max_length=128, padding='max_length'))

data.head()


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[5]:


from transformers import BertModel

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# If you have a GPU, put everything on cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def embed_text(token_ids):
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([token_ids]).to(device)

    # Predict hidden states features for each layer
    with torch.no_grad():
        outputs = model(tokens_tensor)

    # Get the embeddings
    embeddings = outputs.last_hidden_state

    # Calculate the mean embeddings
    mean_embeddings = torch.mean(embeddings, dim=1).cpu().numpy()

    return mean_embeddings

data['question1'] = data['question1'].apply(lambda x: embed_text(x['input_ids']))
data['question2'] = data['question2'].apply(lambda x: embed_text(x['input_ids']))

data.head()


# In[6]:


# Pad the sequences to a fixed length
def pad_sequence(sequence):
    if len(sequence) > 768:
        return sequence[:768]
    else:
        return np.pad(sequence, (0, 768 - len(sequence)), 'constant')

data['question1'] = data['question1'].apply(pad_sequence)
data['question2'] = data['question2'].apply(pad_sequence)

data.head()


# In[7]:


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(768, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 768)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 768))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# In[10]:


model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 768), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# In[11]:


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data1, data2) in enumerate(zip(data['question1'], data['question2'])):
        data1 = torch.from_numpy(data1).to(device)
        data2 = torch.from_numpy(data2).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data1)
        loss = loss_function(recon_batch, data2, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(data['question1']),
                100. * batch_idx / len(data['question1']),
                loss.item() / len(data1)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data['question1'])))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 10 + 1):
    train(epoch)


# In[13]:


def generate_text(epoch):
    model.eval()
    sample = torch.randn(64, 20).to(device)  # Move the tensor to the GPU
    sample = model.decode(sample)
    print('====> Generated text after epoch {}: {}'.format(epoch, sample))

for epoch in range(1, 10 + 1):
    generate_text(epoch)


# ### Project and sample sentences from the latent space
# Now we build an encoder model model that takes a sentence and projects it on the latent space and a decoder model that goes from the latent space back to the text representation

# In[18]:


# build a model to project inputs on the latent space
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(768, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

# build a generator that can sample from the learned distribution
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 768)
    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

encoder = Encoder().to(device)
decoder = Decoder().to(device)


# ### Test on validation sentences

# In[19]:


index2word = {v: k for k, v in tokenizer.get_vocab().items()}
index2word[0] = 'pad'

#test on a validation sentence
sent_idx = 100
sent_encoded = encoder(torch.from_numpy(data['question1'][sent_idx]).to(device))
x_test_reconstructed = decoder(sent_encoded)
reconstructed_indexes = torch.argmax(x_test_reconstructed, dim=1).cpu().numpy()
word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
print(' '.join(word_list))
original_sent = list(np.vectorize(index2word.get)(data['question1'][sent_idx]))
print(' '.join(original_sent))


# ### Sentence processing and interpolation

# In[ ]:


# function to parse a sentence
def sent_parse(sentence, mat_shape):
    sequence = tokenizer.encode(sentence)
    padded_sent = pad_sequence(sequence)
    return padded_sent

# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in data['question1']:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = data['question1'][maximum]
    return new_vec

# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample

# input: original dimension sentence vector
# output: sentence text
def print_latent_sentence(sent_vect):
    sent_vect = torch.tensor(sent_vect).to(device)
    sent_reconstructed = decoder(sent_vect)
    reconstructed_indexes = torch.argmax(sent_reconstructed, dim=1).cpu().numpy()
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    print(' '.join(word_list))

def new_sents_interp(sent1, sent2, n):
    tok_sent1 = sent_parse(sent1, [MAX_SEQUENCE_LENGTH + 2])
    tok_sent2 = sent_parse(sent2, [MAX_SEQUENCE_LENGTH + 2])
    enc_sent1 = encoder(torch.tensor(tok_sent1).to(device))
    enc_sent2 = encoder(torch.tensor(tok_sent2).to(device))
    test_hom = shortest_homology(enc_sent1.detach().cpu().numpy(), enc_sent2.detach().cpu().numpy(), n)
    for point in test_hom:
        print_latent_sentence(point)


# ### Example
# Now we can try to parse two sentences and interpolate between them generating new sentences

# In[ ]:


sentence1='where can i find a bad restaurant'
mysent = sent_parse(sentence1, [MAX_SEQUENCE_LENGTH + 2])
mysent_encoded = encoder(torch.tensor(mysent).to(device))
print_latent_sentence(mysent_encoded.detach().cpu().numpy())
print_latent_sentence(find_similar_encoding(mysent_encoded.detach().cpu().numpy()))

sentence2='where can i find an extremely good restaurant'
mysent2 = sent_parse(sentence2, [MAX_SEQUENCE_LENGTH + 2])
mysent_encoded2 = encoder(torch.tensor(mysent2).to(device))
print_latent_sentence(mysent_encoded2.detach().cpu().numpy())
print_latent_sentence(find_similar_encoding(mysent_encoded2.detach().cpu().numpy()))
print('-----------------')

new_sents_interp(sentence1, sentence2, 5)


# In[ ]:


# function to parse a sentence
def sent_parse(sentence, tokenizer, device):
    sequence = tokenizer.encode_plus(sentence, return_tensors='pt')
    return sequence['input_ids'].to(device)

# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect, encoded_data):
    all_cosine = []
    for sent in encoded_data:
        result = 1 - spatial.distance.cosine(sent_vect.cpu().numpy(), sent.cpu().numpy())
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = encoded_data[maximum]
    return new_vec

# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample

# input: original dimension sentence vector
# output: sentence text
def print_latent_sentence(sent_vect, decoder, index2word):
    sent_vect = sent_vect.unsqueeze(0)
    sent_reconstructed = decoder(sent_vect)
    reconstructed_indexes = torch.argmax(sent_reconstructed, dim=2).squeeze().cpu().numpy()
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    print(' '.join(word_list))

def new_sents_interp(sent1, sent2, n, tokenizer, device, encoder, decoder, index2word):
    tok_sent1 = sent_parse(sent1, tokenizer, device)
    tok_sent2 = sent_parse(sent2, tokenizer, device)
    enc_sent1 = encoder(tok_sent1)
    enc_sent2 = encoder(tok_sent2)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    for point in test_hom:
        print_latent_sentence(point, decoder, index2word)


# In[ ]:


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence1='where can i find a bad restaurant'
sentence2='where can i find an extremely good restaurant'
new_sents_interp(sentence1, sentence2, 5, tokenizer, device, encoder, decoder, index2word)

