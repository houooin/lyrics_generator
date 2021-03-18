from models import RNNLM
from sentence_EncoderDecoder import EncoderDecoder
import MeCab
import pandas as pd
import torch.nn.utils.rnn as rnn
import torch
import torch.nn as nn
import torch.optim as optim

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu=torch.device("cpu")
def batchfy(sentences_id, batch_size = 50):
    in_batch = []
    out_batch = []
    i = 1
    for sentence_id in sentences_id:
        inp = torch.tensor(sentence_id[:-1])
        out = torch.tensor(sentence_id[1:])
        in_batch.append(inp)
        out_batch.append(out)
    in_batch = rnn.pad_sequence(in_batch, batch_first=True)
    batch_in = torch.split(in_batch, 50, dim=0)
    in_batch = rnn.pad_sequence(batch_in, batch_first=True)
    out_batch = rnn.pad_sequence(out_batch, batch_first=True)
    batch_out = torch.split(out_batch, 50, dim=0)
    out_batch = rnn.pad_sequence(batch_out, batch_first=True)
    return in_batch, out_batch
    
m = MeCab.Tagger('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
df = pd.read_csv("../data/lyrics.csv")
sentences = []
converter = EncoderDecoder()
for text in df['lyrics']:
    text = text.replace('　','')
    texts = text.split("/")
    for i in texts:
        if i == "":
            continue
        result = m.parse(i).strip().split(" ")
        sentences.append(result)
converter.fit(sentences)
sentences_id = converter.transform(sentences, bos=True, eos=True)
in_batch, out_batch = batchfy(sentences_id, batch_size = 50)
vocab_size = len(converter.i2w)
print("vocab_size = ",vocab_size)
embedding_dim = 1000
hidden_dim = 600
model = RNNLM(embedding_dim, hidden_dim, vocab_size).to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for i in range(len(in_batch)):
    model.zero_grad()
    model.init_hidden()
    inp = in_batch[i].to(device)
    oup = out_batch[i].to(device)
    y, hidden = model(inp)
    loss = criterion(y.view(50, vocab_size, -1), oup)
    loss.backward()
    optimizer.step()
print("finish")
hidden = model.init_hidden(1)
input  = torch.tensor([converter.encode(['<s>'])]).to(device)
answer = []
for i in range(10):
    print(input)
    output, hidden = model(input)
    _, pred = torch.max(output, 2)
    input = pred
    answer.append(converter.decode([int(pred.squeeze())])[0])
print(answer)