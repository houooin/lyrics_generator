"""
モデルを定義するプログラム
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu=torch.device("cpu")
class RNNLM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size=50, num_layers=1):
        super(RNNLM, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=0.5)

        self.GRU = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=self.num_layers)

        self.output = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, size=50):
        self.hidden_state = torch.zeros(self.num_layers, size, self.hidden_dim, device=device)

    def forward(self, sentence):
        embed = self.word_embeddings(sentence) # batch_len x sequence_length x embedding_dim
        drop_out = self.dropout(embed)
        if drop_out.dim() == 2:
            drop_out = torch.unsqueeze(drop_out, 1)
        gru_out, self.hidden_state = self.GRU(drop_out, self.hidden_state)# batch_len x sequence_length x hidden_dim
        gru_out = gru_out.contiguous()
        return self.output(gru_out), self.hidden_state