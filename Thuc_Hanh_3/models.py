import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import PAD_ID


class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, rnn_type="LSTM", emb_dim=300, hidden=256,
                 layers=5, dropout=0.3, num_classes=3):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = rnn_cls(emb_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, ids, lengths):
        x = self.embedding(ids)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, state = self.rnn(packed)
        h_n = state[0] if self.rnn_type == "LSTM" else state
        # h_n[-1]: trạng thái cuối của lớp trên cùng
        return self.fc(self.dropout(h_n[-1]))


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=300, hidden=256,
                 layers=5, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_tags)

    def forward(self, ids, lengths):
        x = self.embedding(ids)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=ids.size(1))
        return self.fc(self.dropout(out))