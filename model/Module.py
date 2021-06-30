import  torch
from    torch import nn
from    torch.nn import functional as F
from    numpy import inf
import  math
from    copy import deepcopy
from    torch.autograd import Variable

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_qk, d_v, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_qk = d_qk
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, num_head * d_qk)
        self.W_K = nn.Linear(d_model, num_head * d_qk)
        self.W_V = nn.Linear(d_model, num_head * d_v)
        self.W_out = nn.Linear(d_v * num_head, d_model)

    def ScaledDotProductAttention(self, query, keys, values, mask):
        score = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.d_model)
        if mask is not None:
            score.masked_fill_(mask.unsqueeze(1), -inf)   
        weight = F.softmax(score, dim=-1)
        return torch.matmul(weight, values), weight

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        query = self.W_Q(Q).view(batch_size, Q.size(1), self.num_head, self.d_qk)
        keys = self.W_K(K).view(batch_size, K.size(1), self.num_head, self.d_qk)
        values = self.W_V(V).view(batch_size, V.size(1), self.num_head, self.d_v)
        query.transpose_(1, 2)
        keys.transpose_(1, 2)
        values.transpose_(1, 2)

        outputs, weight = self.ScaledDotProductAttention(query, keys, values, mask)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v*self.num_head)
        return self.W_out(outputs), weight

class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0., max_length=5000):
        super().__init__()
        self.embed = nn.Linear(vocab_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.register_buffer('PE', self.PositionalEncoding(int(max_length / 2), embed_size))
        self.max_length = max_length
        self.d_model = embed_size
        self.vocab_size = vocab_size

    def PositionalEncoding(self, seq_length, embedding_dim):

        position = torch.arange(0., seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * 
                             -(math.log(10000.0) / embedding_dim))
        tmp = position * div_term
        pe = torch.zeros(seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(tmp)
        pe[:, 1::2] = torch.cos(tmp)  

        return pe.detach_()

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = F.one_hot(inputs, num_classes=self.vocab_size).float()
        seq_length = inputs.size(1)
        if seq_length <= self.max_length:
            outputs = self.embed(inputs) * math.sqrt(self.d_model) + self.PE[:seq_length]
        else:
            outputs = self.embed(inputs) * math.sqrt(self.d_model) \
                    + self.PositionalEncoding(seq_length, self.d_model).to(inputs.device)
        # print(outputs.size())
        return self.dropout(outputs)

class PositionWiseFeedForwardNetworks(nn.Module):

    def __init__(self, input_size, output_size, d_ff):
        super().__init__()
        self.W_1 = nn.Linear(input_size, d_ff, bias=True)
        self.W_2 = nn.Linear(d_ff, output_size, bias=True)
        nn.init.constant_(self.W_1.bias, 0.)
        nn.init.constant_(self.W_2.bias, 0.)
    def forward(self, inputs):
        outputs = F.relu(self.W_1(inputs))
        return self.W_2(outputs)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        out = sublayer(self.norm(x))
        if isinstance(out, tuple):
            return x + self.dropout(out[0]), out[1]
        else:
            return x + self.dropout(out)

class EncoderCell(nn.Module):

    def __init__(self, d_model, attn, FFNlayer, dropout):
        super().__init__()
        self.self_attn = deepcopy(attn)
        self.FFN = deepcopy(FFNlayer)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, inputs, pad_mask):
        inputs, weight = self.sublayer[0](inputs, lambda x: self.self_attn(x, x, x, pad_mask))
        return self.sublayer[1](inputs, self.FFN), weight

class DecoderCell(nn.Module):

    def __init__(self, d_model, attn, FFNlayer, dropout):
        super().__init__()
        self.self_attn = deepcopy(attn)
        self.cross_attn = deepcopy(attn)
        self.FFN = deepcopy(FFNlayer)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, inputs, encoder_outputs, src_mask, tgt_mask):
        m = encoder_outputs
        inputs, self_attn_weight = self.sublayer[0](inputs, lambda x: self.self_attn(x, x, x, tgt_mask))
        inputs, cross_attn_weight = self.sublayer[1](inputs, lambda x: self.cross_attn(x, m, m, src_mask))
        return self.sublayer[2](inputs, self.FFN), self_attn_weight, cross_attn_weight

class Encoder(nn.Module):
    def __init__(self, d_model, num_layer, layer):
        super().__init__()
        self.encoder = nn.ModuleList([deepcopy(layer) for _ in range(num_layer)])    
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, inputs, pad_mask):
        for encoder_cell in self.encoder:
            inputs, _ = encoder_cell(inputs, pad_mask)
        return self.layer_norm(inputs)

class Decoder(nn.Module):

    def __init__(self, d_model, vocab_size, num_layer, layer):
        super().__init__()
        self.decoder = nn.ModuleList([deepcopy(layer) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, inputs, encoder_outputs, pad_mask, seq_mask):
        for decoder_cell in self.decoder:
            inputs, _, cross_attn_weight = decoder_cell(inputs, encoder_outputs, pad_mask, seq_mask)
        return self.layer_norm(inputs), cross_attn_weight

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0., ignore_index=None):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, norm=1):
        inputs = torch.log(inputs)
        vocab_size = inputs.size(-1)
        batch_size = targets.size(0)
        length = targets.size(1)
        if self.ignore_index is not None:
            mask = (targets == self.ignore_index).view(-1)
        
        index = targets.unsqueeze(-1)
        targets = F.one_hot(targets, num_classes=inputs.size(-1))
        targets = targets * (1 - self.smoothing) + self.smoothing / vocab_size
        loss = self.criterion(inputs.view(-1, vocab_size), 
                              targets.view(-1, vocab_size).detach()).sum(dim=-1)
        if self.ignore_index is not None:
            return loss.masked_fill(mask, 0.).sum() / norm
        else:
            return loss.sum() / norm

class WarmUpOpt:

    def __init__(self, optimizer, d_model, warmup_steps, factor=1):
        self.__step = 0
        self.factor = factor
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer

    def updateRate(self):
        return self.factor * (self.d_model**(-0.5) * 
                    min(self.__step**(-0.5), self.__step * self.warmup_steps**(-1.5)))

    def step(self):
        self.__step += 1
        for param in self.optimizer.param_groups:
            param['lr'] = self.updateRate()
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()