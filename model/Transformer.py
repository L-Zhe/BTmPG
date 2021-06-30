import  torch
from    torch import nn
from    torch.nn import functional as F
from    .Module import Encoder, Decoder, PositionWiseFeedForwardNetworks, \
                       Embedding, MultiHeadAttention, EncoderCell, DecoderCell
from    copy import deepcopy


def pad_mask(inputs, PAD):
    return (inputs==PAD).unsqueeze(1)


def triu_mask(length):
    mask = torch.ones(length, length).triu(1)
    return mask.unsqueeze(0) == 1


def copy(prob, p_g, eta, src_index):
    p_g = (1 - eta) * p_g
    prob = eta * prob
    if src_index.dim() == 2:
        src_index = src_index.unsqueeze(1).repeat(1, prob.size(1), 1)
        return prob.scatter_add(2, src_index, p_g)
    else:
        return prob + torch.matmul(p_g, src_index)


class SearchMethod:

    def __init__(self, search_method, BOS, EOS):
        self.search_method = search_method
        self.BOS = BOS
        self.EOS = EOS
    
    def greedy_search(self, tgt_embed, src_index, src_pad_mask, encoder_outputs, max_length):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        sentence = torch.LongTensor([self.BOS]).repeat(batch_size).reshape(-1, 1).to(device)
        EOS_flag = torch.BoolTensor(batch_size).fill_(False).to(device)
        EOS_index = torch.zeros(batch_size).fill_(max_length).to(device)
        total_prob = torch.FloatTensor().to(device)
        for i in range(max_length):
            embed = tgt_embed(sentence)
            seq_mask = triu_mask(i + 1).to(device)
            outputs, attn_weight = self.Decoder(embed, encoder_outputs, src_pad_mask, seq_mask)
            eta = torch.sigmoid(self.w_eta(outputs[:, -1:, :]))
            prob = F.softmax(self.project(outputs[:, -1:, :]), dim=-1)
            prob = copy(prob, attn_weight.mean(1)[:, -1:, :], eta, src_index)
            total_prob = torch.cat((total_prob, prob), dim=1)
            word = prob.max(dim=-1)[1].long()
            sentence = torch.cat((sentence, word), dim=-1)
            mask = (word==self.EOS).view(-1).masked_fill_(EOS_flag, False)
            EOS_index.masked_fill_(mask, i + 1)
            EOS_flag |= mask
            if (EOS_flag==False).sum() == 0:   break
        sent = sentence.tolist()
        for i in range(batch_size):
            sent[i] = sent[i][1:int(EOS_index[i].item())]
        return sentence, total_prob, sent

    def beam_search(self):
        pass

class transformer(nn.Module, SearchMethod):

    def __init__(self, **params):
        for key in params.keys():
            if key not in ['embedding_dim', 'vocab_size', 'src_vocab_size', 'tgt_vocab_size', 
                           'num_head', 'num_layer_encoder', 'num_layer_decoder', 'd_ff', 'share_embed', 
                           'BOS_index', 'EOS_index', 'PAD_index', 'dropout_embed', 'dropout_sublayer']:
                raise ValueError('Invalid Parameter Name: ' + key)

        d_model  = params['embedding_dim'] if params.get('embedding_dim') is not None else 512
        num_head = params['num_head'] if params.get('num_head') is not None else 8
        num_layer_encoder = params['num_layer_encoder'] if params.get('num_layer_encoder') is not None else 6
        num_layer_decoder = params['num_layer_decoder'] if params.get('num_layer_decoder') is not None else 6
        d_ff = params['d_ff'] if params.get('d_ff') is not None else 2048
        dropout_embed    = params['dropout_embed'] if params.get('dropout_embed') is not None else 0.1
        dropout_sublayer = params['dropout_sublayer'] if params.get('dropout_sublayer') is not None else 0.1
        share_embed = params['share_embed'] if params.get('share_embed') is not None else False
        BOS_index   = params['BOS_index']
        EOS_index   = params['EOS_index']
        PAD_index   = params['PAD_index']

        super(transformer, self).__init__()
        super(nn.Module, self).__init__(None, BOS_index, EOS_index)

        if d_model % num_head != 0:
            raise ValueError("Parameter Error, require embedding_dim % num head == 0.")

        d_qk = d_v = d_model // num_head
        attention = MultiHeadAttention(d_model, d_qk, d_v, num_head)
        FFN = PositionWiseFeedForwardNetworks(d_model, d_model, d_ff)
        if share_embed:
            vocab_size = params['vocab_size']
            self.src_embed = Embedding(vocab_size, d_model, dropout=dropout_embed)
            self.tgt_embed = self.src_embed

        else:
            src_vocab_size = params['src_vocab_size']
            tgt_vocab_size = params['tgt_vocab_size']
            self.src_embed = Embedding(src_vocab_size, d_model, dropout=dropout_embed)
            self.tgt_embed = Embedding(tgt_vocab_size, d_model, dropout=dropout_embed)
            vocab_size = tgt_vocab_size
        
        self.Encoder = Encoder(d_model=d_model, num_layer=num_layer_encoder,
                               layer=EncoderCell(d_model, attention, FFN, dropout_sublayer))
        self.Decoder = Decoder(d_model=d_model, num_layer=num_layer_decoder, vocab_size=vocab_size,
                               layer=DecoderCell(d_model, attention, FFN, dropout_sublayer))
        self.PAD_index = PAD_index
        self.w_eta = nn.Linear(d_model, 1, bias=True)
        self.project = nn.Linear(d_model, vocab_size)
        nn.init.constant_(self.w_eta.bias, 0.)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source, target):
        src_embed = self.src_embed(source)
        tgt_embed = self.tgt_embed(target)
        src_mask = None
        # src_mask = pad_mask(source, self.PAD_index)
        tgt_mask = triu_mask(target.size(1)).to(tgt_embed.device) | pad_mask(target, self.PAD_index)
        encoder_outputs = self.Encoder(src_embed, src_mask)
        outputs, attn_weight = self.Decoder(tgt_embed, encoder_outputs, src_mask, tgt_mask)
        eta = torch.sigmoid(self.w_eta(outputs))
        return copy(F.softmax(self.project(outputs), dim=-1), attn_weight.mean(1), eta, source)
        # return F.softmax(self.project(outputs), dim=-1)

    def generate(self, source, max_length):
        src_embed = self.src_embed(source)
        # src_pad_mask = pad_mask(source, self.PAD_index).to(source.device)
        src_pad_mask = None
        encoder_outputs = self.Encoder(src_embed, src_pad_mask)
        return self.greedy_search(self.tgt_embed, source, src_pad_mask, encoder_outputs, max_length)
