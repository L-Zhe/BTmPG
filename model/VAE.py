import  torch
from    torch import nn
from    torch.nn import functional as F
from    copy import deepcopy
from    .Transformer import pad_mask
from    .Module import MultiHeadAttention, SublayerConnection
from    .gumbleSoftmax import gumble_softmax
from    math import inf


class CopyLoss(nn.Module):
    def __init__(self, ignore_index, reduction):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, outputs, target):
        loss = (torch.log(outputs) * F.one_hot(target, num_classes=outputs.size(-1))).sum(dim=-1)
        mask = (target == self.ignore_index)
        loss = loss.masked_fill(mask, 0)
        if self.reduction == 'none':
            return -loss
        loss = loss.sum()
        if self.reduction == 'mean':
            return -loss / (mask == 0).sum()
        else:
            return -loss


class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size

    def forward(self, inputs, context, src_len=None):
        self.lstm.flatten_parameters()
        if src_len is not None:
            src_len, src_index = src_len.sort(descending=True)
            inputs = inputs[src_index]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, src_len, batch_first=True)
            context = (context[0][:, src_index, :], context[1][:, src_index, :])
        outputs, (h, c) = self.lstm(inputs, context)
        if src_len is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            _, src_index = src_index.sort(descending=False)
            outputs = outputs[src_index]
            h = h[:, src_index, :]
            c = c[:, src_index, :]
        return outputs, (h, c)


def kld_coef(i):
    import math
    return (math.tanh((i - 8000)/1000) + 1)/2
    

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout_embed, src_encoder, 
                 para_encoder, decoder, attn, latent_num, PAD, BOS, EOS):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(vocab_size, embed_size),
                                   nn.Dropout(dropout_embed))
        self.src_encoder = src_encoder
        self.para_encoder = para_encoder
        self.decoder = decoder
        self.num_direction = para_encoder.num_direction
        self.num_layers = para_encoder.num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = para_encoder.hidden_size
        self.latent_num = latent_num

        self.w_h = nn.Linear(self.hidden_size * self.num_direction,
                             self.hidden_size)
        self.w_c = nn.Linear(self.hidden_size * self.num_direction,
                             self.hidden_size)                             

        self.w_out = nn.Linear(2 * self.hidden_size, self.vocab_size)
        self.sublayer = nn.ModuleList([SublayerConnection(src_encoder.hidden_size, 0) for _ in range(2)])
        self.w_mu = nn.Linear(self.hidden_size * self.num_direction * self.num_layers, latent_num)
        self.w_log_var = nn.Linear(self.hidden_size * self.num_direction * self.num_layers, latent_num)
        self.PAD = PAD
        self.BOS = BOS
        self.EOS = EOS
        self.iter_num = 0
        self.GS = gumble_softmax(3500, 100)

        self.w_h_p = nn.Linear(self.hidden_size, 1)
        self.w_s_p = nn.Linear(self.embed_size, 1)
        self.w_x_p = nn.Linear(self.hidden_size, 1)
        self.w_z_p = nn.Linear(self.latent_num, 1)
        self.b_ptr = nn.Parameter(torch.zeros(1))

    def reparameterize(self, h, n):
        mu = self.w_mu(h)
        log_var = self.w_log_var(h)
        z = torch.FloatTensor().to(h.device)
        z = torch.randn_like(mu) * torch.exp(log_var / 2) + mu
        return mu, log_var, torch.cat([z] * n, dim=1)

    def init_hidden(self, hidden, w):
        hidden = hidden.view(self.num_layers, self.num_direction, -1, self.hidden_size).transpose(1, 2) \
                       .contiguous().view(self.num_layers, -1, self.hidden_size * self.num_direction)
        return w(hidden)

    def len_embed(self, inputs):
        if inputs.dim() == 2:
            length = (inputs != self.PAD).sum(dim=-1)
            length.masked_fill_(length == 0, 1)
            inputs = F.one_hot(inputs, num_classes=self.vocab_size).float()
        else:
            length = torch.zeros(inputs.size(0)).fill_(inputs.size(1))
        
        return self.embed(inputs), length

    def attn(self, inputs, memory, mask=None):
        score = torch.matmul(inputs, memory.transpose(-1, -2))
        if mask is not None:
            score.masked_fill_(mask, -inf)
        weight = F.softmax(score, dim=-1)
        attn_vector = torch.matmul(weight, memory)
        return attn_vector, weight

    def copy(self, prob, attn_weight, p_g, src_index):
        
        if src_index.dim() == 2:
            src_index = src_index.unsqueeze(1).repeat(1, prob.size(1), 1)
            attn_weight = (1 - p_g) * attn_weight
            prob = prob * p_g
            return prob.scatter_add(2, src_index, attn_weight)
        else:
            return prob * p_g + (1 - p_g) * torch.matmul(attn_weight, src_index)

    def forward(self, source, paraphrase, tgt_inputs):
        para_embed, para_len = self.len_embed(paraphrase)
        src_embed, src_len = self.len_embed(source)
        out_embed, out_len = self.len_embed(tgt_inputs)

        batch_size = src_embed.size(0)

        hidden = torch.randn(self.num_layers * self.num_direction,
                             batch_size, self.hidden_size).to(source.device)
        context = torch.randn(self.num_layers * self.num_direction,
                              batch_size, self.hidden_size).to(source.device)

        src_outputs, src_context = self.src_encoder(src_embed, (hidden, context), src_len)
        para_outputs, (h, c) = self.para_encoder(para_embed, src_context, para_len)

        src_outputs = src_outputs[..., :self.hidden_size] + src_outputs[..., self.hidden_size:]
        src_context = (self.init_hidden(src_context[0], self.w_h),
                       self.init_hidden(src_context[1], self.w_c))

        h = h.view(self.num_layers, self.num_direction, batch_size, -1).transpose(0, 2) \
             .contiguous().view(batch_size, 1, -1)
        mu, log_var, z = self.reparameterize(h, out_embed.size(1))

        decode_inputs = torch.cat((out_embed, z), dim=-1)
        outputs, (h, _) = self.decoder(decode_inputs, src_context)

        attn_vector, weight = self.attn(outputs, src_outputs, (source == self.PAD).unsqueeze(1))
        p_g = torch.sigmoid(self.w_h_p(attn_vector) + self.w_x_p(outputs) 
                            + self.w_s_p(out_embed) + self.w_z_p(z) + self.b_ptr)
        outputs = F.softmax(self.w_out(torch.cat((outputs, attn_vector), dim=-1)), dim=-1)
    
        outputs = self.copy(outputs, weight, p_g, source)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        self.iter_num += 1
        h_pre = h.transpose(0, 1).contiguous().view(batch_size, -1)
        return outputs, kl_div * kld_coef(self.iter_num - 1), h_pre

    def round(self, source, max_length, paraphrase):
        
        src_embed, src_len = self.len_embed(source)
        batch_size = source.size(0)
        hidden = torch.randn(self.num_layers * self.num_direction,
                             batch_size, self.hidden_size).to(source.device)
        context = torch.randn(self.num_layers * self.num_direction,
                              batch_size, self.hidden_size).to(source.device)
        src_outputs, src_context = self.src_encoder(src_embed, (hidden, context), src_len)
        para_embed, para_len = self.len_embed(paraphrase)
        para_outputs, (h, c) = self.para_encoder(para_embed, src_context, para_len)
        h = h.view(self.num_layers, self.num_direction, batch_size, -1).transpose(0, 2) \
             .contiguous().view(batch_size, 1, -1)
        mu, log_var, z = self.reparameterize(h, 1)
        # kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        src_outputs = src_outputs[..., :self.hidden_size] + src_outputs[..., self.hidden_size:]
        src_context = (self.init_hidden(src_context[0], self.w_h),
                       self.init_hidden(src_context[1], self.w_c))

        sentence = torch.LongTensor(batch_size, 1).fill_(self.BOS).to(source.device)
        sentence = F.one_hot(sentence, num_classes=self.vocab_size).float()
        mask_src = None
        if source.dim() == 2:
            mask_src = (source == self.PAD).unsqueeze(1)
        z_p = self.w_z_p(z)
        for i in range(max_length):
            out_embed = self.embed(self.GS(sentence[:, -1:, :]))
            inputs = torch.cat((out_embed, z), dim=-1)
            outputs, src_context = self.decoder(inputs, src_context)
            attn_vector, weight = self.attn(outputs, src_outputs, mask_src)
            p_g = torch.sigmoid(self.w_h_p(attn_vector) + self.w_x_p(outputs) 
                                + self.w_s_p(out_embed) + z_p + self.b_ptr)
            outputs = F.softmax(self.w_out(torch.cat((outputs, attn_vector), dim=-1)), dim=-1)
            outputs = self.copy(outputs, weight, p_g, source)
            sentence = torch.cat((sentence, outputs), dim=1)
        self.GS.step_n()
        h_now = src_context[0].transpose(0, 1).contiguous().view(batch_size, -1)
        return sentence, h_now

    def generate(self, source, max_length, alpha=1):
        src_embed, src_len = self.len_embed(source)
        batch_size = source.size(0)
        hidden = torch.randn(self.num_layers * self.num_direction,
                             batch_size, self.hidden_size).to(source.device)
        context = torch.randn(self.num_layers * self.num_direction,
                              batch_size, self.hidden_size).to(source.device)
        src_outputs, src_context = self.src_encoder(src_embed, (hidden, context), src_len)
        z = torch.randn(batch_size, 1, self.latent_num).to(source.device) * alpha
        # z = torch.zeros(batch_size, 1, self.latent_num).to(source.device)
        src_outputs = src_outputs[..., :self.hidden_size] + src_outputs[..., self.hidden_size:]
        src_context = (self.init_hidden(src_context[0], self.w_h),
                       self.init_hidden(src_context[1], self.w_c))
        sentence = torch.LongTensor(batch_size, 1).fill_(self.BOS).to(source.device)
        EOS_flag = torch.BoolTensor(batch_size).fill_(False).to(source.device)
        EOS_index = torch.LongTensor(batch_size).fill_(max_length).to(source.device)
        total_weight = torch.FloatTensor().to(source.device)
        flag = False
        mask_src = (source == self.PAD).unsqueeze(1)
        z_p = self.w_z_p(z)
        for i in range(max_length):
            out_embed, _ = self.len_embed(sentence[:, -1:])
            inputs = torch.cat((out_embed, z), dim=-1)
            outputs, src_context = self.decoder(inputs, src_context)
            attn_vector, weight = self.attn(outputs, src_outputs, mask_src)
            total_weight = torch.cat((total_weight, weight), dim=1)
            p_g = torch.sigmoid(self.w_h_p(attn_vector) + self.w_x_p(outputs) 
                                + self.w_s_p(out_embed) + z_p + self.b_ptr)
            outputs = F.softmax(self.w_out(torch.cat((outputs, attn_vector), dim=-1)), dim=-1)
            outputs = self.copy(outputs, weight, p_g, source)
            prob = F.softmax(outputs, dim=-1)
            word = prob.max(dim=-1)[1]
            sentence = torch.cat((sentence, word), dim=-1)
            mask = (word == self.EOS).view(-1).masked_fill_(EOS_flag, False)
            EOS_flag |= mask
            EOS_index.masked_fill_(mask, i + 1)
            if EOS_flag.sum() == batch_size and flag:
                break
            flag = True
        sent = sentence.tolist()
        for i in range(batch_size):
            sent[i] = sent[i][1:int(EOS_index[i].item())]
        return sent, total_weight


def vae(vocab_size, embed_size, hidden_size, num_layers, 
        dropout_embed, dropout_rnn, latent_num, PAD, BOS, EOS):

    encoder = lstm(embed_size, hidden_size, num_layers, True, dropout_rnn)
    decoder = lstm(embed_size+latent_num, hidden_size, num_layers, False, dropout_rnn)
    attn = MultiHeadAttention(hidden_size, hidden_size // 8, hidden_size // 8, 8)
    vae =  Model(vocab_size, embed_size, dropout_embed, deepcopy(encoder), deepcopy(encoder), 
                 decoder, attn, latent_num, PAD, BOS, EOS)
    for p in vae.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return vae