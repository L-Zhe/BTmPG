import  torch
from    torch.nn import functional as F
import  time
from    .utils import save_model
from    model.gumbleSoftmax import gumble_softmax
import  os
import  math

def cal_coef(word):
    count = {}
    device = word.device
    N = word.size(0)
    word = word.tolist()
    for w in word:
        if count.get(w) is None:
            count[w] = 0
        count[w] += 1
    coef = torch.ones(N).to(device)
    for i in range(N):
        coef[i] = math.log2(N / count[word[i]] * 2)

    return coef


class Fit:

    def __init__(self, *args, **kwargs):
        
        self.train_data  = kwargs['train_data']
        self.para_model  = kwargs['para_model']  
        self.back_model  = kwargs['back_model']
        self.criterion_para   = kwargs['criterion_para']
        self.criterion_back   = kwargs['criterion_back']
        self.optim  = kwargs['optim']
        self.BOS         = kwargs['BOS']
        self.EOS         = kwargs['EOS']
        self.PAD         = kwargs['PAD']
        self.device_para = kwargs['device_para'] 
        self.device_back = kwargs['device_back']
        self.num_rounds  = 1 if kwargs.get('num_rounds') is None else kwargs['num_rounds']
        self.EPOCH       = 30 if kwargs.get('epoch') is None else kwargs['epoch']
        self.clip_grad   = None if kwargs.get('clip_grad') is None else kwargs['clip_grad']
        self.batchPrintInfo = 500 if kwargs.get('batchPrintInfo') is None else kwargs['batchPrintInfo']
        self.generator = None if kwargs.get('generator') is None else kwargs['generator']
        self.max_length = 30 if kwargs.get('max_length') is None else kwargs['max_length']
        self.GS = gumble_softmax(3500, 100)

    def run(self):
        self.para_model.train()
        self.back_model.train()
        loss_recons = 0
        loss_kl = 0
        loss_para = 0
        cnt = 0
        cnt_tok = 0
        st_time = time.time()
        for i, (source, target, src_inputs, src_outputs, tgt_inputs, tgt_outputs) in enumerate(self.train_data):
            self.optim.zero_grad()
            # prepare data
            tgt_len = (target != self.PAD).sum(-1).max().item()
            src_len = (source != self.PAD).sum(dim=-1).max().item()
            source = source[:, :src_len]
            target = target[:, :tgt_len]
            src_inputs = src_inputs[:, :src_len + 1]
            src_outputs = src_outputs[:, :src_len + 1]
            tgt_inputs = tgt_inputs[:, :tgt_len + 1]
            tgt_outputs = tgt_outputs[:, :tgt_len + 1]
            source = source.to(self.device_para)
            target = target.to(self.device_para)
            src_inputs = src_inputs.to(self.device_back)
            src_outputs = src_outputs.to(self.device_back)
            tgt_inputs = tgt_inputs.to(self.device_para)
            tgt_outputs = tgt_outputs.to(self.device_para)
            ntoken = (source != self.PAD).sum().item()
            norm = source.size(0)
            cnt_tok += ntoken
            cnt += norm

            # calculate first round
            source, kl_div, h_pre = self.para_model(source=source,
                                                    paraphrase=target,
                                                    tgt_inputs=tgt_inputs)
            loss_pa = self.criterion_para(source, tgt_outputs).to(self.device_back)
            loss_kl += kl_div.item()
            loss_pa[:, 0] *= cal_coef(tgt_outputs[:, 0])
            loss_pa = loss_pa.sum()
            loss_para += loss_pa.item()
            loss = (loss_pa + kl_div.to(self.device_back)) / norm

            outputs = self.back_model(source=target.to(self.device_back),
                                      target=src_inputs)
            loss_re = self.criterion_back(outputs, src_outputs).to(self.device_back)
            loss_recons += loss_re.item()
            loss += loss_re / norm
            
            # source = outputs.max(dim=-1)[1]
            # source = source[:, :max((source != self.PAD).sum(-1).max(), 1)]
            outputs, h_now = self.para_model.round(source=source,
                                                   paraphrase=target,
                                                   max_length=self.max_length)
            outputs = self.back_model(source=self.GS(outputs.to(self.device_back)),
                                      target=src_inputs)
            loss_back = self.criterion_back(outputs, src_outputs).to(self.device_back)
            loss_recons += loss_back.item()
            loss += loss_back / norm
            # d_model = h_pre.size(-1)
            # penalty = (h_pre - h_now).pow(2).sum(-1).pow(0.5).sum() / (d_model**0.5)
            # loss -= penalty.to(self.device_back) / norm
            loss.backward()
            self.GS.step_n()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.para_model.parameters(), self.clip_grad)
                torch.nn.utils.clip_grad_norm_(self.back_model.parameters(), self.clip_grad)
            self.optim.step()
            if i % self.batchPrintInfo == 0:
                total_time = time.time() - st_time
                st_time = time.time()
                print('Batch: %d\t reconst_loss: %f\tkl_div_loss: %f\tpara_loss: %f\tTok pre Sec: %d\t\tTime: %d' % 
                      (i, loss_recons/(cnt * 2), loss_kl/cnt, loss_para/cnt, cnt_tok/total_time, total_time))

                loss_recons = 0
                loss_para = 0
                loss_kl = 0
                cnt = 0
                cnt_tok = 0

    def __call__(self, model_save_path=None, 
                 generation_save_path=None):
        for epoch in range(self.EPOCH):
            print('+' * 80)
            print('EPOCH: %d' % (epoch + 1))
            print('-' * 80)
            self.run()
            save_model(self.para_model, model_save_path)
            if self.generator is not None:
                assert generation_save_path is not None
                self.generator(model=self.para_model, 
                               max_length=30, 
                               num_rounds=10, 
                               device=self.device_para, 
                               save_path=os.path.join(generation_save_path, 'epoch_%d' % (epoch + 1)),
                               save_info=generation_save_path)


class pre_train_fit:

    def __init__(self, *args, **kwargs):
        
        self.train_data = kwargs['train_data']
        self.model      = kwargs['model']
        self.criterion  = kwargs['criterion']
        self.optimizer  = kwargs['optimizer']
        self.BOS        = kwargs['BOS']
        self.EOS        = kwargs['EOS']
        self.PAD        = kwargs['PAD']
        self.device     = torch.device('cpu') if kwargs.get('device') is None else kwargs['device'] 
        self.EPOCH      = 30 if kwargs.get('epoch') is None else kwargs['epoch']
        self.clip_grad  = None if kwargs.get('clip_grad') is None else kwargs['clip_grad']
        self.batchPrintInfo = 500 if kwargs.get('batchPrintInfo') is None else kwargs['batchPrintInfo']
        self.generator = None if kwargs.get('generator') is None else kwargs['generator']
    
    def run(self):

        self.model.train()
        cnt_loss = 0
        total_loss = 0
        cnt = 0
        cnt_tok = 0
        total_num = 0
        st_time = time.time()
        for i, (source, tgt_inputs, tgt_outputs) in enumerate(self.train_data):
            self.optimizer.zero_grad()
      
            src_len = (source != self.PAD).sum(dim=-1).max().item()
            tgt_len = (tgt_inputs != self.PAD).sum(dim=-1).max().item()
            source = source[:, :src_len]
            tgt_inputs = tgt_inputs[:, :tgt_len]
            tgt_outputs = tgt_outputs[:, :tgt_len]
            source = source.to(self.device)
            tgt_inputs = tgt_inputs.to(self.device)
            tgt_outputs = tgt_outputs.to(self.device)

            outputs = self.model(source=source, 
                                 target=tgt_inputs)
            ntoken = (source != self.PAD).sum().item()
            cnt_tok += ntoken
            norm = (tgt_inputs != self.PAD).sum().item()
            loss = self.criterion(outputs, tgt_outputs, norm)
            
            total_loss += loss.item() * norm
            cnt_loss += loss.item() * norm
            cnt += source.size(0)
            total_num += source.size(0)
            loss.backward()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            if i % self.batchPrintInfo == 0:
                total_time = time.time() - st_time
                st_time = time.time()
                print('Batch: %d\tloss: %f\tTok pre Sec: %d\t\tTime: %d'
                      % (i, cnt_loss/cnt, cnt_tok/total_time, total_time))
                cnt_loss = 0
                cnt = 0
                cnt_tok = 0

        return total_loss / total_num

    def __call__(self, model_save_path=None, 
                 generation_save_path=None):
        
        for epoch in range(self.EPOCH):
            print('+' * 80)
            print('EPOCH: %d' % (epoch + 1))
            print('-' * 80)
            self.run()
            save_model(self.model, model_save_path)