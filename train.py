import  torch
import  os
from    torch import nn, optim
from    utils.utils import load_vocab
from    utils.makeModel import get_vae, get_transformer
import  utils.config as config
import  utils.Constants as Constants
from    utils.preprocess import lang, get_dataloader
from    utils.run import Fit
from    model.Module import WarmUpOpt, LabelSmoothing
from    utils.utils import show_info
from    model.VAE import CopyLoss
import  argparse
from    generator import generator
from    utils.eval import Eval
from    itertools import chain


def get_args():
    parser = argparse.ArgumentParser(prog='Pretrain Module',
                                     description='Run a pretrain model.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA.')
    parser.add_argument('--cuda_num', type=str, default='0', nargs='+',
                        help='Choose num of graphic device.')
    parser.add_argument('--train_source', type=str, nargs='+',
                        help='Path of source file.')
    parser.add_argument('--train_target', type=str, nargs='+', 
                        help='Path of target file.')
    parser.add_argument('--test_source', type=str, nargs='+',
                        help='Path to save pretrain model.')
    parser.add_argument('--test_target', type=str, nargs='+',
                        help='Path to save pretrain model.')
    parser.add_argument('--vocab_path', type=str,
                        help='Path of vocab file.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size of train data.')
    parser.add_argument('--epoch', type=int, default=30, 
                        help='The total epoch for train process.')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Num rounds to paraphrase.')
    parser.add_argument('--max_length', type=int, default=50, 
                        help='Max lenght for generate sequence.')
    parser.add_argument('--clip_length', type=int, default=20,
                        help='Cut the train sentence to specify length.')
    parser.add_argument('--model_save_path', type=str, default='./',
                        help='Path to save model.')
    parser.add_argument('--generation_save_path', type=str, default='./',
                        help='Path to save generation data.')    
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_num)
        if len(args.cuda_num) == 1:
            device_para = torch.device('cuda:0')
            device_back = torch.device('cuda:0')
        else:
            device_para = torch.device('cuda:0')
            device_back = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    word2index, index2word = load_vocab(args.vocab_path)
                                                                                  
    show_info(epoch=args.epoch,
              vocab_size=len(word2index),
              USE_CUDA=args.cuda)

    train_source = lang(filelist=args.train_source, 
                        word2index=word2index, 
                        PAD=Constants.PAD_WORD,
                        max_len=args.clip_length)
    
    train_target = lang(filelist=args.train_target,
                        word2index=word2index, 
                        PAD=Constants.PAD_WORD,
                        max_len=args.clip_length)

    train_source_inputs = lang(filelist=args.train_source,
                               word2index=word2index,
                               PAD=Constants.PAD_WORD,
                               BOS=Constants.BOS_WORD,
                               max_len=args.clip_length)
                               
    train_source_outputs = lang(filelist=args.train_source,
                                word2index=word2index,
                                PAD=Constants.PAD_WORD,
                                EOS=Constants.EOS_WORD,
                                max_len=args.clip_length)  

    train_target_inputs = lang(filelist=args.train_target,
                               word2index=word2index,
                               PAD=Constants.PAD_WORD,
                               BOS=Constants.BOS_WORD,
                               max_len=args.clip_length)
                               
    train_target_outputs = lang(filelist=args.train_target,
                                word2index=word2index,
                                PAD=Constants.PAD_WORD,
                                EOS=Constants.EOS_WORD,
                                max_len=args.clip_length)
    
    train_dataloader = get_dataloader(source=train_source,
                                      target=train_target,
                                      src_inputs=train_source_inputs,
                                      src_outputs=train_source_outputs,
                                      tgt_inputs=train_target_inputs,
                                      tgt_outputs=train_target_outputs,
                                      batch_size=args.batch_size,
                                      shuffle=True)

    test_source = lang(filelist=args.test_source, 
                       word2index=word2index, 
                       PAD=Constants.PAD_WORD)   

    test_dataloader = get_dataloader(source=test_source, 
                                     batch_size=args.batch_size,
                                     shuffle=False)                                

    para_model = get_vae(vocab_size=len(word2index),
                         device=device_para)

    back_model = get_transformer(vocab_size=len(word2index),
                                 device=device_back)

    para_optim = optim.Adam(params=para_model.parameters(), 
                            lr=config.learning_rate,
                            betas=(config.beta_1, config.beta_2),
                            eps=config.eps,
                            weight_decay=config.weight_decay)
                               
    optim = WarmUpOpt(optimizer    = optim.Adam(params=chain(para_model.parameters(),
                                                             back_model.parameters()),
                                       lr=config.learning_rate,
                                       betas=(config.beta_1, config.beta_2),
                                       eps=config.eps,
                                       weight_decay=config.weight_decay),
                      d_model      = config.transformer_embedding_dim,
                      warmup_steps = config.warmup_steps,
                      factor       = config.factor)


    criterion_para = CopyLoss(ignore_index=Constants.PAD,
                              reduction='none').to(device_para)
    criterion_back = LabelSmoothing(smoothing=config.smoothing,
                                    ignore_index=Constants.PAD).to(device_back)

    generator = generator(test_data=test_dataloader, 
                          source_path=args.test_source[0], 
                          eval=Eval(args.test_source[0], args.test_target[0]), 
                          word2index=word2index,
                          index2word=index2word, 
                          UNK_WORD=Constants.UNK_WORD, 
                          PAD=Constants.PAD)

    fit = Fit(train_data=train_dataloader,
              para_model=para_model,
              back_model=back_model,
              optim=optim,
              criterion_para=criterion_para,
              criterion_back=criterion_back,
              generator=generator,
              num_rounds=args.num_rounds,
              epoch=args.epoch,
              device_para=device_para,
              device_back=device_back,
              clip_grad=config.gradient_clipper,
              max_length=args.max_length,
              BOS=Constants.BOS,
              EOS=Constants.EOS,
              PAD=Constants.PAD)
    fit(model_save_path=args.model_save_path,
        generation_save_path=args.generation_save_path)