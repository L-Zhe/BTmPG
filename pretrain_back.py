import  torch
from    torch import optim
import  os
from    utils.utils import load_vocab
from    utils.makeModel import get_transformer
import  utils.config as config
import  utils.Constants as Constants
from    utils.preprocess import lang, get_pretrain_dataloader
from    utils.run import pre_train_fit
from    utils.utils import show_info
from    model.Module import WarmUpOpt, LabelSmoothing
import  argparse


def get_args():
    parser = argparse.ArgumentParser(prog='Pretrain Module',
                                     description='Run a pretrain model.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA.')
    parser.add_argument('--cuda_num', type=str, default='0',
                        help='Choose num of graphic device.')
    parser.add_argument('--train_source', type=str, nargs='+',
                        help='Path of source file.')
    parser.add_argument('--train_target', type=str, nargs='+', 
                        help='Path of target file.')
    parser.add_argument('--vocab_path', type=str,
                        help='Path of vocab file.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size of train data.')
    parser.add_argument('--epoch', type=int, default=30, 
                        help='The total epoch for train process.')
    parser.add_argument('--max_length', type=int, default=50, 
                        help='Max lenght for generate sequence.')
    parser.add_argument('--clip_length', type=int, default=20,
                        help='Cut the train sentence to specify length.')
    parser.add_argument('--model_save_path', type=str, default='./',
                        help='Path to save pretrain model.')
    
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    word2index, _ = load_vocab(args.vocab_path)
                                                                                  
    show_info(epoch=args.epoch,
              vocab_size=len(word2index),
              USE_CUDA=args.cuda)

    train_source = lang(filelist=args.train_target, 
                        word2index=word2index, 
                        PAD=Constants.PAD_WORD,
                        EOS=Constants.EOS_WORD,
                        max_len=args.clip_length)

    train_target_inputs = lang(filelist=args.train_source,
                               word2index=word2index,
                               PAD=Constants.PAD_WORD,
                               BOS=Constants.BOS_WORD,
                               max_len=args.clip_length)
                               
    train_target_outputs = lang(filelist=args.train_source,
                                word2index=word2index,
                                PAD=Constants.PAD_WORD,
                                EOS=Constants.EOS_WORD,
                                max_len=args.clip_length)
    
    train_dataloader = get_pretrain_dataloader(source=train_source,
                                               tgt_input=train_target_inputs,
                                               tgt_output=train_target_outputs,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    model = get_transformer(vocab_size=len(word2index),
                            device=device)
    criterion = LabelSmoothing(smoothing=config.smoothing,
                               ignore_index=Constants.PAD).to(device)
    optimizer = WarmUpOpt(optimizer    = optim.Adam(params=model.parameters(), 
                                               lr=config.learning_rate,
                                               betas=(config.beta_1, config.beta_2),
                                               eps=config.eps,
                                               weight_decay=config.weight_decay),
                          d_model      = config.transformer_embedding_dim,
                          warmup_steps = config.warmup_steps,
                          factor       = config.factor)
    
    fit = pre_train_fit(train_data=train_dataloader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        epoch=args.epoch,
                        device=device,
                        clip_grad=config.gradient_clipper,
                        BOS=Constants.BOS,
                        EOS=Constants.EOS,
                        PAD=Constants.PAD)
    fit(model_save_path=args.model_save_path)