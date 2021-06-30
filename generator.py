from    utils.preprocess import translate2word
import  torch
from    torch import LongTensor
import  os
import  shutil
import  argparse
import  math
from    utils.preprocess import lang, get_dataloader
import  utils.Constants as Constants


def read_file(file):
    with open(file, 'r') as f:
        data = [line.lower().strip('\n').split() for line in f.readlines()]
        return data


def save2file(data, file):
    path = os.path.join(*os.path.split(file)[:-1])
    if not os.path.exists(path):
        os.makedirs(path)
    with open(file, 'w') as f:
        for seq in data:
            f.write(' '.join(seq))
            f.write('\n')


class generator:

    def __init__(self, test_data, eval, word2index, index2word, 
                 source_path, UNK_WORD, PAD):
        self.test_data = test_data
        self.eval = eval
        self.UNK_WORD = UNK_WORD
        self.PAD = PAD
        self.word2index = word2index
        self.index2word = index2word
        self.source_word = read_file(source_path)
        self.init_flag = True

    def restore_UNK(self, sentence, weight, src_index):
        for i in range(len(sentence)):
            for j in range(len(sentence[i])):
                if sentence[i][j] == self.UNK_WORD:
                    index = weight[i][j]
                    sentence[i][j] = str(src_index[i][index])
        return sentence

    def round(self, test_data, src_index, model, max_length, device, alpha=1):
        model.eval()

        with torch.no_grad():
            outputs = []
            total_weight = []
            for source in test_data:
                source = source.to(device)
                src_len = (source != self.PAD).sum(dim=-1).max().item()
                source = source[:, :src_len]
                sent, weight = model.generate(source, max_length, alpha)
                total_weight.extend(weight.max(-1)[1].tolist())
                sent = translate2word(sent, self.index2word)
                outputs.extend(sent)
        return self.restore_UNK(outputs, total_weight, src_index)
    
    def __call__(self, model, max_length, num_rounds, device, 
                 save_path, save_info, split_save=True):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        
        path = os.path.join(*os.path.split(save_info)[:-1])

        if self.init_flag:
            self.init_flag = False
            if os.path.exists(path) is False:
                os.makedirs(path)
            
            for i in range(num_rounds):
                file = os.path.join(save_info, 'roungd_%d_score.txt' % (i + 1))
                if os.path.exists(file):
                    os.remove(file)
        test_data = self.test_data
        src_index = self.source_word
        for i in range(num_rounds):
            alpha = 1
            alpha = math.e ** (-i / num_rounds)
            outputs = self.round(test_data, src_index, model, max_length, device, alpha)
            file_path = os.path.join(save_path, 'round_%d.txt' % (i + 1))
            save2file(outputs, file_path)
            eval_info = self.eval(file_path)
            file = 'roungd_%d_score.txt' % (i + 1) if split_save else 'score.txt'
            with open(os.path.join(save_info, file), 'a') as f:
                f.write(eval_info)
                f.write('\n')

            new_data = lang(filelist=[file_path],
                            word2index=self.word2index,
                            PAD=Constants.PAD_WORD)

            test_data = get_dataloader(source=new_data, 
                                       batch_size=self.test_data.batch_size,
                                       shuffle=False)                                

            src_index = read_file(file_path)


def get_args():
    parser = argparse.ArgumentParser(prog='Generate Module',
                                     description='Run a generation process.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA.')
    parser.add_argument('--cuda_num', type=str, default='0', nargs='+',
                        help='Choose num of graphic device.')
    parser.add_argument('--source', type=str, nargs='+',
                        help='Path of source file.')
    parser.add_argument('--target', type=str, nargs='+', default=None,
                        help='Path of target file.')
    parser.add_argument('--vocab_path', type=str,
                        help='Path of vocab file.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size of train data.')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Num rounds to paraphrase.')
    parser.add_argument('--max_length', type=int, default=50, 
                        help='Max lenght for generate sequence.')
    parser.add_argument('--model_path', type=str, default='./',
                        help='Path to load model.')    
    parser.add_argument('--save_path', type=str, default='./',
                        help='Path to save generation data.')    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_num)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    from    utils.utils import load_vocab
    from    utils.makeModel import get_vae
    from    generator import generator
    from    utils.eval import Eval

    word2index, index2word = load_vocab(args.vocab_path)
    source = lang(filelist=args.source,
                  word2index=word2index,
                  PAD=Constants.PAD_WORD)

    dataloader = get_dataloader(source=source, 
                                batch_size=args.batch_size,
                                shuffle=False)                                

    model = get_vae(vocab_size=len(word2index),
                    device=device,
                    checkpoint_path=args.model_path)
    
    eval = Eval(args.source[0], args.target[0]) if args.target is not None else None

    generator = generator(test_data=dataloader, 
                          eval=eval,
                          word2index=word2index,
                          index2word=index2word, 
                          source_path=args.source[0], 
                          UNK_WORD=Constants.UNK_WORD, 
                          PAD=Constants.PAD)
    
    generator(model=model, 
              max_length=args.max_length, 
              num_rounds=args.num_rounds, 
              device=device, 
              save_path=args.save_path,
              save_info=args.save_path,
              split_save=False)
