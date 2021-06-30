from    utils.preprocess import create_vocab
import  argparse
from    utils.utils import save_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vocabulary.',
                                     prog='creat_vocab')

    parser.add_argument('-f', '--file', type=str, nargs='+',
                        help='File list to generate vocabulary.')
    parser.add_argument('--vocab_num', type=int, nargs='?', default=-1, 
                        help='Total number of word in vocabulary.')
    parser.add_argument('--save_path', type=str, default='./',
                        help='Path to save vocab.')
                                                
    args = parser.parse_args()
    word2index, index2word = create_vocab(file_list=args.file,
                                          vocab_num=args.vocab_num)

    save_vocab(word2index, index2word, args.save_path)