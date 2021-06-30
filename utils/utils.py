import  os
import  pickle
import  torch

def save_vocab(word2index, index2word, save_path):
    vocab = {'word2idx':  word2index,
             'idx2word':  index2word}
    
    file_path = os.path.join(*os.path.split(save_path)[:-1])
    if os.path.exists(file_path) == False:
        os.makedirs(file_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('===> Save Vocabulary Successfully.')

def load_vocab(save_path):
    with open(save_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab['word2idx'], vocab['idx2word']

def save_model(model, save_path):
    file_path = os.path.join(*os.path.split(save_path)[:-1])
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
    torch.save(model.state_dict(), save_path)

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model

def show_info(epoch, vocab_size, USE_CUDA):
    print('+' * 22)
    print('EPOCH: \t\t%d' % epoch)
    print('Vocab Size: \t%d' % vocab_size)
    print('USE_CUDA:\t', USE_CUDA)
