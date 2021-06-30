from    nltk.translate.bleu_score import corpus_bleu

class Eval:

    def __init__(self, source_file, reference_file):
        self.source = self.read_file(source_file, reference=False)
        self.source_ref = self.read_file(source_file, reference=True)
        self.reference = self.read_file(reference_file, reference=True)

    def read_file(self, file, reference=False):
        with open(file, 'r') as f:
            if reference:
                data = [[[word.lower() for word in seq.strip('\n').split()] 
                          for seq in line.strip('\n').split('\t')] for line in f.readlines()]
            else:
                data = [[word.lower() for word in line.strip('\n').split()] for line in f.readlines()]
        return data
    def bleu(self, reference, candidate):
        bleu4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)) * 100
        return bleu4

    def unc(self, source, candidate):
        cnt = 0
        for i in range(len(candidate)):
            if candidate[i] == source[i]:
                cnt += 1
        return cnt / len(candidate) * 100
    
    def __call__(self, candidate_file):
        candidate = self.read_file(candidate_file, reference=False)
        bleu = self.bleu(self.reference, candidate)
        self_bleu = self.bleu(self.source_ref, candidate)
        unc = self.unc(self.source, candidate)
        result = 'BLEU: %.2f\tself-BLEU: %.2f\tiBLEU: %.2f\tunc: %.2f' % (bleu, self_bleu, 0.7*bleu - 0.3*self_bleu, unc)
        print(result)
        return result

if __name__ == '__main__':
    source = '../data/wikiAnswer/test_src.txt'
    target = '../data/wikiAnswer/test_tgt.txt'
    eval = Eval(source, target)
    eval(source)