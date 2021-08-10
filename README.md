# BTmPG
Code for paper [Pushing Paraphrase Away from Original Sentence: A Multi-Round Paraphrase Generation Approach](https://aclanthology.org/2021.findings-acl.135/) by Zhe Lin, Xiaojun Wan. This paper is accepted by Findings of ACL'21. Please contact me at [linzhe@pku.edu.cn](mailto:linzhe@pku.edu.cn) for any question.

## Dependencies
```
PyTorch 1.4
NLTK 3.5
```

## Model

<img src="https://github.com/L-Zhe/BTmPG/blob/main/img/model.jpg?raw=true" width = "800" alt="overview" align=center />

## Create Vocabulary

You should first create a vocabulary from your corpora. You can use the following command.

```shell
python createVocab.py --file ~/context/train.tgt ~/context/train.src  \
                      --save_path ~/context/vocab.pkl \
                      --vocab_num 50000
```

## Train

You can train your model leveraged the following command:

``` shell
python train.py --cuda --cuda_num 5 \
                --train_source ~/context/train.src \
                --train_target ~/context/train.tgt \
                --test_source  ~/context/test.src \
                --test_target  ~/context/test.tgt \
                --vocab_path ~/context/vocab.pkl \
                --batch_size 32\
                --epoch 100 \
                --num_rounds 2 \
                --max_length 110 \
                --clip_length 100 \
                --model_save_path ~/context/output/model.pth \
                --generation_save_path ~/context/output
```

## Inference

After training, you can leverage the following command to generate multi-round paraphrase.

``` shell
python generator.py --cuda      --cuda_num 3 \
                    --source ~/context/test.src \
                    --target ~/context/test.tgt \
                    --vocab_path ~/context/vocab.pkl \
                    --batch_size 64 \
                    --num_rounds 10 \
                    --max_length 60 \
                    --model_path ~/context/model.pth \
                    --save_path ~/context/output/
```

We also provide the pretrain-model file in releases page.

## Result

<img src="https://github.com/L-Zhe/BTmPG/blob/main/img/result1.jpg?raw=true" width = "800" alt="overview" align=center />

<img src="https://github.com/L-Zhe/BTmPG/blob/main/img/result2.jpg?raw=true" width = "500" alt="overview" align=center />

## Case Study

<img src="https://github.com/L-Zhe/BTmPG/blob/main/img/case_study.jpg?raw=true" width = "400" alt="overview" align=center />

## Reference

If you use any content of this repo for your work, please cite the following bib entry:

```
@inproceedings{lin-wan-2021-pushing,
    title = "Pushing Paraphrase Away from Original Sentence: A Multi-Round Paraphrase Generation Approach",
    author = "Lin, Zhe  and
      Wan, Xiaojun",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.135",
    doi = "10.18653/v1/2021.findings-acl.135",
    pages = "1548--1557",
}
```
