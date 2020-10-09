# Quasi-Multitask-Learning

In Quasi-Multitask Learning we have a multi task learning architecture, but the tasks are the same.
The average of their predictions tends to be a better predictor than the models by themselves.
This behaviour is similar to an ensemble of classifiers, without the massive computational overhead.

This code was forked from [Barabara Plank's bilstm-aux repository](https://github.com/bplank/bilstm-aux).
Please note, while the original code was intended for (traditional) multitask learning use, 
our modification was not prepared for it. Using that feature may result in undesired behaviour.

## How to run (examples)

k=5 on Tamil pos tagging dataset:
```
python src/qmtl.py --num-out-layers 5 --dynet-mem 1000 --pred_layer 1 --model-to-run all --iters 20 --test data/ta_ttb-ud-test.conllu --train data/ta_ttb-ud-train.conllu --dev data/ta_ttb-ud-dev.conllu --embeds embeddings/polyglot-ta.vec.gz --mlp 20 --output my_model --save my_model --dynet-seed 1
```

k=10 on English pos tagging dataset:
```
python src/qmtl.py --num-out-layers 10 --dynet-mem 1000 --pred_layer 1 --model-to-run all --iters 20 --test data/en_lines-ud-test.conllu --train data/en_lines-ud-train.conllu --dev data/en_lines-ud-dev.conllu --embeds embeddings/polyglot-en.vec.gz --mlp 20 --output my_model --save my_model --dynet-seed 1
```
For Windows users: the .conllu files will be converted to CRLF format, this needs to be converted to LF for the parser to recognise it 
(one easy way to do it is by changing the file ending in notepad++).
