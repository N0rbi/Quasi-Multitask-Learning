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
<<<<<<< HEAD

## How to interpret

The accuracies of the different output layers are evaluated on the dev set for each epoch separated by tabs. 
The last element of each line is the Q-MTL accuracy (which is basically an ensemble created from the previous elements).
This holds for the test set as well.

Example (dev):
```
dev accuracy: 0.9270	0.9257	0.9257	0.9258	0.9248	0.9261	0.9271	0.9251	0.9247	0.9254	0.9270
```

Where the 0.9270 value is calculated by averaging over the output distributions of the previous 10 models.

Example (test):
```
Task0 test accuracy on 11 items: 0.9396	0.9382	0.9388	0.9386	0.9377	0.9400	0.9401	0.9399	0.9375	0.9374	0.9395
```
Like in the previous case, the last element (0.9395) is the Q-MTL prediction, the other ones are 
the individual tasks (trained on the same dataset).

## Note for Windows users
The .conllu files will be converted to CRLF format, this needs to be converted to LF for the parser to recognise it 
(one easy way to do it is by changing the file ending in notepad++).