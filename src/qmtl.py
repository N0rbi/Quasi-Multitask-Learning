#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A neural network based tagger (bi-LSTM)
- hierarchical (word embeddings plus lower-level bi-LSTM for characters)
- supports MTL
"""
import argparse
import random
import time
import sys
import numpy as np
import os
import pickle
import dynet
import codecs
import heterogenious_output_utils
import json

from sklearn.model_selection import train_test_split

from collections import Counter, defaultdict, Sequence
from lib.mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor
from lib.mio import read_conll_file, read_conllUD_file, load_embeddings_file
from itertools import product
import logging

UNK = "_UNK"

PREDICT_ON_LAYER = None


from lib.mmappers import TRAINER_MAP, ACTIVATION_MAP, INITIALIZER_MAP, BUILDERS


def dump_frobenius_values(tagger):

    fro_W = [0 for i in tagger.predictors['output_layers_dict']['task0']]
    for seq_idx, seq_pred in enumerate(tagger.predictors['output_layers_dict']['task0']):
        fro_W[seq_idx] = np.linalg.norm(seq_pred.network_builder.W.value(), 'fro')
    print("Fro W: %s" % '\t'.join([str(num) for num in fro_W]), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="""Run the NN tagger""")
    parser.add_argument("--train", nargs='*', help="train folder for each task") # allow multiple train files, each asociated with a task = position in the list
    parser.add_argument("--pred_layer", nargs='*', help="layer of predictons for each task", default=1) # for each task the layer on which it is predicted (default 1)
    parser.add_argument("--model", help="load model from file", required=False)
    parser.add_argument("--iters", help="training iterations [default: 30]", required=False,type=int,default=30)
    parser.add_argument("--in_dim", help="input dimension [default: 64] (like Polyglot embeds)", required=False,type=int,default=64)
    parser.add_argument("--c_in_dim", help="input dimension for character embeddings [default: 100]", required=False,type=int,default=100)
    parser.add_argument("--h_dim", help="hidden dimension [default: 100]", required=False,type=int,default=100)
    parser.add_argument("--h_layers", help="number of stacked LSTMs [default: 1 = no stacking]", required=False,type=int,default=1)
    parser.add_argument("--test", nargs='*', help="test file(s)", required=False) # should be in the same order/task as train
    parser.add_argument("--raw", help="if test file is in raw format (one sentence per line)", required=False, action="store_true", default=False)
    parser.add_argument("--dev", help="dev file(s)", required=False)
    parser.add_argument("--output", help="output predictions to file", required=False,default=None)
    parser.add_argument("--output-probs", help="output prediction probs to file (last column)", required=False, default=None)
    parser.add_argument("--save", help="save model to file (appends .model as well as .pickle)",default=None)
    parser.add_argument("--embeds", help="word embeddings file", required=False, default=None)
    parser.add_argument("--sigma", help="noise sigma", required=False, default=0.2, type=float)
    parser.add_argument("--ac", help="activation function [rectify, tanh, ...]", default="tanh", choices=ACTIVATION_MAP.keys())
    parser.add_argument("--mlp", help="use MLP layer of this dimension [default 0=disabled]", required=False, default=0, type=int)
    parser.add_argument("--ac-mlp", help="activation function for MLP (if used) [rectify, tanh, ...]", default="rectify", choices=ACTIVATION_MAP.keys())
    parser.add_argument("--trainer", help="trainer [default: sgd]", required=False, choices=TRAINER_MAP.keys(), default="sgd")
    parser.add_argument("--learning-rate", help="learning rate [0: use default]", default=0, type=float) # see: http://dynet.readthedocs.io/en/latest/optimizers.html
    parser.add_argument("--patience", help="patience [default: 0=not used], requires specification of --dev and model path --save", required=False, default=0, type=int)
    parser.add_argument("--log-losses", help="log loss (for each task if multiple active)", required=False, action="store_true", default=False)
    parser.add_argument("--word-dropout-rate", help="word dropout rate [default: 0.25], if 0=disabled, recommended: 0.25 (Kipperwasser & Goldberg, 2016)", required=False, default=0.25, type=float)
    parser.add_argument("--label-noise", help="amount of label noise to be applied [default: 0.0]", required=False, default=0.0, type=float)

    parser.add_argument("--dynet-seed", help="random seed for dynet (needs to be first argument!)", required=False, type=int)
    parser.add_argument("--dynet-mem", help="memory for dynet (needs to be first argument!)", required=False, type=int)
    parser.add_argument("--dynet-gpus", help="1 for GPU usage", default=0, type=int) # warning: non-deterministic results on GPU https://github.com/clab/dynet/issues/399
    parser.add_argument("--dynet-autobatch", help="if 1 enable autobatching", default=0, type=int)
    parser.add_argument("--minibatch-size", help="size of minibatch for autobatching (1=disabled)", default=1, type=int)

    parser.add_argument("--save-embeds", help="save word embeddings file", required=False, default=None)
    parser.add_argument("--disable-backprob-embeds", help="disable backprob into embeddings (default is to update)", required=False, action="store_false", default=True)
    parser.add_argument("--initializer", help="initializer for embeddings (default: constant)", choices=INITIALIZER_MAP.keys(), default="constant")
    parser.add_argument("--builder", help="RNN builder (default: lstmc)", choices=BUILDERS.keys(), default="lstmc")
    # new parameters
    parser.add_argument('--max-vocab-size', type=int, help='the maximum size '
                                                           'of the vocabulary')
    # custom arguments
    parser.add_argument("--num-out-layers", help="redundant layer number at the end of the model", type=int,
                        default=5)
    parser.add_argument("--model-to-run", help="redundant layer number at the end of the model", type=str, default=None)
    parser.add_argument("--training-cutoff", help="what (1/x) portion of the training data is used (default: 1)", type=int, default=1)
    parser.add_argument("--output-builder-query", help="accepts queries with the given form: (activ unit1_num1)xnum_out1 (activ unit2_num2)xnum_out2 ; [WARNING: overrides mlp, ac-mlp and num-out-layers]", type=str, default=None)

    parser.add_argument('--get-model-norm', type=bool, default=False)

    #PTA params
    parser.add_argument('--pta-I', type=int, default=1)
    parser.add_argument('--pta-F', type=int, default=0)
    parser.add_argument('--pta-D', type=float, default=0.0)

    parser.add_argument('--pta-D-Lower', type=float, default=0.0)
    parser.add_argument('--pta-D-Upper', type=float, default=0.0)

    parser.add_argument('--pta-P', type=float, default=0.00)
    parser.add_argument('--pta-H', type=float, default=0.0)
    parser.add_argument('--pta-G', type=int, default=0)

    parser.add_argument('--pta-M', type=int, default=-1)

    args = parser.parse_args()

    output_builder_query = args.output_builder_query if args.output_builder_query else "(%s %d)x%d" % (
        args.ac_mlp, args.mlp, args.num_out_layers)

    assert heterogenious_output_utils.is_query_valid(
        output_builder_query), "You need a valid query for the output generator [--out-builder-query=\"({activ1} {unit_num1})x{num_out1} ({activ} {unit_num2})x{num_out2}\"]"

    print("Output layers query: " + json.dumps(heterogenious_output_utils.get_layer_params(output_builder_query)), file=sys.stderr)


    if args.model_to_run == 'ensemble':
            models = range(args.num_out_layers)
    elif args.model_to_run:
        models = [None if args.model_to_run=="all" else int(args.model_to_run)]
    else:
        models = [None]
        models.extend(range(heterogenious_output_utils.get_output_number(output_builder_query)))

    ensembled_predictions = []
    seed = [args.dynet_seed]
    for current_model, seed in product(models, seed):
        if args.output is not None:
            if os.path.isdir(args.output):
                if not os.path.exists(args.output):
                    os.mkdir(os.path.dirname(args.output))
        save_model = None
        if args.save:
            model_name = str(current_model) if current_model is not None else "all"
            save_model = os.path.join(args.save, model_name)
            model_dirname = os.path.dirname(save_model)
            if not os.path.isdir(model_dirname) and not os.path.exists(model_dirname):
                os.mkdir(model_dirname)

        if args.dev:
            # build computation graph makes it buggy
            assert (args.model or args.save is not None), "Require to save model to --save MODELNAME when --dev is given"

        if args.train:
            if not args.pred_layer:
                print("--pred_layer required!")
                exit()

        if seed or seed == 0:
            print(">>> using seed: {} <<< ".format(seed), file=sys.stderr)
            np.random.seed(seed)
            random.seed(seed)
            dynet.reset_random_seed(args.dynet_seed)

        if args.c_in_dim == 0:
            print(">>> disable character embeddings <<<", file=sys.stderr)

        if args.minibatch_size > 1:
            raise NotImplementedError("Minibatch running is currently not supported")

        if args.patience:
            if not args.dev or not args.save:
                print("patience requires a dev set and model path (--dev and --save)")
                exit()

        if args.save:
            # check if folder exists
            if os.path.isdir(save_model):
                if not os.path.exists(save_model):
                    print("Creating {}..".format(save_model))
                    os.makedirs(save_model)

        if args.output:
                outdir = os.path.dirname(args.output)
                if outdir != "" and not os.path.exists(outdir):
                    os.makedirs(outdir)

        if args.model:
            model_to_load = os.path.join(args.model, str(current_model) if current_model is not None else "all")
            model_to_load = model_to_load.replace('_NEW_', '')
            if args.model_to_run == 'ensemble':
                model_to_load = model_to_load.replace('ensemble', str(current_model))
            print("loading model from file {}".format(model_to_load), file=sys.stderr)
            tagger = load(model_to_load, args.embeds)

            if args.get_model_norm:
                dump_frobenius_values(tagger)
                exit()
        else:
            pta_params = defaultdict()
            pta_params['I'] = args.pta_I == 1
            pta_params['F'] = args.pta_F == 1
            pta_params['D'] = args.pta_D
            pta_params['P'] = args.pta_P
            pta_params['H'] = args.pta_H
            pta_params['G'] = args.pta_G == 1
            pta_params['M'] = args.pta_M
            pta_params['D-Lower'] = args.pta_D_Lower
            pta_params['D-Upper'] = args.pta_D_Upper

            tagger = NNTagger(args.in_dim,
                              args.h_dim,
                              args.c_in_dim,
                              args.h_layers,
                              args.pred_layer,
                              embeds_file=args.embeds,
                              activation=ACTIVATION_MAP[args.ac],
                              noise_sigma=args.sigma,
                              learning_algo=args.trainer,
                              learning_rate=args.learning_rate,
                              backprob_embeds=args.disable_backprob_embeds,
                              initializer=INITIALIZER_MAP[args.initializer],
                              builder=BUILDERS[args.builder],
                              max_vocab_size=args.max_vocab_size,
                              predict_on_layer=current_model,
                              output_builder_query=output_builder_query,
                              pta_params=pta_params,
                              )

        start = time.time()
        if args.train and len(args.train) != 0:
            tagger.fit(args.train, args.iters, args.training_cutoff,
                       dev=args.dev, word_dropout_rate=args.word_dropout_rate,
                       model_path=save_model, patience=args.patience, minibatch_size=args.minibatch_size,
                       log_losses=args.log_losses, label_noise=args.label_noise, build_cg=True)
            print(("Done. Training took {0:.2f} seconds.".format(time.time()-start)),file=sys.stderr)

            if args.save and not args.patience:  # in case patience is active it gets saved in the fit function
                save(tagger, save_model)
                tagger = load(save_model, args.embeds)

            if args.patience:
                tagger = load(save_model, args.embeds)

        if args.test and len(args.test) != 0:
            if not args.model:
                if not args.train:
                    print("specify a model!")
                    sys.exit()

            start = time.time()
            for i, test in enumerate(args.test):

                if args.output is not None and args.model_to_run != 'ensemble':
                    file_pred = "{}.{}_task{}".format(args.output, 'all' if current_model is None else current_model, i)
                    sys.stdout = codecs.open(file_pred, 'w', encoding='utf-8')

                sys.stderr.write('\nTesting Task'+str(i)+'\n')
                sys.stderr.write('*******\n')
                test_X, test_Y, org_X, org_Y, task_labels = tagger.get_data_as_indices(test, "task"+str(i), raw=args.raw)
                correct_list, total_list, predictions = tagger.evaluate(test_X, test_Y, org_X, org_Y, task_labels,
                                                 output_predictions=args.output, output_probs=args.output_probs, raw=args.raw, get_predictions_array=args.model_to_run=='ensemble')

                if args.model_to_run=='ensemble':
                    if current_model==0:
                        ensembled_predictions = [np.array(p) for p in predictions[current_model]]
                    else:
                        for pi, p in enumerate(predictions[current_model]):
                            ensembled_predictions[pi] += np.array(p)

                if not args.raw:
                    test_accuracy = "\t".join(["%.4f"%(0 if total==0 else correct/total) for correct, total in zip(correct_list, total_list)])
                    print("\nTask%s test accuracy on %s items: %s" % (i, len(total_list), test_accuracy), file=sys.stderr)
                    dump_frobenius_values(tagger)

                print(("[{}] Done. Testing took {:.2f} seconds.".format(i, time.time()-start)),file=sys.stderr)

        if args.train:
            print("Info: biLSTM\n\t"+"\n\t".join(["{}: {}".format(a,v) for a, v in vars(args).items()
                                              if a not in ["train","test","dev","pred_layer"]]),file=sys.stderr)
        else:
            # print less when only testing, as not all train params are stored explicitly
            print("Info: biLSTM\n\t" + "\n\t".join(["{}: {}".format(a, v) for a, v in vars(args).items()
                                                    if a not in ["train", "test", "dev", "pred_layer",
                                                                 "initializer","ac","word_dropout_rate",
                                                                 "patience","sigma","disable_backprob_embed",
                                                                 "trainer", "dynet_seed", "dynet_mem","iters"]]),file=sys.stderr)

        if args.save_embeds:
            tagger.save_embeds(args.save_embeds)

    if args.model_to_run=='ensemble':
        task_id = "task0"
        ensemble_file_pred = None
        correct, total = 0, 0
        if args.output:
            ensemble_file_pred = open("{}.ensemble_{}".format(args.output, task_id), "w")
        for pred_per_sentence, tokens_per_sentence, etalon_tags_per_sentence in zip(ensembled_predictions, org_X, org_Y):
            i2t = {tagger.task2tag2idx[task_id][t] : t for t in tagger.task2tag2idx[task_id].keys()}
            predicted_tags = np.argmax(pred_per_sentence, axis=1)
            if ensemble_file_pred:
                for tok, gold, pred_id in zip(tokens_per_sentence, etalon_tags_per_sentence, predicted_tags):
                    total += 1
                    correct += 1 if gold == i2t[pred_id] else 0
                    ensemble_file_pred.write("{}\t{}\t{}\n".format(tok, gold, i2t[pred_id]))
            ensemble_file_pred.write('\n')
        if ensemble_file_pred:
            ensemble_file_pred.close()
        print(args.output, correct / total)


def load(model_path, embeds_file=None):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    myparams = pickle.load(open(model_path+".params.pickle", "rb"))
    query = myparams["output_builder_query"] if "output_builder_query" in myparams \
        else "(%s %d)x%d" % (myparams["activation_mlp"].__name__, myparams["mlp"], myparams["out_num"])
    tagger = NNTagger(myparams["in_dim"],
                      myparams["h_dim"],
                      myparams["c_in_dim"],
                      myparams["h_layers"],
                      myparams["pred_layer"],
                      activation=myparams["activation"],
                      tasks_ids=myparams["tasks_ids"],
                      builder=myparams["builder"],
                      predict_on_layer=myparams["predict_on_layer"],
                      output_builder_query=query,
                      pta_params= myparams['pta_params'],
                      )
    if embeds_file:
        tagger.embeds_file = embeds_file
    tagger.set_indices(myparams["w2i"],myparams["c2i"],myparams["task2tag2idx"])
    tagger.predictors, tagger.char_rnn, tagger.wembeds, tagger.cembeds = \
        tagger.build_computation_graph(myparams["num_words"],
                                       myparams["num_chars"])

    tagger.model.populate(model_path+".model")

    print("model loaded: {}".format(model_path), file=sys.stderr)
    return tagger


def save(nntagger, model_path):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = model_path + ".model"

    open(modelname, "w").close()

    for param in nntagger.model.parameters_list() + nntagger.model.lookup_parameters_list():
        param.save(modelname, append=True)

    myparams = {"num_words": len(nntagger.w2i),
                "num_chars": len(nntagger.c2i),
                "tasks_ids": nntagger.tasks_ids,
                "w2i": nntagger.w2i,
                "c2i": nntagger.c2i,
                "task2tag2idx": nntagger.task2tag2idx,
                "activation": nntagger.activation,
                "in_dim": nntagger.in_dim,
                "h_dim": nntagger.h_dim,
                "c_in_dim": nntagger.c_in_dim,
                "h_layers": nntagger.h_layers,
                "embeds_file": nntagger.embeds_file,
                "pred_layer": nntagger.pred_layer,
                "builder": nntagger.builder,
                "predict_on_layer": nntagger.predict_on_layer,
                "output_builder_query": nntagger.output_builder_query,
                "pta_params": nntagger.pta_params,
                }
    pickle.dump(myparams, open( model_path+".params.pickle", "wb" ) )
    print("model stored: {}".format(modelname), file=sys.stderr)


class NNTagger(object):

    def __init__(self,in_dim,h_dim,c_in_dim,h_layers,pred_layer, learning_algo="sgd", learning_rate=0,
                 embeds_file=None,activation=ACTIVATION_MAP["tanh"],
                 backprob_embeds=True,noise_sigma=0.1, tasks_ids=[],
                 initializer=INITIALIZER_MAP["glorot"], builder=BUILDERS["lstmc"],
                 max_vocab_size=None, predict_on_layer=PREDICT_ON_LAYER,
                 output_builder_query="(%s %d)*%d" % (ACTIVATION_MAP["rectify"], 0, 5), pta_params = defaultdict()):
        self.w2i = {}  # word to index mapping
        self.c2i = {}  # char to index mapping
        self.tasks_ids = tasks_ids # list of names for each task
        self.task2tag2idx = {} # need one dictionary per task
        self.pred_layer = [int(layer) for layer in pred_layer] # at which layer to predict each task
        self.model = dynet.ParameterCollection() #init model
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.activation = activation
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {"inner": [], "output_layers_dict": {}, "task_expected_at": {} } # the inner layers and predictors
        self.wembeds = None # lookup: embeddings for words
        self.cembeds = None # lookup: embeddings for characters
        self.embeds_file = embeds_file
        trainer_algo = TRAINER_MAP[learning_algo]
        if learning_rate > 0:
            self.trainer = trainer_algo(self.model, learning_rate=learning_rate)
        else:
            # using default learning rate
            self.trainer = trainer_algo(self.model)
        self.backprob_embeds = backprob_embeds
        self.initializer = initializer
        self.char_rnn = None # biRNN for character input
        self.builder = builder # default biRNN is an LSTM
        self.max_vocab_size = max_vocab_size

        self.predict_on_layer = predict_on_layer
        self.output_builder_query = output_builder_query
        self.output_builder = heterogenious_output_utils.query_to_dynet_builder(output_builder_query)
        self.out_num = heterogenious_output_utils.get_output_number(output_builder_query)

        if not isinstance(pta_params['D'], list):
            pta_params['D'] = [pta_params['D']] * self.out_num
        self.pta_params = pta_params

        self.train_log = []

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def set_indices(self, w2i, c2i, task2t2i):
        for task_id in task2t2i:
            self.task2tag2idx[task_id] = task2t2i[task_id]
        self.w2i = w2i
        self.c2i = c2i

    def fit(self, list_folders_name, num_iterations, training_fraction, dev=None, word_dropout_rate=0.0, model_path=None, patience=0, minibatch_size=0, log_losses=False, label_noise=0.0, build_cg=True):
        """
        train the tagger
        """
        print("read training data",file=sys.stderr)

        nb_tasks = len( list_folders_name )

        losses = {} # log losses

        train_X, train_Y, task_labels, w2i, c2i, task2t2i = self.get_train_data(list_folders_name)

        train_X = train_X[0:len(train_X)//training_fraction]
        train_Y = train_Y[0:len(train_X)]
        print("{} many training sentences used".format(len(train_X)), file=sys.stderr)
        assert (len(train_X) == len(train_Y))

        ## after calling get_train_data we have self.tasks_ids
        self.task2layer = {task_id: out_layer for task_id, out_layer in zip(self.tasks_ids, self.pred_layer)}
        print("task2layer", self.task2layer, file=sys.stderr)

        # store mappings of words and tags to indices
        self.set_indices(w2i,c2i,task2t2i)

        # if we use word dropout keep track of counts
        if word_dropout_rate > 0.0:
            widCount = Counter()
            for sentence, _ in train_X:
                widCount.update([w for w in sentence])

        if dev:
            if not os.path.exists(dev):
                print('%s does not exist. Using 10 percent of the training '
                      'dataset for validation.' % dev)
                train_X, dev_X, train_Y, dev_Y = train_test_split(
                    train_X, train_Y, test_size=0.1)
                org_X, org_Y = None, None
                dev_task_labels = ['task0'] * len(train_X)
            else:
                dev_X, dev_Y, org_X, org_Y, dev_task_labels = self.get_data_as_indices(dev, "task0")

        # init lookup parameters and define graph
        print("build graph",file=sys.stderr)

        num_words = len(self.w2i)
        num_chars = len(self.c2i)

        assert(nb_tasks==len(self.pred_layer))

        if build_cg:
            self.predictors, self.char_rnn, self.wembeds, self.cembeds = self.build_computation_graph(num_words, num_chars)


        if self.backprob_embeds == False:
            ## disable backprob into embeds (default: True)
            self.wembeds.set_updated(False)
            print(">>> disable wembeds update <<< (is updated: {})".format(self.wembeds.is_updated()), file=sys.stderr)

        train_data = list(zip(train_X,train_Y, task_labels))

        best_val_acc, epochs_no_improvement = 0.0, 0

        if dev and model_path is not None and patience > 0:
            print('Using early stopping with patience of %d...' % patience)

        batch = []

        # DecInit

        output_layers_dict = self.predictors['output_layers_dict']

        if not self.pta_params['I']:  # copy QMTL[0] params to other heads
            for task_id in self.tasks_ids:
                first_builder = output_layers_dict[task_id][0].network_builder
                for i in range(len(output_layers_dict[task_id])):
                    output_layers_dict[task_id][i].network_builder.W.set_value(first_builder.W.value())
                    output_layers_dict[task_id][i].network_builder.b.set_value(first_builder.b.value())

                    if output_layers_dict[task_id][i].network_builder.mlp:
                        output_layers_dict[task_id][i].network_builder.W_mlp.set_value(first_builder.W_mlp.value())
                        output_layers_dict[task_id][i].network_builder.b_mlp.set_value(first_builder.b_mlp.value())

            dynet.renew_cg()

        if self.pta_params['F']:
            for task_id in self.tasks_ids:
                for i in range(len(output_layers_dict[task_id])):
                    if i == 0:
                        continue
                    output_layers_dict[task_id][i].network_builder.W.set_updated(False)
                    output_layers_dict[task_id][i].network_builder.b.set_updated(False)

                    if output_layers_dict[task_id][i].network_builder.mlp:
                        output_layers_dict[task_id][i].network_builder.W_mlp.set_updated(False)
                        output_layers_dict[task_id][i].network_builder.b_mlp.set_updated(False)
            dynet.renew_cg()

        for iter in range(num_iterations):

            total_loss=0.0
            dynet_losses = []
            total_tagged=0.0
            random.shuffle(train_data)

            loss_accum_loss = defaultdict(float)
            loss_accum_tagged = defaultdict(float)

            for batch_num, ((word_indices,char_indices),y, task_of_instance) in enumerate(train_data):

                if word_dropout_rate > 0.0:
                    word_indices = [self.w2i[UNK] if
                                        (random.random() > (widCount.get(w)/(word_dropout_rate+widCount.get(w))))
                                        else w for w in word_indices]

                if task_of_instance not in losses:
                    losses[task_of_instance] = [] #initialize

                if minibatch_size > 1:
                    output = self.predict(word_indices, char_indices, task_of_instance, train=True)
                    total_tagged += len(word_indices)

                    loss1 = dynet.esum([self.pick_neg_log(pred,gold) for pred, gold in zip(output, y)])
                    batch.append(loss1)
                    if len(batch) == minibatch_size:
                        loss = dynet.esum(batch)
                        dynet_losses.append(loss)
                        total_loss += loss.value()

                        # logging
                        loss_accum_tagged[task_of_instance] += len(word_indices)
                        loss_accum_loss[task_of_instance] += loss.value()

                        loss.backward()
                        self.trainer.update()
                        dynet.renew_cg()  # use new computational graph for each BATCH when batching is active
                        batch = []
                else:
                    dynet.renew_cg() # new graph per item
                    output_list = self.predict(word_indices, char_indices, task_of_instance, train=True)
                    total_tagged += len(word_indices)
                    loss_avg = []
                    loss_objts = []
                    y = [np.random.randint(len(self.task2tag2idx[task_of_instance])) if b else v for (v,b) in zip(y, np.random.rand(len(y)) < label_noise)]


                    for layer, output in enumerate(output_list):
                        loss1 = dynet.esum([self.pick_neg_log(pred,gold) for pred, gold in zip(output, y)])
                        lv = loss1.value()
                        loss_avg.append(lv)
                        loss_objts.append(loss1)

                    total_loss += np.average(loss_avg)

                    # logging
                    loss_accum_tagged[task_of_instance] += len(word_indices)
                    loss_accum_loss[task_of_instance] += np.average(loss_avg)

                    dynet.esum(loss_objts).backward()
                    self.trainer.update()


                if self.pta_params['M'] and batch_num % (len(train_data) // self.pta_params['M']) == 0:
                    if not dev:
                        continue
                    correct_list, total_list, _ = self.evaluate(dev_X, dev_Y, org_X, org_Y, dev_task_labels, verbose=False)
                    dev_accuracy = '\t'.join(["%.4f" % (0 if total == 0 else correct / total) for (correct, total) in
                                              zip(correct_list, total_list)])
                    # DecUpdate
                    best_model_idx = np.argmax(np.array(dev_accuracy.split('\t')[:-1]))

                    if self.pta_params['G']:
                        best_W = self.predictors['output_layers_dict']['task0'][best_model_idx].network_builder.W
                        best_b = self.predictors['output_layers_dict']['task0'][best_model_idx].network_builder.b
                        if self.predictors['output_layers_dict']['task0'][best_model_idx].network_builder.mlp:
                            best_W_mlp = self.predictors['output_layers_dict']['task0'][
                                best_model_idx].network_builder.W_mlp
                            best_b_mlp = self.predictors['output_layers_dict']['task0'][
                                best_model_idx].network_builder.b_mlp
                        for i in range(len(self.predictors['output_layers_dict']['task0'])):
                            if i == best_model_idx:
                                continue

                            W = self.predictors['output_layers_dict']['task0'][i].network_builder.W
                            b = self.predictors['output_layers_dict']['task0'][i].network_builder.b
                            W.set_value(best_W.value())
                            b.set_value(best_b.value())
                            self.pta_params['D'][i] = self.pta_params['D'][best_model_idx]

                            if self.predictors['output_layers_dict']['task0'][i].network_builder.mlp:
                                W_mlp = self.predictors['output_layers_dict']['task0'][i].network_builder.W_mlp
                                b_mlp = self.predictors['output_layers_dict']['task0'][i].network_builder.b_mlp
                                W_mlp.set_value(best_W_mlp.value())
                                b_mlp.set_value(best_b_mlp.value())

                        dynet.renew_cg()

                    if self.pta_params['P']:
                        # todo: evaluate only returns 1 task

                        for i in range(len(self.predictors['output_layers_dict']['task0'])):
                            if i == best_model_idx:
                                continue

                            W = self.predictors['output_layers_dict']['task0'][i].network_builder.W
                            noise = np.random.multivariate_normal([0] * W.shape()[0],
                                                                  self.pta_params['P'] * np.eye(W.shape()[0]), W.shape()[1])
                            W.set_value(noise.T + W.value())
                        dynet.renew_cg()

                    if self.pta_params['H']:
                        for i in range(len(self.predictors['output_layers_dict']['task0'])):
                            if i == best_model_idx:
                                continue
                            noise = np.random.normal(0, self.pta_params['H'])
                            noised_dropout = self.pta_params['D'][i] + noise
                            if noised_dropout >= self.pta_params['D-Upper'] or noised_dropout < self.pta_params['D-Lower']:
                                print("Omitting dropout of %f in head %d" % (noised_dropout, i), file=sys.stderr, flush=True)
                                continue
                            self.pta_params['D'][i] += noise

            print("iter {2} {0:>12}: {1:.2f}".format("total loss",
                                                     total_loss/total_tagged,
                                                     iter), file=sys.stderr, flush=True)

            # log losses
            for task_id in sorted(losses):
                losses[task_id].append(loss_accum_loss[task_id] / loss_accum_tagged[task_id])

            if log_losses:
                pickle.dump(losses, open(model_path + ".model" + ".losses.pickle", "wb"))

            if dev:
                # evaluate after every epoch
                correct_list, total_list, _ = self.evaluate(dev_X, dev_Y, org_X, org_Y, dev_task_labels)
                dev_accuracy = '\t'.join(["%.4f" % (0 if total == 0 else correct/total) for (correct, total) in zip(correct_list, total_list)])
                print("\ndev accuracy: %s" % dev_accuracy, file=sys.stderr, flush=True)

                self.train_log.append(("%d\t" % iter) + dev_accuracy)

                for i, (correct, total) in enumerate(zip(correct_list, total_list)):
                    val_accuracy = 0 if total == 0 else correct / total
                    if patience:
                        if val_accuracy > best_val_acc:
                            print('Accuracy %.4f is better than best val accuracy '
                                  '%.4f.' % (val_accuracy, best_val_acc),
                                  file=sys.stderr, flush=True)
                            best_val_acc = val_accuracy
                            epochs_no_improvement = 0
                            save(self, model_path)
                        else:
                            print('Accuracy %.4f is worse than best val loss %.4f.' %
                                  (val_accuracy, best_val_acc), file=sys.stderr, flush=True)
                            epochs_no_improvement += 1
                        if epochs_no_improvement == patience:
                            print('No improvement for %d epochs. Early stopping...' %
                                  epochs_no_improvement, file=sys.stderr, flush=True)
                            break


    def load_embeddings(self):
        print("loading embeddings", file=sys.stderr)
        embeddings, emb_dim = load_embeddings_file(self.embeds_file)
        assert(emb_dim==self.in_dim)
        num_words=len(set(embeddings.keys()).union(set(self.w2i.keys()))) # initialize all with embeddings
        # init model parameters and initialize them
        wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)

        init=0
        for word in embeddings:
            if word not in self.w2i:
                self.w2i[word]=len(self.w2i.keys()) # add new word
                wembeds.init_row(self.w2i[word], embeddings[word])
                init +=1
            elif word in embeddings:
                wembeds.init_row(self.w2i[word], embeddings[word])
                init += 1
        print("initialized: {}".format(init), file=sys.stderr)

        return wembeds

    def build_computation_graph(self, num_words, num_chars):
        """
        build graph and link to parameters
        """
        ## initialize word embeddings
        if self.embeds_file:
            wembeds = self.load_embeddings()
        else:
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)


        ## initialize character embeddings
        cembeds = None
        if self.c_in_dim > 0:
            cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim), init=self.initializer)


        layers = [] # inner layers
        output_layers_dict = {}   # from task_id to actual softmax predictor
        task_expected_at = {} # map task_id => output_layer_#

        # connect output layers to tasks
        for output_layer, task_id in zip(self.pred_layer, self.tasks_ids):
            if output_layer > self.h_layers:
                raise ValueError("cannot have a task at a layer (%d) which is "
                                 "beyond the model, increase h_layers (%d)"
                                 % (output_layer, self.h_layers))
            task_expected_at[task_id] = output_layer
        nb_tasks = len( self.tasks_ids )

        for layer_num in range(0,self.h_layers):
            if layer_num == 0:
                if self.c_in_dim > 0:
                    # in_dim: size of each layer
                    f_builder = self.builder(1, self.in_dim+self.c_in_dim*2, self.h_dim, self.model)
                    b_builder = self.builder(1, self.in_dim+self.c_in_dim*2, self.h_dim, self.model)
                else:
                    f_builder = self.builder(1, self.in_dim, self.h_dim, self.model)
                    b_builder = self.builder(1, self.in_dim, self.h_dim, self.model)

                layers.append(BiRNNSequencePredictor(f_builder, b_builder)) #returns forward and backward sequence
            else:
                # add inner layers (if h_layers >1)
                f_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                b_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(f_builder, b_builder))

        # store at which layer to predict task
        for task_id in self.tasks_ids:
            task_num_labels= len(self.task2tag2idx[task_id])
            output_layers_dict[task_id] = []
            for output in self.output_builder(self.model, self.h_dim, task_num_labels):
                output_layers_dict[task_id].append(output)

        char_rnn = BiRNNSequencePredictor(self.builder(1, self.c_in_dim, self.c_in_dim, self.model),
                                          self.builder(1, self.c_in_dim, self.c_in_dim, self.model))


        predictors = {}
        predictors["inner"] = layers
        predictors["output_layers_dict"] = output_layers_dict
        predictors["task_expected_at"] = task_expected_at

        return predictors, char_rnn, wembeds, cembeds

    def get_features(self, words):
        """
        from a list of words, return the word and word char indices
        """
        word_indices = []
        word_char_indices = []
        for word in words:
            if word in self.w2i:
                word_indices.append(self.w2i[word])
            else:
                word_indices.append(self.w2i[UNK])

            if self.c_in_dim > 0:
                chars_of_word = [self.c2i["<w>"]]
                for char in word:
                    if char in self.c2i:
                        chars_of_word.append(self.c2i[char])
                    else:
                        chars_of_word.append(self.c2i[UNK])
                chars_of_word.append(self.c2i["</w>"])
                word_char_indices.append(chars_of_word)
        return word_indices, word_char_indices

    def get_data_as_indices(self, folder_name, task, raw=False):
        """
        X = list of (word_indices, word_char_indices)
        Y = list of tag indices
        """
        X, Y = [],[]
        org_X, org_Y = [], []
        task_labels = []
        for (words, tags) in read_conll_file(folder_name, raw=raw):
            word_indices, word_char_indices = self.get_features(words)
            tag_indices = [self.task2tag2idx[task].get(tag) for tag in tags]
            X.append((word_indices,word_char_indices))
            Y.append(tag_indices)
            org_X.append(words)
            org_Y.append(tags)
            task_labels.append( task )
        return X, Y, org_X, org_Y, task_labels

    def predict(self, word_indices, char_indices, task_id, train=False):
        """
        predict tags for a sentence represented as char+word embeddings
        """

        # word embeddings
        wfeatures = [self.wembeds[w] for w in word_indices]

        # char embeddings
        if self.c_in_dim > 0:
            char_emb = []
            rev_char_emb = []
            # get representation for words
            for chars_of_token in char_indices:
                char_feats = [self.cembeds[c] for c in chars_of_token]
                # use last state as word representation
                f_char, b_char = self.char_rnn.predict_sequence(char_feats, char_feats)
                last_state = f_char[-1]
                rev_last_state = b_char[-1]
                char_emb.append(last_state)
                rev_char_emb.append(rev_last_state)

            features = [dynet.concatenate([w,c,rev_c]) for w,c,rev_c in zip(wfeatures,char_emb,rev_char_emb)]
        else:
            features = wfeatures

        if train: # only do at training time
            features = [dynet.noise(fe,self.noise_sigma) for fe in features]

        output_expected_at_layer = self.predictors["task_expected_at"][task_id]
        output_expected_at_layer -=1

        # go through layers
        # input is now combination of w + char emb
        prev = features
        prev_rev = features
        num_layers = self.h_layers

        for i in range(num_layers):
            predictor = self.predictors["inner"][i]
            forward_sequence, backward_sequence = predictor.predict_sequence(prev, prev_rev)
            if i > 0 and self.activation:
                # activation between LSTM layers
                forward_sequence = [self.activation(s) for s in forward_sequence]
                backward_sequence = [self.activation(s) for s in backward_sequence]

            if i == output_expected_at_layer:
                output = []
                output_predictors = self.predictors["output_layers_dict"][task_id]
                for j, output_predictor in enumerate(output_predictors):
                    concat_layer = [dynet.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]

                    if train and self.noise_sigma > 0.0:
                        concat_layer = [dynet.noise(fe,self.noise_sigma) for fe in concat_layer]
                    if train:
                        output.append(output_predictor.predict_sequence(concat_layer, dropout=self.pta_params['D'][j]))
                    else:
                        output.append(output_predictor.predict_sequence(concat_layer))
                return [o for i,o in enumerate(output) if self.predict_on_layer is None or i == self.predict_on_layer]

            prev = forward_sequence
            prev_rev = backward_sequence

        raise Exception("oops should not be here")
        return None

    def evaluate(self, test_X, test_Y, org_X, org_Y, task_labels, output_predictions=None, output_probs=False, verbose=True, raw=False, get_predictions_array=False):
        """
        compute accuracy on a test file
        """
        correct = (self.out_num+1) * [0]
        total = (self.out_num+1) * [0.0]
        prediction_array = [[] for _ in range(self.out_num+1)]

        if output_predictions != None:
            i2w = {self.w2i[w] : w for w in self.w2i.keys()}
            task_id = task_labels[0] # get first
            i2t = {self.task2tag2idx[task_id][t] : t for t in self.task2tag2idx[task_id].keys()}

        for i, ((word_indices, word_char_indices), gold_tag_indices, task_of_instance) in enumerate(zip(test_X, test_Y, task_labels)):
            if verbose:
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

            output_list = self.predict(word_indices, word_char_indices, task_of_instance)
            #meta_output = np.mean([[token.value() for token in output] for output in output_list], axis=0)
            expressions_to_average = [[] for _ in range(len(output_list[0]))]
            for ol in output_list:
                for j, o in enumerate(ol):
                    expressions_to_average[j].append(o)
            output_list.append([dynet.average(e) for e in expressions_to_average])
            labeled_output_list = enumerate(output_list) if self.predict_on_layer is None else \
                [(self.predict_on_layer, output_list[0]), (self.out_num, output_list[1])]
            for out_index, output in labeled_output_list:
                predicted_tag_indices = [np.argmax(o.value()) for o in output]  # logprobs to indices
                if output_predictions:
                    prediction = [i2t[idx] for idx in predicted_tag_indices]
                    tag_confidences = [np.max(o.value()) for o in output]

                    words = org_X[i]
                    gold = org_Y[i]

                    for w, g, p, c in zip(words, gold, prediction, tag_confidences):
                        if raw:
                            print(u"{}\t{}".format(w, p)) # do not print DUMMY tag when --raw is on
                        else:
                            if output_probs:
                                print(u"%s\t%s\t%s\t%.2f" % (w, g, p, c))
                            else:
                                print(u"%s\t%s\t%s" % (w, g, p))
                    print("")
                correct[out_index] += sum([1 for (predicted, gold) in zip(predicted_tag_indices, gold_tag_indices) if predicted == gold])
                total[out_index] += len(gold_tag_indices)
                prediction_array[out_index].append([o.value() for o in output])

        return correct, total, prediction_array if get_predictions_array else []

    def get_train_data(self, list_folders_name):
        """
        Get train data: read each train set (linked to a task)

        :param list_folders_name: list of folders names

        transform training data to features (word indices)
        map tags to integers
        """
        X = []
        Y = []
        task_labels = [] # keeps track of where instances come from "task1" or "task2"..
        self.tasks_ids = [] # record ids of the tasks

        # word 2 indices and tag 2 indices
        w2i = {} # word to index
        c2i = {} # char to index
        task2tag2idx = {} # id of the task -> tag2idx

        w2i[UNK] = 0  # unk word / OOV
        c2i[UNK] = 0  # unk char
        c2i["<w>"] = 1   # word start
        c2i["</w>"] = 2  # word end index

        if self.max_vocab_size is not None:
            word_counter = Counter()
            print('Reading files to create vocabulary of size %d.' %
                  self.max_vocab_size, file=sys.stderr)
            for i, folder_name in enumerate(list_folders_name):
                for words, _ in read_conll_file(folder_name):
                    word_counter.update(words)
            word_count_pairs = word_counter.most_common(self.max_vocab_size-1)
            for word, _ in word_count_pairs:
                w2i[word] = len(w2i)

        for i, folder_name in enumerate(list_folders_name):
            num_sentences=0
            num_tokens=0
            task_id = 'task'+str(i)
            self.tasks_ids.append( task_id )
            if task_id not in task2tag2idx:
                task2tag2idx[task_id] = {}
            for instance_idx, (words, tags) in enumerate(read_conll_file(folder_name)):
                num_sentences += 1
                instance_word_indices = [] #sequence of word indices
                instance_char_indices = [] #sequence of char indices
                instance_tags_indices = [] #sequence of tag indices

                for i, (word, tag) in enumerate(zip(words, tags)):
                    num_tokens += 1

                    # map words and tags to indices
                    if word not in w2i and self.max_vocab_size is not None:
                        # if word is not in the created vocab, add an UNK token
                        instance_word_indices.append(w2i[UNK])
                    else:
                        if word not in w2i:
                            w2i[word] = len(w2i)
                        instance_word_indices.append(w2i[word])

                    if self.c_in_dim > 0:
                        chars_of_word = [c2i["<w>"]]
                        for char in word:
                            if char not in c2i:
                                c2i[char] = len(c2i)
                            chars_of_word.append(c2i[char])
                        chars_of_word.append(c2i["</w>"])
                        instance_char_indices.append(chars_of_word)

                    if tag not in task2tag2idx[task_id]:
                        task2tag2idx[task_id][tag]=len(task2tag2idx[task_id])

                    instance_tags_indices.append(task2tag2idx[task_id].get(tag))

                X.append((instance_word_indices, instance_char_indices)) # list of word indices, for every word list of char indices
                Y.append(instance_tags_indices)
                task_labels.append(task_id)

            if num_sentences == 0 or num_tokens == 0:
                sys.exit( "No data read from: "+folder_name )

            print("TASK "+task_id+" "+folder_name, file=sys.stderr)
            print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
            print("%s w features, %s c features " % (len(w2i),len(c2i)), file=sys.stderr)

        assert(len(X)==len(Y))
        return X, Y, task_labels, w2i, c2i, task2tag2idx  #sequence of features, sequence of labels, necessary mappings

    def save_embeds(self, out_filename):
        """
        save final embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping
        i2w = {self.w2i[w]: w for w in self.w2i.keys()}
    
        OUT = open(out_filename+".w.emb","w")
        for word_id in i2w.keys():
            wembeds_expression = self.wembeds[word_id]
            word = i2w[word_id]
            OUT.write("{} {}\n".format(word," ".join([str(x) for x in wembeds_expression.npvalue()])))
        OUT.close()


if __name__=="__main__":
    main()
