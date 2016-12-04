"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  hard_attention.py [--dynet-mem MEM][--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION]
  [--learning=LEARNING] [--plot] [--eval] [--init-epochs=INIT_EPOCHS] [--ensemble=ENSEMBLE] TRAIN_PATH DEV_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH SYMS_PATH

Arguments:
  TRAIN_PATH    destination path
  DEV_PATH      development set path
  TEST_PATH     test path
  RESULTS_PATH  results file to be written
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs
  SYMS_PATH     symbol file path

Options:
  -h --help                     show this help message and exit
  --dynet-mem MEM                 allocates MEM bytes for (py)cnn
  --input=INPUT                 input vector dimensions
  --hidden=HIDDEN               hidden layer dimensions
  --feat-input=FEAT             feature input vector dimension
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
  --reg=REGULARIZATION          regularization parameter for optimization
  --learning=LEARNING           learning rate parameter for optimization
  --plot                        draw a learning curve plot while training each model
  --eval                        run evaluation without training
  --init-epochs=INIT_EPOCHS     number of initialization epochs (i.e. epochs in which the original objective is being minimized)
  --ensemble=ENSEMBLE           ensemble model paths, separated by comma
"""

import traceback
import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import common
from matplotlib import pyplot as plt
from docopt import docopt
import dynet as pc
from collections import defaultdict
import sys
import fst


# default values
INPUT_DIM = 200
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 200
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1
PARALLELIZE = True

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'
STEP = '^'
ALIGN_SYMBOL = '~'


def load_preprocessed(path):
    from io import open as uopen
    with uopen(path+'.word', encoding='utf-8') as w_fh, \
            uopen(path+'.lemma', encoding='utf-8') as l_fh, \
            uopen(path+'.align', encoding='utf-8') as a_fh, \
            uopen(path+'.answer', encoding='utf-8') as ans_fh, \
            uopen(path+'.goods', encoding='utf-8') as g_fh:
        lemmas = [x.strip() for x in l_fh]
        answers = [x.strip() for x in ans_fh]
        words_lines = [x.strip() for x in w_fh]
        alignment_lines = [x.strip() for x in a_fh]
        goods = [int(x.strip()) for x in g_fh]
        assert len(lemmas) == len(alignment_lines) and len(lemmas) == len(words_lines) and len(lemmas) == len(goods)
        words = []
        alignments = []
        feats = []
        for l, w_line, a_line in zip(lemmas, words_lines, alignment_lines):
            words.append(w_line.split(u' '))
            from itertools import izip
            a_iter = iter(a_line.split(u' '))
            alignment = []
            for in_s, out_s in izip(a_iter, a_iter):
                alignment.append((in_s, out_s))
            alignments.append(alignment)
            feats.append({'pos': 'V'}) # dummy
        return words, lemmas, alignments, feats, goods, answers


def main(train_path, dev_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, feat_input_dim,
         epochs, layers, optimization, regularization, learning_rate, plot, eval_only, ensemble, init_epochs,
         syms_file):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': regularization,
                    'LEARNING_RATE': learning_rate, 'INIT_EPOCHS': init_epochs}

    print 'train path = ' + str(train_path)
    print 'dev path =' + str(dev_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load train and test data
    (_, train_lemmas, _, train_feat_dicts, _, train_answers) = \
        load_preprocessed(train_path)
    (_, dev_lemmas, _, dev_feat_dicts, _, dev_answers) = \
        load_preprocessed(dev_path)
    (_, test_lemmas, _, test_feat_dicts, _, test_answers) = \
        load_preprocessed(test_path)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_answers, train_lemmas, train_feat_dicts)

    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)

    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(3 * MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # indicates the FST to step forward in the input
    alphabet.append(STEP)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    if not eval_only:

        last_epochs = []
        trained_model, last_epoch = train_model_wrapper(input_dim, hidden_dim, layers, train_lemmas, train_feat_dicts,
                                                        dev_lemmas, dev_feat_dicts,
                                                        alphabet, alphabet_index, inverse_alphabet_index, epochs,
                                                        optimization, results_file_path,
                                                        feat_index, feature_types, feat_input_dim, feature_alphabet,
                                                        plot, train_answers, dev_answers,
                                                        init_epochs,
                                                        syms_file=syms_file
                                                        )

        # print when did each model stop
        print 'stopped on epoch {}'.format(last_epoch)

        with open(results_file_path + '.epochs', 'w') as f:
            f.writelines(last_epochs)

        print 'finished training all models'
    else:
        print 'skipped training by request. evaluating best models:'

    if False:
        # eval on dev
        print '=========DEV EVALUATION:========='
        evaluate_ndst(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet, feature_types,
                      hidden_dim, hyper_params, input_dim, inverse_alphabet_index, layers, results_file_path,
                      sigmorphon_root_dir, dev_feat_dicts, dev_lemmas, dev_path,
                      dev_answers, train_path)

        # eval on test
        print '=========TEST EVALUATION:========='
        evaluate_ndst(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet, feature_types,
                      hidden_dim, hyper_params, input_dim, inverse_alphabet_index, layers, results_file_path,
                      sigmorphon_root_dir, test_feat_dicts, test_lemmas, test_path,
                      dev_answers, train_path)

    return


def train_model_wrapper(input_dim, hidden_dim, layers, train_lemmas, train_feat_dicts,
                        dev_lemmas, dev_feat_dicts,
                        alphabet, alphabet_index, inverse_alphabet_index, epochs,
                        optimization, results_file_path, feat_index,
                        feature_types, feat_input_dim, feature_alphabet, plot, train_answers,
                        dev_answers, init_epochs, syms_file):
    # build model
    initial_model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = \
        build_model(alphabet, input_dim, hidden_dim, layers, feature_types, feat_input_dim,
                    feature_alphabet)

    # train model
    trained_model, last_epoch = train_model(initial_model, char_lookup, feat_lookup, R, bias, encoder_frnn,
                                            encoder_rrnn, decoder_rnn,
                                            train_lemmas,
                                            train_feat_dicts, dev_lemmas,
                                            dev_feat_dicts, alphabet_index,
                                            inverse_alphabet_index,
                                            epochs, optimization, results_file_path,
                                            feat_index, feature_types,
                                            plot, train_answers, dev_answers, init_epochs, syms_file)

    # evaluate last model on dev
    predicted_sequences = sample_decode_sequences(trained_model, char_lookup, feat_lookup, R, bias, encoder_frnn,
                                                  encoder_rrnn, decoder_rnn, alphabet_index, inverse_alphabet_index,
                                                  dev_lemmas, dev_feat_dicts, feat_index, feature_types)
    if len(predicted_sequences) > 0:
        evaluate_model(predicted_sequences, dev_lemmas, dev_feat_dicts, dev_answers, feature_types, print_results=False)
    else:
        print 'no examples in dev set to evaluate'

    return trained_model, last_epoch


def build_model(alphabet, input_dim, hidden_dim, layers, feature_types, feat_input_dim, feature_alphabet):
    print 'creating model...'

    model = pc.Model()

    # character embeddings
    char_lookup = model.add_lookup_parameters((len(alphabet), input_dim))

    # feature embeddings
    feat_lookup = model.add_lookup_parameters((len(feature_alphabet), feat_input_dim))

    # used in softmax output
    R = model.add_parameters((len(alphabet), hidden_dim))
    bias = model.add_parameters(len(alphabet))

    # rnn's
    # encoder_frnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_frnn = pc.GRUBuilder(layers, input_dim, hidden_dim, model)
    # encoder_rrnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = pc.GRUBuilder(layers, input_dim, hidden_dim, model)

    # 2 * HIDDEN_DIM + input_dim, as it gets BLSTM[i], previous output
    concatenated_input_dim = 2 * hidden_dim + input_dim + len(feature_types) * feat_input_dim
    # decoder_rnn = pc.LSTMBuilder(layers, concatenated_input_dim, hidden_dim, model)
    decoder_rnn = pc.GRUBuilder(layers, concatenated_input_dim, hidden_dim, model)
    print 'decoder lstm dimensions are {} x {}'.format(concatenated_input_dim, hidden_dim)
    print 'finished creating model'

    return model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn


def load_best_model(alphabet, results_file_path, input_dim, hidden_dim, layers, feature_alphabet,
                    feat_input_dim, feature_types):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim,
                                                                 layers, feature_types,
                                                                 feat_input_dim,
                                                                 feature_alphabet)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn


def log_to_file(file_name, e, avg_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(file_name, 'epoch', 'avg_loss', 'train_accuracy', 'dev_accuracy')

    with open(file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_loss, train_accuracy, dev_accuracy))


def read_fst(lemma, inv_sigma, dir, syms):
    fname = dir + '/' + '.'.join([str(inv_sigma[x]) for x in lemma]) + '.fst'
    machine = fst.read(fname)
    machine.isyms = syms
    machine.osyms = syms
    return fst.LogVectorFst(machine).push_weights()


def get_clamped_machine(lemma_fst, answer, syms):
    answer_fst = fst.linear_chain(answer, syms=syms, semiring='log')
    return lemma_fst.compose(answer_fst)


def sample(machine, sigma, num=64, inv_tau=1.):
    # put into log-semiring with local normalization
    # machine = fst.LogVectorFst(machine).push_weights()
    import numpy as np
    from scipy.misc import logsumexp
    paths = []
    for i in xrange(num):
        state = machine[machine.start]
        iseq = []
        oseq = []
        path_prob = 0.0
        while True:
            logprobs = np.zeros(len(state)+1)
            logprobs[0] = -inv_tau*float(state.final)
            arcs = []
            for arcid, arc in enumerate(state):
                arcs.append(arc)
                logprobs[arcid+1] = -inv_tau*float(arc.weight)

            # should already be normalized (due to weight pushing),
            # but it can be off by a fudge factor
            normalized_logprobs = logprobs - logsumexp(logprobs)
            probs = np.exp(normalized_logprobs)
            sample = np.random.choice(len(state)+1, 1, p=probs)[0]
            path_prob += normalized_logprobs[sample]
            if sample == 0:
                break
            else:
                arc = arcs[sample-1]
                iseq.append(arc.ilabel)
                oseq.append(arc.olabel)
                state = machine[arc.nextstate]
        iseq = [sigma[x] if x != 0 else '~' for x in iseq]
        oseq = [sigma[x] if x != 0 else '~' for x in oseq]
        word = [x for x in oseq if x != '~']
        paths.append({'weight': path_prob, 'alignment': (iseq, oseq), 'word': word})
    return paths


def read_syms(syms_file):
    sigma, inv_sigma, syms = {}, {}, fst.SymbolTable()
    with open(syms_file) as fh:
        for l in fh:
            i, s = l.strip().split()
            i = int(i)
            sigma[i] = s
            inv_sigma[s] = i
            syms[s] = i
    eps = fst.EPSILON
    sigma[0] = eps
    inv_sigma[eps] = 0
    syms[eps] = 0
    return sigma, inv_sigma, syms


def train_model(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, train_lemmas,
                train_feat_dicts, dev_lemmas, dev_feat_dicts, alphabet_index,
                inverse_alphabet_index, epochs, optimization, results_file_path,
                feat_index, feature_types, plot, train_answers, dev_answers, init_epochs, syms_file):
    print 'training...'
    sigma, inv_sigma, syms = read_syms(syms_file)
    fst_dir = '/export/a10/kitsing/ryanouts/ryanout-2PKE-z-0/train/'

    np.random.seed(17)
    random.seed(17)

    if optimization == 'ADAM':
        trainer = pc.AdamTrainer(model, lam=REGULARIZATION, alpha=LEARNING_RATE, beta_1=0.9, beta_2=0.999, eps=1e-8)
    elif optimization == 'MOMENTUM':
        trainer = pc.MomentumSGDTrainer(model)
    elif optimization == 'SGD':
        trainer = pc.SimpleSGDTrainer(model)
    elif optimization == 'ADAGRAD':
        trainer = pc.AdagradTrainer(model)
    elif optimization == 'ADADELTA':
        trainer = pc.AdadeltaTrainer(model)
    else:
        trainer = pc.SimpleSGDTrainer(model)

    total_loss = 0
    best_avg_dev_loss = 999
    best_dev_accuracy = -1
    best_train_accuracy = -1
    patience = 0
    train_len = len(train_lemmas)
    sanity_set_size = 10
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_accuracy_y = []
    dev_accuracy_y = []
    e = -1

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
    avg_loss = -1

    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_lemmas, train_feat_dicts, train_answers)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            lemma, feats, word = example
            free_fst = read_fst(lemma, inv_sigma, fst_dir, syms)
            clamped_fst = get_clamped_machine(free_fst, word, syms)
            if e < init_epochs:
                # FIXME
                assert False
            clamped_log_likelihoods = []
            clamped_weights = []
            free_log_likelihoods = []
            free_weights = []
            pc.renew_cg()
            expr_R = pc.parameter(R)
            expr_bias = pc.parameter(bias)
            num_clamped_samples = 3
            num_free_samples = 3
            clamped_samples = sample(clamped_fst, sigma, num_clamped_samples, inv_tau=3e-3)
            free_samples = sample(free_fst, sigma, num_free_samples, inv_tau=3e-3)
            padded_lemma = BEGIN_WORD + lemma + END_WORD
            lemma_char_vecs = encode_lemma(alphabet_index, char_lookup, padded_lemma)
            blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs)
            for clamped_sample in clamped_samples:
                alignment = clamped_sample['alignment']
                loss = - one_word_loss(model, char_lookup, feat_lookup,
                                       expr_R, expr_bias, decoder_rnn, padded_lemma,
                                       feats, word, alphabet_index,
                                       alignment, feat_index,
                                       feature_types, blstm_outputs=blstm_outputs)
                clamped_log_likelihoods.append(loss)
                # print loss.value()
                clamped_weights.append(loss - clamped_sample['weight'])
                # print (loss - clamped_sample['weight']).value()

            for free_sample in free_samples:
                alignment = free_sample['alignment']
                loss = - one_word_loss(model, char_lookup, feat_lookup,
                                       expr_R, expr_bias, decoder_rnn, padded_lemma,
                                       feats, word, alphabet_index,
                                       alignment, feat_index,
                                       feature_types, blstm_outputs=blstm_outputs)
                free_log_likelihoods.append(loss)
                free_weights.append(loss - free_sample['weight'])
            if e < init_epochs:
                # FIXME
                assert False
                nll = - pc.logsumexp(clamped_log_likelihoods)
                coeff = float(e) / init_epochs
                hope = clamped_log_likelihoods[:goods]
                loss = - (pc.logsumexp(hope) - nll * coeff)
            else:
                clamped_weighted_ll = []
                clamped_weight_sum = pc.logsumexp(clamped_weights)
                # print clamped_weight_sum.value()
                for ll, w in zip(clamped_log_likelihoods, clamped_weights):
                    clamped_weighted_ll.append(ll * pc.exp(w - clamped_weight_sum))
                # print 'clamped weights: {}'.format([x.value() for x in clamped_weights])
                # print 'sum: {}'.format(clamped_weight_sum.value())
                # c_max = pc.emax(clamped_weights)
                # lsumexp = pc.esum([pc.exp(x-c_max) for x in clamped_weights])
                # print 'sum 2: {}'.format(lsumexp.value())
                # print 'normalized {}'.format((w - clamped_weight_sum).value())
                free_weighted_ll = []
                free_weight_sum = pc.logsumexp(free_weights)
                for ll, w in zip(free_log_likelihoods, free_weights):
                    free_weighted_ll.append(ll * pc.exp(w - free_weight_sum))
                loss = - (pc.esum(clamped_weighted_ll) - pc.esum(free_weighted_ll)) / num_free_samples
            loss_value = loss.value()
            print 'loss: {}'.format(loss_value)
            from numpy import isnan, isinf
            if isnan(loss_value) or isinf(loss_value):
                print 'losses: {}'.format([x.value() for x in clamped_weighted_ll])
                print 'losses: {}'.format([x.value() for x in free_weighted_ll])
                assert False
            # losses_concat = pc.concatenate(losses)
            # losses_concat = pc.exp(losses_concat - maximum)
            # z = pc.log(pc.sum_cols(losses_concat))
            # hope = losses_concat[0:goods]
            # loss = - (pc.log(pc.sum_cols(hope)) - z)
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

        # FIXME
        if EARLY_STOPPING and e >= init_epochs:

            # get train accuracy
            print 'evaluating on train...'
            train_predictions = sample_decode_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn,
                                                        encoder_rrnn, decoder_rnn, alphabet_index,
                                                        inverse_alphabet_index, train_lemmas[:sanity_set_size],
                                                        train_feat_dicts[:sanity_set_size], feat_index, feature_types,
                                                        sigma, inv_sigma, fst_dir, syms)

            train_accuracy = evaluate_model(train_predictions, train_lemmas[:sanity_set_size],
                                            train_feat_dicts[:sanity_set_size],
                                            train_answers[:sanity_set_size],
                                            feature_types, print_results=False)[1]

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            dev_accuracy = 0
            avg_dev_loss = 0

            if len(dev_lemmas) > 0:

                # get dev accuracy
                dev_predictions = sample_decode_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn,
                                                          encoder_rrnn, decoder_rnn, alphabet_index,
                                                          inverse_alphabet_index, dev_lemmas, dev_feat_dicts,
                                                          feat_index, feature_types, sigma, inv_sigma, fst_dir, syms)
                print 'evaluating on dev...'
                # get dev accuracy
                dev_accuracy = evaluate_model(dev_predictions, dev_lemmas, dev_feat_dicts, dev_answers,
                                              feature_types,
                                              print_results=True)[1]

                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                # found "perfect" model
                if dev_accuracy == 1:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

                # get dev loss
                total_dev_loss = 0
                '''
                for i in xrange(len(dev_lemmas)):
                    total_dev_loss += one_word_loss(model, char_lookup, feat_lookup, R, bias, encoder_frnn,
                                                    encoder_rrnn, decoder_rnn, dev_lemmas[i],
                                                    dev_feat_dicts[i], dev_words[i], alphabet_index,
                                                    dev_aligned_pairs[i], feat_index, feature_types).value()
                avg_dev_loss = total_dev_loss / float(len(dev_lemmas))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss
                '''
                print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} train accuracy = {4:.4f} \
 best dev accuracy {5:.4f} best train accuracy: {6:.4f} patience = {7}'.format(e, avg_loss, avg_dev_loss, dev_accuracy,
                                                                               train_accuracy, best_dev_accuracy,
                                                                               best_train_accuracy, patience)

                log_to_file(results_file_path + '_log.txt', e, avg_loss, train_accuracy, dev_accuracy)

                if patience == MAX_PATIENCE:
                    print 'out of patience after {0} epochs'.format(str(e))
                    # TODO: would like to return best model but pycnn has a bug with save and load. Maybe copy via code?
                    # return best_model[0]
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e
            else:

                # if no dev set is present, optimize on train set
                print 'no dev set for early stopping, running all epochs until perfectly fitting or patience was \
                reached on the train set'

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                print 'epoch: {0} train loss: {1:.4f} train accuracy = {2:.4f} best train accuracy: {3:.4f} \
                patience = {4}'.format(e, avg_loss, train_accuracy, best_train_accuracy, patience)

                # found "perfect" model on train set or patience has reached
                if train_accuracy == 1 or patience == MAX_PATIENCE:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

            # update lists for plotting
            train_accuracy_y.append(train_accuracy)
            epochs_x.append(e)
            train_loss_y.append(avg_loss)
            dev_loss_y.append(avg_dev_loss)
            dev_accuracy_y.append(dev_accuracy)

        # finished epoch
        train_progress_bar.update(e)
        if plot:
            with plt.style.context('fivethirtyeight'):
                p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
                p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
                p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
                p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
                plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
            plt.savefig(results_file_path + '.png')
    train_progress_bar.finish()
    if plot:
        plt.cla()
    print 'finished training. average loss: ' + str(avg_loss)
    return model, e


def save_pycnn_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


# noinspection PyPep8Naming,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def one_word_loss(model, char_lookup, feat_lookup, R, bias, decoder_rnn, padded_lemma, feats, word,
                  alphabet_index, aligned_pair,
                  feat_index, feature_types, encoder_frnn=None, encoder_rrnn=None, blstm_outputs=None):
    from numpy import isnan, isinf
    # pc.renew_cg()

    # read the parameters
    # char_lookup = model["char_lookup"]
    # feat_lookup = model["feat_lookup"]
    # R = pc.parameter(model["R"])
    # bias = pc.parameter(model["bias"])
    # R = pc.parameter(R)
    # bias = pc.parameter(bias)

    # padded_lemma = BEGIN_WORD + lemma + END_WORD

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = encode_feats(feat_index, feat_lookup, feats, feature_types)

    feats_input = pc.concatenate(feat_vecs)

    # convert characters to matching embeddings
    if blstm_outputs is None:
        lemma_char_vecs = encode_lemma(alphabet_index, char_lookup, padded_lemma)
        blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    loss = []

    # i is input index, j is output index
    i = 0
    j = 0

    # go through alignments, progress j when new output is introduced, progress i when new char is seen on lemma (no ~)
    # TODO: try sutskever flip trick?
    # TODO: attention on the lemma chars/feats could help here?
    aligned_lemma, aligned_word = aligned_pair
    aligned_lemma += END_WORD
    aligned_word += END_WORD
    eps = 1e-10
    bias = bias

    # run through the alignments
    for index, (input_char, output_char) in enumerate(zip(aligned_lemma, aligned_word)):
        possible_outputs = []

        # feedback, feedback char, blstm[i], feats
        decoder_input = pc.concatenate([prev_output_vec,
                                        blstm_outputs[i],
                                        feats_input])

        d_check = decoder_input.npvalue()
        assert not any(isnan(d_check)), (d_check, feats_input.npvalue(), prev_output_vec.npvalue(), index, input_char, output_char)
        assert not any(isinf(d_check)), (d_check, feats_input.npvalue(), prev_output_vec.npvalue())
        # if reached the end word symbol
        if output_char == END_WORD:
            s = s.add_input(decoder_input)
            decoder_rnn_output = s.output()
            probs = pc.log_softmax(R * decoder_rnn_output + bias)

            p_check = probs.npvalue()
            assert not any(isnan(p_check)), (p_check, decoder_rnn_output.npvalue())
            assert not any(isinf(p_check)), (p_check, decoder_rnn_output.npvalue())

            # compute local loss
            loss.append(-pc.pick(probs, alphabet_index[END_WORD]))
            continue

        # first step: if there is no prefix in the output (shouldn't delay on current input), step forward
        if padded_lemma[i] == BEGIN_WORD and aligned_lemma[index] != ALIGN_SYMBOL:
            # perform rnn step
            # feedback, i, j, blstm[i], feats
            s = s.add_input(decoder_input)
            decoder_rnn_output = s.output()
            probs = pc.log_softmax(R * decoder_rnn_output + bias)

            p_check = probs.npvalue()
            assert not any(isnan(p_check)), (p_check, decoder_rnn_output.npvalue())
            assert not any(isinf(p_check)), (p_check, decoder_rnn_output.npvalue())

            # compute local loss
            loss.append(-pc.pick(probs, alphabet_index[STEP]))

            # prepare for the next iteration - "feedback"
            prev_output_vec = char_lookup[alphabet_index[STEP]]
            prev_char_vec = char_lookup[alphabet_index[EPSILON]]
            i += 1

        # if there is new output
        if aligned_word[index] != ALIGN_SYMBOL:
            decoder_input = pc.concatenate([prev_output_vec,
                                            blstm_outputs[i],
                                            feats_input])

            # perform rnn step
            s = s.add_input(decoder_input)
            decoder_rnn_output = s.output()
            probs = pc.log_softmax(R * decoder_rnn_output + bias)

            p_check = probs.npvalue()
            assert not any(isnan(p_check)), (p_check, decoder_rnn_output.npvalue())
            assert not any(isinf(p_check)), (p_check, decoder_rnn_output.npvalue())

            if aligned_word[index] in alphabet_index:
                current_loss = -pc.pick(probs, alphabet_index[aligned_word[index]])

                # prepare for the next iteration - "feedback"
                prev_output_vec = char_lookup[alphabet_index[aligned_word[index]]]
            else:
                current_loss = -pc.pick(probs, alphabet_index[UNK])

                # prepare for the next iteration - "feedback"
                prev_output_vec = char_lookup[alphabet_index[UNK]]
            loss.append(current_loss)

            j += 1

        # now check if it's time to progress on input - input's not done, should not delay on the character
        if i < len(padded_lemma) - 1 and aligned_lemma[index + 1] != ALIGN_SYMBOL:
            # perform rnn step
            # feedback, i, j, blstm[i], feats
            decoder_input = pc.concatenate([prev_output_vec,
                                            blstm_outputs[i],
                                            feats_input])

            s = s.add_input(decoder_input)
            decoder_rnn_output = s.output()
            probs = pc.log_softmax(R * decoder_rnn_output + bias)

            p_check = probs.npvalue()
            assert not any(isnan(p_check)), (p_check, decoder_rnn_output.npvalue())
            assert not any(isinf(p_check)), (p_check, decoder_rnn_output.npvalue())

            # compute local loss
            loss.append(-pc.pick(probs, alphabet_index[STEP]))

            # prepare for the next iteration - "feedback"
            prev_output_vec = char_lookup[alphabet_index[STEP]]
            i += 1

    # loss = esum(loss)
    check_l = [l.value() for l in loss]
    if any(isnan(check_l)) or any(isinf(check_l)):
        print 'one_word_loss assertion failed!'
        print check_l
        print padded_lemma
        print word
        print aligned_pair
        assert False
    loss = pc.esum(loss)
    # loss = pc.esum(loss) / len(loss)
    # loss = pc.average(loss)

    return loss


def encode_feats(feat_index, feat_lookup, feats, feature_types):
    feat_vecs = []
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feats:
            feat_str = feat + ':' + feats[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
        else:
            feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
    return feat_vecs


# noinspection PyPep8Naming
def predict_output_sequence(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, alphabet_index,
                            inverse_alphabet_index, feat_index, feature_types):
    pc.renew_cg()

    # read the parameters
    # char_lookup = model["char_lookup"]
    # feat_lookup = model["feat_lookup"]
    # R = pc.parameter(model["R"])
    # bias = pc.parameter(model["bias"])
    R = pc.parameter(R)
    bias = pc.parameter(bias)

    # convert characters to matching embeddings, if UNK handle properly
    padded_lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = encode_lemma(alphabet_index, char_lookup, padded_lemma)

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = encode_feats(feat_index, feat_lookup, feats, feature_types)

    feats_input = pc.concatenate(feat_vecs)

    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]

    # i is input index, j is output index
    i = 0
    num_outputs = 0
    predicted_output_sequence = []

    # run the decoder through the sequence and predict characters, twice max prediction as step outputs are added
    while num_outputs < MAX_PREDICTION_LEN * 3:

        # prepare input vector and perform LSTM step
        decoder_input = pc.concatenate([prev_output_vec,
                                        blstm_outputs[i],
                                        feats_input])

        s = s.add_input(decoder_input)

        # compute softmax probs vector and predict with argmax
        decoder_rnn_output = s.output()
        probs = pc.softmax(R * decoder_rnn_output + bias)
        probs = probs.vec_value()
        predicted_output_index = common.argmax(probs)
        predicted_output = inverse_alphabet_index[predicted_output_index]
        predicted_output_sequence.append(predicted_output)

        # check if step or char output to promote i.
        if predicted_output == STEP:
            if i < len(padded_lemma) - 1:
                i += 1

        num_outputs += 1

        # check if reached end of word
        if predicted_output_sequence[-1] == END_WORD:
            break

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[predicted_output_index]

    # remove the end word symbol

    return u''.join(predicted_output_sequence[0:-1])


def bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs):
    # BiLSTM forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    frnn_outputs = []
    for c in lemma_char_vecs:
        s = s.add_input(c)
        frnn_outputs.append(s.output())

    # BiLSTM backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    rrnn_outputs = []
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
        rrnn_outputs.append(s.output())

    # BiLTSM outputs
    blstm_outputs = []
    lemma_char_vecs_len = len(lemma_char_vecs)
    for i in xrange(lemma_char_vecs_len):
        blstm_outputs.append(pc.concatenate([frnn_outputs[i], rrnn_outputs[lemma_char_vecs_len - i - 1]]))

    return blstm_outputs


def encode_lemma(alphabet_index, char_lookup, padded_lemma):
    lemma_char_vecs = []
    for char in padded_lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    return lemma_char_vecs


def predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, alphabet_index, inverse_alphabet_index, lemmas,
                      feats, feat_index, feature_types):
    predictions = {}
    for i, (lemma, feat_dict) in enumerate(zip(lemmas, feats)):
        predicted_sequence = predict_output_sequence(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, lemma,
                                                     feat_dict, alphabet_index, inverse_alphabet_index, feat_index,
                                                     feature_types)

        # index each output by its matching inputs - lemma + features
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predictions[joint_index] = predicted_sequence

    return predictions


def rerank(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                  alphabet_index, feat_index, feature_types, lemma, feats, words, alignments):
    from numpy import argmin, arange, isnan, isinf
    from numpy.random import shuffle
    losses = []
    order = arange(len(alignments))
    shuffle(order)
    prev_blstm = None
    prev_lemma = None
    for i in order:
        alignment = alignments[i]
        word = words[i]
        pc.renew_cg()
        expr_R = pc.parameter(R)
        expr_bias = pc.parameter(bias)
        padded_lemma = BEGIN_WORD + lemma + END_WORD
        # FIXME
        loss = one_word_loss(model, char_lookup, feat_lookup, expr_R, expr_bias, encoder_frnn,
                                                     encoder_rrnn, decoder_rnn, padded_lemma, feats, word,
                                                     alphabet_index, alignment, feat_index, feature_types)
        losses.append(loss.value())
    # print 'losses: {}'.format(losses)
    assert not any(isnan(losses))
    assert not any(isinf(losses))
    return words[order[argmin(losses)]]


def word_from_alignment(alignment):
    return ''.join([unicode(x) for x in alignment[1] if x != '~'])


def sample_decode(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                  alphabet_index, feat_index, feature_types, lemma, feats, sigma, inv_sigma, fst_dir, syms):
    from scipy.misc import logsumexp
    from numpy.random import shuffle
    free_fst = read_fst(lemma, inv_sigma, fst_dir, syms)
    num_free_samples=30
    free_samples = sample(free_fst, sigma, num_free_samples, inv_tau=3e-3)
    order = np.arange(num_free_samples)
    shuffle(order)
    crunched_dict = {}
    padded_lemma = BEGIN_WORD + lemma + END_WORD
    for i in order:
        alignment = free_samples[i]['alignment']
        word = word_from_alignment(alignment)
        weight = free_samples[i]['weight']
        pc.renew_cg()
        expr_R = pc.parameter(R)
        expr_bias = pc.parameter(bias)
        loss = one_word_loss(model, char_lookup, feat_lookup, expr_R, expr_bias, decoder_rnn, padded_lemma, feats, word,
                             alphabet_index, alignment, feat_index, feature_types, encoder_frnn=encoder_frnn,
                             encoder_rrnn=encoder_rrnn)
        l = - loss.value()
        corrected = l - weight
        if word not in crunched_dict:
            crunched_dict[word] = [corrected]
        else:
            crunched_dict[word] += [corrected]
    largest_sum = None
    largest = None
    for w, l in crunched_dict.iteritems():
        s = logsumexp(l)
        if largest is None or s > largest_sum:
            largest = w
            largest_sum = s
    return largest


def sample_decode_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                            alphabet_index, inverse_alphabet_index, lemmas, feats, feat_index, feature_types, sigma,
                            inv_sigma, fst_dir, syms):
    predictions = {}
    for i, (lemma, feat_dict) in enumerate(zip(lemmas, feats)):

        predicted_sequence = sample_decode(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn,
                                           decoder_rnn,
                                           alphabet_index, feat_index, feature_types, lemma, feats,
                                           sigma, inv_sigma, fst_dir, syms)

        # index each output by its matching inputs - lemma + features
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predictions[joint_index] = predicted_sequence

    return predictions


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def evaluate_model(predicted_sequences, lemmas, feature_dicts, words, feature_types, print_results=False):
    if print_results:
        print 'evaluating model...'

    test_data = zip(lemmas, feature_dicts, words)
    c = 0
    for i, (lemma, feat_dict, word) in enumerate(test_data):
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predicted_template = predicted_sequences[joint_index]
        predicted_word = predicted_sequences[joint_index].replace(STEP, '')
        if predicted_word == word:
            c += 1
            sign = u'V'
        else:
            sign = u'X'
        if print_results:# and sign == 'X':
            enc_l = lemma.encode('utf8')
            enc_w = word.encode('utf8')
            enc_t = ''.join([t.encode('utf8') for t in predicted_template])
            enc_p = predicted_word.encode('utf8')
            print 'lemma: {}'.format(enc_l)
            print 'gold: {}'.format(enc_w)
            print 'template: {}'.format(enc_t)
            print 'prediction: {}'.format(enc_p)
            print sign

    accuracy = float(c) / len(predicted_sequences)
    if print_results:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predicted_sequences)) + '=' + \
              str(accuracy) + '\n\n'

    return len(predicted_sequences), accuracy


def evaluate_ndst(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet, feature_types,
                  hidden_dim, hyper_params, input_dim, inverse_alphabet_index, layers, results_file_path,
                  sigmorphon_root_dir, test_feat_dicts, test_lemmas, test_path,
                  test_words, train_path, print_results=False):
    accuracies = []
    final_results = {}
    if ensemble:
        # load ensemble models
        ensemble_model_names = ensemble.split(',')
        print 'ensemble paths:\n'
        print '\n'.join(ensemble_model_names)
        ensemble_models = []
        for ens in ensemble_model_names:
            model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(
                alphabet,
                ens,
                input_dim,
                hidden_dim,
                layers,
                feature_alphabet,
                feat_input_dim,
                feature_types)

            ensemble_models.append((model, encoder_frnn, encoder_rrnn, decoder_rnn))

        # predict the entire test set with each model in the ensemble
        print 'predicting...'
        ensemble_predictions = []
        count = 0
        for em in ensemble_models:
            count += 1
            model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = em
            predicted_sequences = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                                                    alphabet_index,
                                                    inverse_alphabet_index,
                                                    test_lemmas,
                                                    test_feat_dicts,
                                                    feat_index,
                                                    feature_types)
            ensemble_predictions.append(predicted_sequences)
            print 'finished to predict with ensemble: {}/{}'.format(count, len(ensemble_model_names))

        predicted_sequences = {}
        string_to_sequence = {}

        # perform voting for each test input - joint_index is a lemma+feats representation
        test_data = zip(test_lemmas, test_feat_dicts, test_words)
        for i, (lemma, feat_dict, word) in enumerate(test_data):
            joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
            prediction_counter = defaultdict(int)

            # count votes
            for en in ensemble_predictions:
                prediction_str = ''.join(en[joint_index]).replace(STEP, '')
                prediction_counter[prediction_str] += 1
                string_to_sequence[prediction_str] = en[joint_index]
                if print_results:
                    print 'template: {} prediction: {}'.format(en[joint_index].encode('utf8'),
                                                               prediction_str.encode('utf8'))

            # return the most predicted output
            predicted_sequence_string = max(prediction_counter, key=prediction_counter.get)

            # hack: if chosen without majority, pick shortest prediction
            if prediction_counter[predicted_sequence_string] == 1:
                predicted_sequence_string = min(prediction_counter, key=len)

            if print_results:
                print 'chosen:{} with {} votes\n'.format(predicted_sequence_string.encode('utf8'),
                                                         prediction_counter[predicted_sequence_string])

            predicted_sequences[joint_index] = string_to_sequence[predicted_sequence_string]

            # progress indication
            sys.stdout.write("\r%d%%" % (float(i) / len(test_lemmas) * 100))
            sys.stdout.flush()
    else:
        # load best model - no ensemble
        best_model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(alphabet,
                                                                              results_file_path, input_dim,
                                                                              hidden_dim, layers,
                                                                              feature_alphabet, feat_input_dim,
                                                                              feature_types)
        try:
            predicted_sequences = predict_sequences(best_model,
                                                    char_lookup, feat_lookup, R, bias, encoder_frnn,
                                                    encoder_rrnn, decoder_rnn,
                                                    alphabet_index,
                                                    inverse_alphabet_index,
                                                    test_lemmas,
                                                    test_feat_dicts,
                                                    feat_index,
                                                    feature_types)
        except Exception as e:
            print e
            traceback.print_exc()

    # run internal evaluation
    try:
        accuracy = evaluate_model(predicted_sequences,
                                  test_lemmas,
                                  test_feat_dicts,
                                  test_words,
                                  feature_types,
                                  print_results=False)
        accuracies.append(accuracy)
    except Exception as e:
        print e
        traceback.print_exc()

    # get predicted_sequences in the same order they appeared in the original file
    # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
    for i, lemma in enumerate(test_lemmas):
        joint_index = test_lemmas[i] + ':' + common.get_morph_string(test_feat_dicts[i], feature_types)
        inflection = ''.join(predicted_sequences[joint_index]).replace(STEP, '')
        final_results[i] = (test_lemmas[i], test_feat_dicts[i], inflection)

    accuracy_vals = [accuracies[i][1] for i in xrange(len(accuracies))]
    macro_avg_accuracy = sum(accuracy_vals) / len(accuracies)
    print 'macro avg accuracy: ' + str(macro_avg_accuracy)

    mic_nom = sum([accuracies[i][0] * accuracies[i][1] for i in xrange(len(accuracies))])
    mic_denom = sum([accuracies[i][0] for i in xrange(len(accuracies))])
    micro_average_accuracy = mic_nom / mic_denom
    print 'micro avg accuracy: ' + str(micro_average_accuracy)

    if 'test' in test_path:
        suffix = '.best.test'
    else:
        suffix = '.best'

    common.write_results_file_and_evaluate_externally(hyper_params, micro_average_accuracy, train_path,
                                                      test_path, results_file_path + suffix, sigmorphon_root_dir,
                                                      final_results)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path_param = arguments['TRAIN_PATH']
    else:
        train_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['DEV_PATH']:
        dev_path_param = arguments['DEV_PATH']
    else:
        dev_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['TEST_PATH']:
        test_path_param = arguments['TEST_PATH']
    else:
        test_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-dev'
    if arguments['RESULTS_PATH']:
        results_file_path_param = arguments['RESULTS_PATH']
    else:
        results_file_path_param = \
            '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir_param = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
    if arguments['SYMS_PATH']:
        syms_file = arguments['SYMS_PATH']
    else:
        syms_file = '/export/a10/kitsing/fsts/13SIA-13SKE_0/fsts/train/sigma.txt'

    if arguments['--input']:
        input_dim_param = int(arguments['--input'])
    else:
        input_dim_param = INPUT_DIM
    if arguments['--hidden']:
        hidden_dim_param = int(arguments['--hidden'])
    else:
        hidden_dim_param = HIDDEN_DIM
    if arguments['--feat-input']:
        feat_input_dim_param = int(arguments['--feat-input'])
    else:
        feat_input_dim_param = FEAT_INPUT_DIM
    if arguments['--epochs']:
        epochs_param = int(arguments['--epochs'])
    else:
        epochs_param = EPOCHS
    if arguments['--layers']:
        layers_param = int(arguments['--layers'])
    else:
        layers_param = LAYERS
    if arguments['--optimization']:
        optimization_param = arguments['--optimization']
    else:
        optimization_param = OPTIMIZATION
    if arguments['--reg']:
        regularization_param = float(arguments['--reg'])
    else:
        regularization_param = REGULARIZATION
    if arguments['--learning']:
        learning_rate_param = float(arguments['--learning'])
    else:
        learning_rate_param = LEARNING_RATE
    if arguments['--plot']:
        plot_param = True
    else:
        plot_param = False
    if arguments['--eval']:
        eval_param = True
    else:
        eval_param = False
    if arguments['--ensemble']:
        ensemble_param = arguments['--ensemble']
    else:
        ensemble_param = False

    if arguments['--init-epochs']:
        init_epochs_param = int(arguments['--init-epochs'])
    else:
        init_epochs_param = -1

    print arguments

    main(train_path_param, dev_path_param, test_path_param, results_file_path_param, sigmorphon_root_dir_param,
         input_dim_param,
         hidden_dim_param, feat_input_dim_param, epochs_param, layers_param, optimization_param, regularization_param,
         learning_rate_param, plot_param, eval_param, ensemble_param, init_epochs_param, syms_file)


def encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index, feat_lookup, feats,
                           feature_types, lemma):
    feat_vecs = []

    # convert features to matching embeddings, if UNK handle properly
    for feat in sorted(feature_types):

        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feats:
            feat_str = feat + ':' + feats[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                print 'Bad feat: {}'.format(feat_str)
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])

    # convert characters to matching embeddings, if UNK handle properly
    lemma_char_vecs = [char_lookup[alphabet_index[BEGIN_WORD]]]
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK character
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    # add terminator symbol
    lemma_char_vecs.append(char_lookup[alphabet_index[END_WORD]])

    # create bidirectional representation
    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs)
    return blstm_outputs, feat_vecs
