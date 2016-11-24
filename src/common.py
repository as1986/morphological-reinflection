import align
import codecs
import os
import heapq

NULL = '%'

def cluster_data_by_morph_type(feat_dicts, feature_types):
    morphs_to_indices = {}
    for i, feat_dict in enumerate(feat_dicts):
        s = get_morph_string(feat_dict, feature_types)
        if s in morphs_to_indices:
            morphs_to_indices[s].append(i)
        else:
            morphs_to_indices[s] = [i]
    return morphs_to_indices


def get_morph_string(feat_dict, feature_types):
    s = ''
    for f in sorted(feature_types):
        if f in feat_dict:
            s += f + '=' + feat_dict[f] + ':'
        else:
            s += f + '=' + NULL + ':'
    s = s[:-1]
    return s


def cluster_data_by_pos(feat_dicts):
    pos_to_indices = {}
    pos_key = 'pos'
    for i, d in enumerate(feat_dicts):
        if pos_key in d:
            s = pos_key + '=' + d[pos_key]
        else:
            s = pos_key + '=' + NULL
        if s in pos_to_indices:
            pos_to_indices[s].append(i)
        else:
            pos_to_indices[s] = [i]
    return pos_to_indices


def get_single_pseudo_cluster(feat_dicts):
    key = 'all'
    cluster_to_indices = {key:[]}
    for i, d in enumerate(feat_dicts):
        cluster_to_indices[key].append(i)
    return cluster_to_indices


def argmax(iterable, n=1):
    if n==1:
        return max(enumerate(iterable), key=lambda x: x[1])[0]
    else:
        return heapq.nlargest(n, xrange(len(iterable)), iterable.__getitem__)


def get_feature_alphabet(feat_dicts):
    feature_alphabet = []
    for f_dict in feat_dicts:
        for f in f_dict:
            feature_alphabet.append(f + ':' + f_dict[f])
    feature_alphabet = list(set(feature_alphabet))
    return feature_alphabet


def dumb_align(wordpairs, align_symbol):
    alignedpairs = []
    for idx, pair in enumerate(wordpairs):
        ins = pair[0]
        outs = pair[1]
        if len(ins) > len(outs):
            outs += align_symbol * (len(ins) - len(outs))
        elif len(outs) > len(ins):
            ins += align_symbol * (len(outs) - len(ins))
            alignedpairs.append((ins, outs))
    return alignedpairs


def mcmc_align(wordpairs, align_symbol, k=500):
    to_return = []
    to_return.append(align.Aligner(wordpairs, align_symbol=align_symbol).alignedpairs)
    for _ in xrange(1, k):
        to_return.append(dumb_align(wordpairs, align_symbol=align_symbol))
    return to_return


def med_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol, mode='med')
    return a.alignedpairs

def write_results_file_and_evaluate_externally(hyper_params, accuracy, train_path, test_path, output_file_path,
                                               sigmorphon_root_dir,
                                               final_results, nbest=False):
    if 'test' in test_path:
        output_file_path += '.test'

    if 'dev' in test_path:
        output_file_path += '.dev'

    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train path = ' + str(train_path) + '\n')
        f.write('test path = ' + str(test_path) + '\n')

        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')

        f.write('Prediction Accuracy = ' + str(accuracy) + '\n')

    # write predictions in sigmorphon format
    # if final results, write the special file name format
    if 'test-covered' in test_path:
        if 'task1' in test_path:
            task='1'
        if 'task2' in test_path:
            task='2'
        if 'task3' in test_path:
            task='3'

        if 'blstm' in output_file_path:
            model_dir = 'solutions/blstm'
        elif 'nfst' in output_file_path:
            model_dir = 'solutions/nfst'
        else:
            model_dir = 'solutions'

        if nbest:
            model_dir += '/nbest'

        results_prefix = '/'.join(output_file_path.split('/')[:-1])
        lang = train_path.split('/')[-1].replace('-task{0}-train'.format(task),'')
        predictions_path = '{0}/{3}/{1}-task{2}-solution'.format(results_prefix, lang, task, model_dir)
    else:
        predictions_path = output_file_path + '.predictions'
        if nbest:
            predictions_path += '.nbest'

    with codecs.open(test_path, 'r', encoding='utf8') as test_file:
        lines = test_file.readlines()
        print 'len of test file is {}'.format(len(lines))
        print 'len of predictions file is {}'.format(len(final_results))
        with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(lines):
                if 'test-covered' in test_path:
                    lemma, morph = line.split()
                else:
                    lemma, morph, word = line.split()
                if i in final_results:
                    if nbest:
                        for p in final_results[i][2]:
                            predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, p))
                    else:
                        predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, final_results[i][2]))
                else:
                    # TODO: handle unseen morphs?
                    # print u'could not find prediction for {0} {1}'.format(lemma, morph)
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, 'ERROR'))

    # evaluate with sigmorphon script
    evaluation_path = output_file_path + '.evaluation'
    os.chdir(sigmorphon_root_dir)
    os.system('python ' + sigmorphon_root_dir + '/src/evalm.py --gold ' + test_path + ' --guesses ' + predictions_path +
              ' > ' + evaluation_path)
    os.system('python ' + sigmorphon_root_dir + '/src/evalm.py --gold ' + test_path + ' --guesses ' + predictions_path)

    print 'wrote results to: ' + output_file_path + '\n' + evaluation_path + '\n' + predictions_path
    return


def mirror_data(train_target_words, train_source_words, train_target_feat_dicts, train_source_feat_dicts):
    mirrored_train_target_words = train_target_words + train_source_words
    mirrored_train_source_words = train_source_words + train_target_words
    mirrored_train_target_feat_dicts = train_target_feat_dicts + train_source_feat_dicts
    mirrored_train_source_feat_dicts = train_source_feat_dicts + train_target_feat_dicts

    return mirrored_train_target_words, \
           mirrored_train_source_words, \
           mirrored_train_target_feat_dicts, \
           mirrored_train_source_feat_dicts