from dynet import *
import time
import random

LAYERS = 2
INPUT_DIM = 256 #50  #256
HIDDEN_DIM = 256 # 50  #1024
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import sys
import util

import numpy as np
from scipy.misc import logsumexp


class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder):
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    def BuildLMGraph(self, sent):
        renew_cg()
        init_state = self.builder.initial_state()

        R = parameter(self.R)
        bias = parameter(self.bias)
        errs = [] # will hold expressions
        es=[]
        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            # assume word is already a word-id
            x_t = lookup(self.lookup, int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = esum(errs)
        return nerr

    def sample(self, first=1, nchars=0, stop=-1):
        res = [first]
        renew_cg()
        state = self.builder.initial_state()

        R = parameter(self.R)
        bias = parameter(self.bias)
        cw = first
        prob = 0.

        while True:
            x_t = lookup(self.lookup, cw)
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            ydist = log_softmax(r_t).npvalue()
            dist = np.exp(ydist)
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            prob += ydist[i]
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res, prob

    def evaluate(self, upto):
        renew_cg()
        state = self.builder.initial_state()

        R = parameter(self.R)
        bias = parameter(self.bias)
        prob = 0.
        for idx, (cw, next_cw) in enumerate(zip(upto[:-1], upto[1:])):
            x_t = lookup(self.lookup, cw)
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            next_prob = pickneglogsoftmax(r_t, next_cw)
            prob -= next_prob.value()
        return prob

    def evaluate_single_state(self, state, last, next_up, R=None, bias=None):
        """
        evaluates a single state. won't renew cg.
        :param state:
        :param last:
        :param next_up:
        :param R:
        :param bias:
        :return: next state, prob
        """

        if R is None:
            R = parameter(self.R)
        if bias is None:
            bias = parameter(self.bias)
        x_t = lookup(self.lookup, last)
        state = state.add_input(x_t)
        y_t = state.output()
        r_t = bias + (R * y_t)
        next_prob = pickneglogsoftmax(r_t, next_up)
        return state, next_prob


class Sample:
    def __init__(self, lm, log_prob, history, num_sources, states=None):
        if states is None:
            self.states = [lm.builder.initial_state() for _ in num_sources]
        else:
            self.states = states
        self.log_prob = log_prob
        self.history = history


def particle_filter(lm, samples, last, next_up, num_sources, R, bias):
    assert isinstance(lm, RNNLanguageModel)
    candidates = [None] * (num_sources * len(samples))
    for source in xrange(num_sources):
        for s_idx, s in enumerate(samples):
            assert isinstance(s, Sample)
            next_state, next_prob = lm.evaluate_single_state(s.states[source], last, next_up, R, bias)
            updated_weight = s.log_prob + next_prob.value()
            candidates[source * num_sources + s_idx] = Sample(lm, updated_weight, s.history + [source], s.states)
            candidates[source * num_sources + s_idx].states[source] = next_state

    candidate_log_probs = [x.log_prob for x in candidates]
    z = logsumexp(candidate_log_probs)
    normalized = [np.exp(x-z) for x in candidate_log_probs]
    chosen = np.random.choice(len(candidates), len(samples), p=normalized)
    to_return = []
    for c in chosen:
        candidates[c].log_prob = normalized[c]
        to_return.append(candidates[c])
    return candidates

if __name__ == '__main__':
    train = util.CharsCorpusReader(sys.argv[1],begin="<s>")
    vocab = util.Vocab.from_corpus(train)
    
    VOCAB_SIZE = vocab.size()

    model = Model()
    sgd = SimpleSGDTrainer(model)

    #lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder)
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=LSTMBuilder)

    train = list(train)

    example_sentence = ['<s>', 't', 'h', 'e', ' ', 'N', '.', '\n']
    example_encoded = [vocab.w2i[x] for x in example_sentence]

    chars = loss = 0.0
    for ITER in xrange(100):
        random.shuffle(train)
        for i,sent in enumerate(train):
            _start = time.time()
            if i % 50 == 0:
                sgd.status()
                if chars > 0: print loss / chars,
                for _ in xrange(1):
                    samp, prob = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
                    print "".join([vocab.i2w[c] for c in samp]).strip()
                    print 'log prob: {}'.format(prob)
                    print 'example logprob: {}'.format(lm.evaluate(example_encoded))
                loss = 0.0
                chars = 0.0
                
            chars += len(sent)-1
            isent = [vocab.w2i[w] for w in sent]
            errs = lm.BuildLMGraph(isent)
            loss += errs.scalar_value()
            errs.backward()
            sgd.update(1.0)
            #print "TM:",(time.time() - _start)/len(sent)
        print "ITER",ITER,loss
        sgd.status()
        sgd.update_epoch(1.0)
