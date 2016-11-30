import os

def main():
    prefixes = ['13SIA-13SKE', '2PKE-z']
    K=30
    for p in prefixes:
        for s in xrange(5):
            os.system('python hard_attention.py --dynet-mem 7000 --input=100 --hidden=100 '
                      '--feat-input=20 --epochs=100 --layers=2 --optimization=ADADELTA '
                      '/home/as1986/git/neural_wfst/res/roee/yoav_outputs/{0}-{1}.train.{2} '
                      '/home/as1986/git/neural_wfst/res/roee/yoav_outputs/{0}-{1}.dev.{2} '
                      '/home/as1986/git/neural_wfst/res/roee/yoav_outputs/{0}-{1}.dev.{2} '
                      '/home/as1986/git/neural_wfst/res/roee/rerank/results.original.{0}-{1}.{2} '
                      '/home/as1986/git/neural_wfst/res/roee/rerank'.format(p, s, K))

if __name__ == '__main__':
    main()