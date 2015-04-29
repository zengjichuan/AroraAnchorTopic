__author__ = 'admin'

from anchorTopic import mine_topics
from ioProcess import load_uci, write_topics

def driver_uci(basename, min_tf=0, min_tfidf=0, ntopic=100):
    (vocab, Q) = load_uci(basename, min_tf=min_tf, min_tfidf=min_tfidf)
    (p, r, A, TW) = mine_topics(Q, ntopic)
    write_topics('topics_%s.txt' % basename, vocab, p, r, A, TW)

driver_uci('nips', min_tf=10)