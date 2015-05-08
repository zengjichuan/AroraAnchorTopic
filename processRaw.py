__author__ = 'zengjichuan'

import nltk
import os
import os.path
import re
import csv

def load_data(dir):
    data_folder = dir
    corpus = {}
    for root, dirs, files in os.walk(data_folder):
        for file_name in files:
            if re.match(r'\d+', file_name):
                raw = open(os.path.join(root, file_name)).read()
                corpus[file_name] = raw
    return corpus

if __name__ == '__main__':

    # files' content list
    files_content = []
    vocab_set = set()
    file_output = []
    total_tokens = 0
    total_items = 0

    # file to read
    input_dir = r'F:\KuaiPan\sharedDataset\text\20newsUsedInSentencReglarizerPaper\test'

    # file to write
    basename = os.path.split(input_dir)[1]
    docword_file = os.path.join('uci', 'docword.' + basename + '.txt')
    vocab_file = os.path.join('uci', 'vocab.' + basename + '.txt')
    f_doc_word_out = open(docword_file, 'w')
    f_vocab_out = open(vocab_file, 'w')

    # load stopwords
    stop_words = [w for (w,) in csv.reader(open(os.path.join('uci', 'stopword.en.txt')))]

    # load raw corpus
    my_corpus = load_data(input_dir)

    # generate words bag
    for file_raw in my_corpus:
        print 'processing ', file_raw, my_corpus[file_raw][:20]
        tokens = [w.lower() for w in nltk.word_tokenize(my_corpus[file_raw]) if w.isalnum() and not w.isdigit() and w.lower() not in stop_words]
        vocab_set = vocab_set | set(tokens)
        file_word_bag = dict.fromkeys(set(tokens), 0)
        for w in tokens:
            file_word_bag[w] += 1
            total_tokens += 1
        total_items += len(file_word_bag)
        files_content.append(file_word_bag)

    vocab_map = dict(zip(vocab_set, range(len(vocab_set))))

    # write vocabulary file
    for vocab in vocab_set:
        f_vocab_out.write(vocab+'\n')

    # write doc_word file's intro
    f_doc_word_out.write('%d\n' % len(files_content))
    f_doc_word_out.write('%d\n' % len(vocab_set))
    f_doc_word_out.write('%d\n' % total_items)

    for doc_id in range(len(files_content)):
        doc_word_bag = {}
        for word, counts in files_content[doc_id].items():
            doc_word_bag[vocab_map[word]] = counts
        # sort and print
        for key_id in sorted(doc_word_bag):
            f_doc_word_out.write('%d %d %d\n' % (doc_id+1, key_id + 1, doc_word_bag[key_id]))
    f_vocab_out.close()
    f_doc_word_out.close()

    print 'UCI Text Token Generated!'