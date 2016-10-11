# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys,json

log = logging.getLogger()
##log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
##log.addHandler(ch)

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


from gensim.summarization import keywords
import MySQLdb
import nltk
import string
##import six
##import rake
##import operator
##import io

# connect
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="buds")
cursor = db.cursor()

### EXAMPLE ONE - SIMPLE
##stoppath = "stopwords.txt"
##
### 1. initialize RAKE by providing a path to a stopwords file
##rake_object = rake.Rake(stoppath, 4, 1, 1)

# pass variable from PHP
##try:
##    grouppost_id = json.loads(sys.argv[1])
##    
##except:
##    print "ERROR"
##    sys.exit(1)
grouppost_id = '12'
##datas = [str(x) for x in data]
from nltk.corpus import stopwords
raw = open("C:\Python27\word2vec\stopwords.txt", "rU")
stop = raw.read()

cursor.execute("SELECT posts FROM groupposts WHERE grouppost_id=%s",grouppost_id)
test = cursor.fetchone()
strtest = str(test).lower()
test_punc = "".join(l for l in strtest if l not in string.punctuation)
test_tok = nltk.word_tokenize(test_punc)
test_stop = [i for i in test_tok if i not in stop]
new_test = " ".join(test_stop)

with open("Test.txt", "w") as text_file:
    text_file.writelines(new_test)
 
print 'Uji:'
print test
##uji = keywords(new_test,split=True)
##print 'Keyword uji'
##print uji

### 2. run on RAKE on a given text
##uji2 = rake_object.run(datauji)
##print str(uji2).split()

# execute SQL select statement
cursor.execute("SELECT posts FROM groupposts WHERE group_id='1' ")

# commit your changes
db.commit()

# get the number of rows in the resultset
numrows = int(cursor.rowcount)

# create sources
with open("Output.txt", "w") as text_file:
    for x in range(0,numrows):
        data = cursor.fetchone()
        strdata = str(data).lower()
        data_punc = "".join(l for l in strdata if l not in string.punctuation)
        data_tok = nltk.word_tokenize(data_punc)
        data_stop = [i for i in data_tok if i not in stop]
        new_data = " ".join(data_stop)
        with open("Output.txt", "a") as text_file:
            text_file.writelines(new_data)
            text_file.writelines("\n")

sources = {'Output.txt':'TRAIN_POS','Test.txt':'TEST'}

##log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

##log.info('D2V')
model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=8, alpha=0.025, hs=1, sorted_vocab=1, iter=5)
model.build_vocab(sentences.to_array())

##log.info('Epoch')
for epoch in range(3):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm())

##log.info('Model Save')
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

#loop for similarity
cursor.execute("SELECT posts FROM groupposts WHERE group_id='1'")
numrow = int(cursor.rowcount)
max_sim = 0
for i in range (0,numrow):
    row = cursor.fetchone()
    strrow = str(row).lower()
    row_punc = "".join(l for l in strrow if l not in string.punctuation)
    row_tok = nltk.word_tokenize(row_punc)
    row_stop = [i for i in row_tok if i not in stop]
    new_row = " ".join(row_stop)

    with open("Data.txt", "w") as text_file:
        text_file.writelines(new_row)

    result = model.n_similarity(test_stop,row_stop)
    if result>0.3:
        if max_sim < result:
            max_sim = result
        else:
            max_sim = max_sim
        print 'Data:'
        print row
        ##data = keywords(new_row,split=True)
        ##print 'Keyword data:'
        ##print data

        ### 2. run on RAKE on a given text
        ##data2 = rake_object.run(text)
        ##print str(data2).split()

        ##print model.vocab
        ##print model.most_similar('rumah')
        print "Similarity: "
        print model.n_similarity(test_stop,row_stop)
        
print "Maximal: "
print max_sim











