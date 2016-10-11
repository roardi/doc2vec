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
try:
    grouppost_id = json.loads(sys.argv[1])
    
except:
    print "ERROR"
    sys.exit(1)
##grouppost_id = '12'
##datas = [str(x) for x in data]
cursor.execute("SELECT posts FROM groupposts WHERE grouppost_id=%s",grouppost_id)
test = cursor.fetchone()
strtest = str(test).lower()
with open("Test.txt", "w") as text_file:
    text_file.writelines(strtest)
 
print 'Uji:'
uji = keywords(strtest,ratio=0.1, split=True)
print test

### 2. run on RAKE on a given text
##uji2 = rake_object.run(datauji)
##print str(uji2).split()

#loop for similarity
cursor.execute("SELECT posts FROM groupposts WHERE grouppost_id='15'")
numrow = int(cursor.rowcount)
##for i in range (0,numrow):
row = cursor.fetchone()
strrow = str(row).lower()
with open("Data.txt", "w") as text_file:
    text_file.writelines(strrow)
    
print 'Data:'
data = keywords(strrow,ratio=0.1, split=True)
print row

### 2. run on RAKE on a given text
##data2 = rake_object.run(text)
##print str(data2).split()

# execute SQL select statement
cursor.execute("SELECT posts FROM groupposts WHERE group_id='1' ")

# commit your changes
db.commit()

# get the number of rows in the resultset
numrows = int(cursor.rowcount)

##raw = open("stopwords.txt", "rU")
##stop = raw.read()
# create sources
with open("Output.txt", "w") as text_file:
    for x in range(0,numrows):
        aa = cursor.fetchone()
##        print aa
##        rowss = [i for i in aa if i not in stop]
##        isi = str(aa).lower()
##        print isi
        with open("Output.txt", "a") as text_file:
            text_file.writelines(aa)
##            text_file.writelines("\n")

sources = {'Output.txt':'TRAIN_POS','Test.txt':'TEST'}

##log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

##log.info('D2V')
model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=8, alpha=0.025, hs=1, sorted_vocab=1, iter=5)
model.build_vocab(sentences.to_array())

##log.info('Epoch')
for epoch in range(5):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm())

##log.info('Model Save')
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

##print model.vocab
##print model.most_similar('tangga')
print model.n_similarity(uji,data)
##print model.accuracy(sentences)
    ##distance = model.wmdistance(text,data2)
    ##print 100-distance

##train_arrays = numpy.zeros((200, 100))
##train_labels = numpy.zeros(200)
##for i in range(1):
##    prefix_train_pos = 'DATA_'+ str(i)
##    train_arrays[i] = model.docvecs[prefix_train_pos]
##    train_labels[i] = 1
##print train_arrays
##
##test_arrays = numpy.zeros((200, 100))
##test_labels = numpy.zeros(200)
##for i in range(1):
##    prefix_test_pos = 'TEST_'+ str(i)
##    test_arrays[i] = model.docvecs[prefix_test_pos]
##    test_labels[i] = 0
##print test_arrays









