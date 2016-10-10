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

# connect
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="buds")
cursor = db.cursor()

# pass variable from PHP
##try:
##    grouppost_id = json.loads(sys.argv[1])
##    
##except:
##    print "ERROR"
##    sys.exit(1)
grouppost_id = '12'
##datas = [str(x) for x in data]
print 'Keywords uji:'

cursor.execute("SELECT posts FROM groupposts WHERE grouppost_id=%s",grouppost_id)
data2 = cursor.fetchone()
datauji = str(data2)
uji = keywords(datauji, split=True)
print uji

# execute SQL select statement
cursor.execute("SELECT posts FROM groupposts WHERE group_id='1' ")

# commit your changes
db.commit()

# get the number of rows in the resultset
numrows = int(cursor.rowcount)

# create sources
with open("Output.txt", "w") as text_file:
    for x in range(0,numrows):
        rows = cursor.fetchone()
        with open("Output.txt", "a") as text_file:
            text_file.writelines(rows)
            text_file.writelines("\n")

sources = {'Output.txt':'TRAIN_POS'}

##log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

##log.info('D2V')
model = Doc2Vec(min_count=1, window=2, size=100, sample=1e-4, negative=5, workers=8, alpha=0.025, hs=1, sorted_vocab=1, iter=5)
model.build_vocab(sentences.to_array())

##log.info('Epoch')
for epoch in range(1):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm())

##log.info('Model Save')
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

#loop for similarity
cursor.execute("SELECT posts FROM groupposts WHERE group_id='1' ")
numrow = int(cursor.rowcount)
for i in range (0,numrow):
    row = cursor.fetchone()
    text = str(row)
##    print text
    print 'Keywords data:'
    data = keywords(text, split=True)
    print data

    ##print model.vocab
    ##print model.most_similar(datas)
    print model.n_similarity(uji,data)
    ##print model.accuracy(sentences)
    ##distance = model.wmdistance(text,data2)
    ##print 100-distance







