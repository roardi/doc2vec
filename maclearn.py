import MySQLdb
import nltk
import string
import logging
import sys,json
import os
import gensim

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

from nltk.corpus import stopwords

log = logging.getLogger()
log.setLevel(logging.DEBUG)

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

# connect
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="media_buds")
cursor = db.cursor()

# pass variable from PHP
##try:
##    grouppost_id = json.loads(sys.argv[1])
##    
##except:
##    print "ERROR"
##    sys.exit(1)
grouppost_id = '14'

raw = open("C:\Python27\word2vec\stopwords.txt", "rU")
stop = raw.read()

cursor.execute("SELECT grouppost_id,posts FROM groupposts WHERE grouppost_id=%s",grouppost_id)
test_fetch = cursor.fetchall()
for tes in test_fetch:
    gpid = tes[0]
    post = tes[1]
strtest = str(post).lower()
test_punc = "".join(l for l in strtest if l not in string.punctuation)
test_tok = nltk.word_tokenize(test_punc)
test_stop = [i for i in test_tok if i not in stop]
new_test = " ".join(test_stop)

with open("Test.txt", "w") as text_file:
    text_file.writelines(new_test)

# execute SQL select statement
cursor.execute("SELECT posts FROM groupposts WHERE group_id='1' AND status='2' ")

# commit your changes
db.commit()

# get the number of rows in the resultset
numrows = int(cursor.rowcount)

# create sources
with open("Output.txt", "w") as text_file:
    for x in range(0,numrows):
        data_fetch = cursor.fetchone()
        for datas in data_fetch:
            data1 = datas
        strdata = str(data1).lower()
        data_punc = "".join(l for l in strdata if l not in string.punctuation)
        data_tok = nltk.word_tokenize(data_punc)
        data_stop = [i for i in data_tok if i not in stop]
        new_data = " ".join(data_stop)
        with open("Output.txt", "a") as text_file:
            text_file.writelines(new_data)
            text_file.writelines("\n")
            
train_sources = {'Output.txt':'TRAIN','Test.txt':'TEST'}
##test_sources = {'Test.txt':'TEST'}

##log.info('TaggedDocument')
sentences = TaggedLineSentence(train_sources)

##log.info('D2V')
model = Doc2Vec(min_count=1, window=5, size=200, workers=3, alpha=0.025, iter=5)
model.build_vocab(sentences)

model.train(sentences)

##log.info('Model Save')
model.save('./maclearn.d2v')
model = Doc2Vec.load('./maclearn.d2v')

##print model.docvecs['TRAIN_2']

def json_list(array1,array2,array3):
    arr = []
    for i in range(0,len(array1)):
        d = {}
        arr.append({"id":array1[i],"data":array2[i],"similarity":array3[i]})
    return arr

#loop for similarity
cursor.execute("SELECT grouppost_id,posts FROM groupposts WHERE group_id='1' AND status='2'")
row = cursor.fetchone()
numrow = int(cursor.rowcount)

max_sim = 0
id_row = []
isi = []
sims = []

while row is not None:
    gpid_row = row[0]
    post_row = row[1]
    
    strrow = str(post_row).lower()
    row_punc = "".join(l for l in strrow if l not in string.punctuation)
    row_tok = nltk.word_tokenize(row_punc)
    row_stop = [i for i in row_tok if i not in stop]
    new_row = " ".join(row_stop)
    
    with open("Data.txt", "w") as text_file:
        text_file.writelines(new_row)

    similar = model.n_similarity(test_stop,row_stop)
    result = round(100*similar,2)
    if result>50:
        id_row.append(gpid_row)
        isi.append(post_row)        
        sims.append(result)
        if max_sim < result:
            max_sim = result
        else:
            max_sim = max_sim

    row = cursor.fetchone()
        
arrr = json_list(id_row,isi,sims)
row = {
    "gpid":gpid,
    "uji":post,
    "row":arrr,       
    "max":max_sim
    }
print json.dumps(row)

  





