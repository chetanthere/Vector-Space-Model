
# this is to read files
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.book import *
import collections
import math


import os
corpus_root = 'C:\Python35/presidential_debates'
dfiles = []
filename_list = []
for filename in os.listdir(corpus_root):
   #Storing Filename in a list
   filename_list.append(filename)
   file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
   doc = file.read()
   file.close()
   doc = doc.lower()
   dfiles.append(doc)   

   
# tokenize each debate file
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
tokens = []
for i in range(30):
    tokens.append(tokenizer.tokenize(dfiles[i]))
      

#to remove stop words
stopwordslist = (sorted(stopwords.words('english')))
newtokens = []
ntind = []
newtokenslist = []

for k in range(30):
    ntind = []
    for i in tokens[k]:
        matchf = 0
        for j in stopwordslist:
            if i == j:
                matchf = 1
                break
        if matchf == 0:            
            newtokens.append(i)
            ntind.append(i)
    newtokenslist.append(ntind)            


# stemming tokens
stemmer = PorterStemmer()
newtokens2 = []
nt2ind = []
newtokens2list = []
for i in newtokens:
    val = stemmer.stem(i)
    newtokens2.append(val)
   

#changes for ind list
for k in range(30):
    nt2ind = []
    for i in newtokenslist[k]:
        val = stemmer.stem(i)
        nt2ind.append(val)
    
    newtokens2list.append(nt2ind)   
      
      
# This is to remove duplicate stemming tokens
set2 = set(newtokens2)
res = list(set2)
ressorted = []
ressorted = sorted(res)


#CALCULATING NORMALIZED VECTOR FOR ALL DOCS
#Step 1 : {[(),()],[],...}form a dictionary of all docs with (term,frequency) in that doc 
docfslist = []
#for i in range(1,30):
for i in range(30):
    fd1 = FreqDist(newtokens2list[i])     
    len1 = len(fd1)
    doc1f = {}
    doc1fs= {}
    doc1f = dict(fd1.most_common(len1))
    doc1fs = collections.OrderedDict(sorted(doc1f.items()))   
    docfslist.append(doc1fs)
      
      
#Step 2 : Calculating tfidf matrix for each document
doc1fs_tf = {}
doc1fstflist = []
doc1fs_tfidf = {}
doc1fstfidflist = []
docsf_tfidf = {}
docsfs_tfidf = {}
docsfstfidflist = []

for im in range(30):
    doc1fs = docfslist[im]
    doc1fstflist = []
    doc1fstfidflist = []
    for k,v in doc1fs.items():
        tfval = 1 + math.log10(v)
        df = 0
        for i in range(30):
            tp = []
            tp = newtokens2list[i]
            if k in tp:
                df = df + 1
            
        idfval = math.log10(30/df)
        tfidfval = tfval * idfval
        
        doc1fstflist.append((k,tfval))
        doc1fstfidflist.append((k,tfidfval))

    doc1fs_tf = {}
    doc1fs_tfidf = {} 
    doc1fs_tf = dict(doc1fstflist)
    doc1fs_tfidf = dict(doc1fstfidflist)

    doc1fs_tfs = {}
    doc1fs_tfidfs = {}

    doc1fs_tfidfs = dict(collections.OrderedDict(sorted(doc1fs_tfidf.items())))
     
    docsfstfidflist.append(doc1fs_tfidfs)


#Normalizing each vecor
docs_length = []
for i in range(30):
    sumval = 0
    doc1fs_tfidfs = {}
    doc1fs_tfidfs = dict(docsfstfidflist[i])
    for k,v in doc1fs_tfidfs.items():
        addval = v * v
        sumval = sumval + addval
       
    doc1fs_length = math.sqrt(sumval)
  
    docs_length.append(doc1fs_length)

docstfidf_nmvlist = []
for i in range(30):
    tfidf_nmvlist = []
    doc1fs_tfidfs = {}
    doc1fs_tfidfs = dict(docsfstfidflist[i])
    for k,v in doc1fs_tfidfs.items():
        divval = v / docs_length[i]
        tfidf_nmvlist.append((k,divval))
    docstfidf_nmvlist.append(tfidf_nmvlist)
    
docstfidf_nmvlist_d = {}
docstfidf_nmvlist_dlist = []

for i in range(30):
    doc1tfidf_nmvlist = docstfidf_nmvlist[i]
    docstfidf_nmvlist_d = dict(doc1tfidf_nmvlist)
    docstfidf_nmvlist_dlist.append(docstfidf_nmvlist_d)

print("Processing Completed")


#Calculating query vecor and finding cosine similarity 
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
qtokens = []
abc = []

def query(qstring):
    qtokens = []
    abc = []   
    doc = qstring.lower()    
    abc.append(tokenizer.tokenize(doc))
    qtokens = abc[0]
    qnewtokenslist = []

    for i in qtokens:
        matchf = 0
        for j in stopwordslist:
            if i == j:
                matchf = 1
                break
        if matchf == 0:           
            qnewtokenslist.append(i)    

    #stemming
    stemmer = PorterStemmer()
    qnewtokens2 = []
    for i in qnewtokenslist:
        val = stemmer.stem(i)        
        qnewtokens2.append(val)     
    
    # forming dictionary of term-frequency
    fd1 = FreqDist(qnewtokens2)    
    len1 = len(fd1)   
    qf = {}
    qfs= {}
    qf = dict(fd1.most_common(len1))
    
    qfs = collections.OrderedDict(sorted(qf.items()))
    
    #forming tf vector for query
    qfs_tf = {}
    qfstflist = []

    for k,v in qfs.items():
        tfval = 1 + math.log10(v)         
        qfstflist.append((k,tfval))
    
    qfs_tf = dict(qfstflist)
    
    #Normalizing query vector
    sumval = 0
    for k,v in qfs_tf.items():
        addval = v * v
        sumval = sumval + addval       
    qfs_length = math.sqrt(sumval)   

    qtf_nmvlist = []
    for k,v in qfs_tf.items():
        divval = v / qfs_length
        qtf_nmvlist.append((k,divval))    
    
    qtf_nmv = {}
    qtf_nmv = dict(qtf_nmvlist)   

    #to chk value of length of normalizes vector
    sumval = 0
    for k,v in qtf_nmv.items():
        addval = v * v
        sumval = sumval + addval
    qnmlength = math.sqrt(sumval)
        
    #to check the similarity 
    similarity = []
    for i in range(30):
        sumval = 0
        tpd = {}
        tpd = docstfidf_nmvlist_dlist[i]        
        doc1k = []
        doc1k = tpd.keys()      
        for k,v in qtf_nmv.items():
            if k in doc1k:
                vd = tpd[k]               
                mulval = v * vd
                sumval = sumval + mulval
        similarity.append(sumval)       
    
    maxsim = 0    
    for i in range(30):
        if similarity[i] > maxsim:
            maxsim = similarity[i]
            maxdoc = i
    fi = maxdoc
    
    return(filename_list[fi])


def getcount(str):
    count_str = newtokens2.count(str)    
    return(count_str)
    
   
def getidf(str):
    df = 0
    for i in range(30):
        tp = []
        tp = newtokens2list[i]
        if str in tp:
            df = df + 1
            
    idfval = math.log10(30/df)
    
    return(idfval)
    

def docdocsim(str1,str2):    
    for i in range(30):
        if filename_list[i] == str1:            
            doc1_dict = docstfidf_nmvlist_dlist[i]            
        if filename_list[i] == str2:            
            doc2_dict = docstfidf_nmvlist_dlist[i]
               
    sumval = 0
    doc2k = []
    doc2k = doc2_dict.keys()       
    for k,v in doc1_dict.items():
        if k in doc2k:
            vd = doc2_dict[k]          
            mulval = v * vd
            sumval = sumval + mulval
        
    return(sumval)

           
#Calculating query vecor and finding cosine similarity
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
qtokens = []
abc = []

def query2(qstring):
    qtokens = []
    abc = []    
    doc = qstring.lower()   
    abc.append(tokenizer.tokenize(doc))
    qtokens = abc[0]    
    qnewtokenslist = []

    for i in qtokens:
        matchf = 0
        for j in stopwordslist:
            if i == j:
                matchf = 1
                break
        if matchf == 0:           
            qnewtokenslist.append(i)
       
    #stemming
    stemmer = PorterStemmer()
    qnewtokens2 = []
    for i in qnewtokenslist:
        val = stemmer.stem(i)       
        qnewtokens2.append(val)
        
       
    # forming dictionary of term-frequency
    fd1 = FreqDist(qnewtokens2)     
    len1 = len(fd1)
        
    qf = {}
    qfs= {}
    qf = dict(fd1.most_common(len1))
    
    qfs = collections.OrderedDict(sorted(qf.items()))
        
    #forming tf vector for query
    
    qfs_tf = {}
    qfstflist = []

    for k,v in qfs.items():
        tfval = 1 + math.log10(v)          
        qfstflist.append((k,tfval))
    
    qfs_tf = dict(qfstflist)
        
    #Normalizing query vector
    sumval = 0
    for k,v in qfs_tf.items():
        addval = v * v
        sumval = sumval + addval
        
    qfs_length = math.sqrt(sumval)    

    qtf_nmvlist = []
    for k,v in qfs_tf.items():
        divval = v / qfs_length
        qtf_nmvlist.append((k,divval))    
    
    global  qtf_nmv2
    qtf_nmv2 = {}
    qtf_nmv2 = dict(qtf_nmvlist)

def querydocsim(str1,str2):
    
    query2(str1)  
    
    for i in range(30):
        if filename_list[i] == str2:            
            doc2_dict = docstfidf_nmvlist_dlist[i]
               
    sumval = 0
    doc2k = []
    doc2k = doc2_dict.keys()        
    for k,v in qtf_nmv2.items():
        if k in doc2k:
            vd = doc2_dict[k]            
            mulval = v * vd
            sumval = sumval + mulval
       
    return(sumval)



