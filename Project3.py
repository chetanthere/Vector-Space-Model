'''
This program builds Vector space model for given documents of US election debates from 1960 till date
Each document represented by tf-idf vector
For a given user query it finds most relevant document and returns back to user

Author : Chetan There

'''

# Import NLTK and other required modules
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
import os

# Read files
corpus_root = 'presidential_debates'

dfiles = []
filename_list = []
for filename in os.listdir(corpus_root):
    # Storing Filename in a list
    filename_list.append(filename)
    file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()
    dfiles.append(doc)

tot_docs = len(dfiles)
# Tokenize each debate file
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
df_tokens = []
for i in range(tot_docs):
    df_tokens.append(tokenizer.tokenize(dfiles[i]))

# Remove stop words
stopwordslist = (sorted(stopwords.words('english')))
df_tokens_list = []
for i in range(tot_docs):
    temp_list = []
    temp_list = [x for x in df_tokens[i] if x not in stopwordslist]
    df_tokens_list.append(temp_list)

# Remove Temporary data
temp_list = []
df_tokens = []

# Stemming tokens
stemmer = PorterStemmer()
df_tokens_list2 = []
for i in range(tot_docs):
    temp_list = []
    temp_list = [stemmer.stem(x) for x in df_tokens_list[i] ]
    df_tokens_list2.append(temp_list)

# Remove Temporary data
temp_list = []
df_tokens_list = []

# Forming Normalized Vector for each document
df_tokens_list3 = []
for i in range(tot_docs):
    temp_dict = {}
    temp_dict = {x:df_tokens_list2[i].count(x) for x in df_tokens_list2[i] }
    df_tokens_list3.append(temp_dict)

# Remove Temporary data
temp_dict = {}
df_tokens_list2 = []

df_tokens_list4 = []
for item_dict in df_tokens_list3:
    temp_dict2 = {}
    for k,v in item_dict.items():
        tf = 0
        tf = 1 + math.log10(v)
        df = 0
        for i in range(tot_docs):
            temp_dict = {}
            temp_dict = df_tokens_list3[i]
            if k in temp_dict:
                df += 1
        idf = 0
        idf = math.log10(tot_docs / df)
        tfidf = tf * idf
        temp_dict2[k] = tfidf

    df_tokens_list4.append(temp_dict2)

# Remove Temporary data
temp_dict = {}
temp_dict2 = {}
df_tokens_list3 = []

# Normalising each vector
df_tokens_list5 = []
for doc_dict in df_tokens_list4:
    doc_list = list(doc_dict.values())
    doc_list2 = []
    doc_list2 = [x*x for x in doc_list]
    doc_len = sum(doc_list2)
    doc_len_sqrt = math.sqrt(doc_len)
    temp_dict = {}
    for k, v in doc_dict.items():
        temp_dict[k] = v / doc_len_sqrt
    df_tokens_list5.append(temp_dict)

print('Processing Completed')

# Finding cosine similarity of query and returns most relevant doc
#def query(qstring):

print('Enter a query')
qstring = input().strip()

qdoc = qstring.lower()
qdoc_tokens = tokenizer.tokenize(qdoc)
# Remove stop words
qdoc_tokens2 = [x for x in qdoc_tokens if x not in stopwordslist]
# Stemming
qdoc_tokens3 = [stemmer.stem(x) for x in qdoc_tokens2]
# Forming Normalized query string
qdoc_dict = {}
qdoc_dict = {x: qdoc_tokens3.count(x) for x in qdoc_tokens3}

qdoc_dict2 = {}
for k,v in qdoc_dict.items():
    tf = 0
    tf = 1 + math.log10(v)
    qdoc_dict2[k] = tf

# Remove Temporary Data
qdoc_tokens = []
qdoc_tokens2 = []
qdoc_tokens3 = []
qdoc_dict = {}

qdoc_list = list(qdoc_dict2.values())
qdoc_list2 = []
qdoc_list2 = [x * x for x in qdoc_list]
qdoc_len = sum(qdoc_list2)
qdoc_len_sqrt = math.sqrt(qdoc_len)

qdoc_dict3 = {}
for k, v in qdoc_dict2.items():
    qdoc_dict3[k] = v / qdoc_len_sqrt

#Finding cosine similarity
cos_sim_list = []
for doc  in df_tokens_list5:
    cos_sim = 0
    for k,v in doc.items():
        if k in qdoc_dict3:
            cos_sim += v * qdoc_dict3[k]
    cos_sim_list.append(cos_sim)

max_cos_sim = max(cos_sim_list)
index = 0
for i in cos_sim_list:
    if i == max_cos_sim:
        break
    index += 1

print('Most relevent doc = ', filename_list[index])

exit(0)

