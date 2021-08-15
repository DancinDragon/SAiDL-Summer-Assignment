from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
print("Here")
punctuations = ["." ,"," , "'" , "!" , ";" , ":" , '"']
f = open("Word2Vec.txt")
data = f.read()
f.close()
data0 = data.split("\n")
print("Here")
data1 = [word_tokenize(sentence) for sentence in data0]
data2 = []
for sentence in tqdm(data1):
	templist = []
	for word in sentence:
		if word not in punctuations:
			templist.append(word.lower())
	data2.append(templist)
	templist = []
word_vec = Word2Vec(sentences=data2,vector_size=300,window=5,min_count=1,workers=4)
word_vec.save("True_word_embeddings.model")
