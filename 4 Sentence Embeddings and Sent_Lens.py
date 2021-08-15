import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from tqdm import tqdm
punctuations = ["." ,"," , "'" , "!" , ";" , ":" , '"']
print("HERE")
word_vec = Word2Vec.load("True_word_embeddings.model")
print("HERE")
def vectorizer(File_Name):
	f = open(f"{File_Name}.txt","r")
	raw_text = f.read()
	f.close()
	Data0 = raw_text.split("\n")
	Data1 = [word_tokenize(Sentence) for Sentence in Data0]
	Sent_Len = []
	for sentence in tqdm(Data1):
		Sent_Len.append(len(sentence))
	
	List = []
	for Sentence in tqdm(Data1):
		for word in Sentence:
			temp = [word.lower() for word in Sentence if word not in punctuations]
		List.append(temp)

	vectorized_List = []
	for sentence in tqdm(List):
		temp_list = []
		for word in (sentence):
			try:
				temp_list.append(word_vec.wv[word])
			except KeyError:
				print(sentence)
		##vectorized_List.append(sentence)
		vectorized_List.append(temp_list)
		temp_list = []
	return np.array(vectorized_List),np.array(Sent_Len)
	
##Probably not the best way to create a sentence vector but i will replace it with a better one if I find something good
def sentence_vectorizer(word_vectors_of_the_sentence):
	sent_vec = []
	for i in tqdm(range(len(word_vectors_of_the_sentence))):
		temp = 0
		for j in range(len(word_vectors_of_the_sentence[i])):
			temp += word_vectors_of_the_sentence[i][j]
		temp = temp/len(word_vectors_of_the_sentence[i])
		sent_vec.append(temp)
		temp = 0
	return np.asarray(sent_vec).astype("float32")

Sentence_Embeddings,Sent_Len = (vectorizer("Word2Vec"))
Sentence_Embeddings = sentence_vectorizer(Sentence_Embeddings)
np.save("Sentence_Embeddings.npy",Sentence_Embeddings)
np.save("Sent_Len.npy",Sent_Len)