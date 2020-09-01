
from django.shortcuts import render
import json
from keras.models import load_model
import numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import os
import tensorflow as tf
from tensorflow import Graph, Session
import pyttsx3
nltk.download('wordnet')

filename = "./dataset/hagrid.json"
h5_file = "./models/hagrid_emb_v2_2.h5"
with open(filename, 'r') as f:
	d = f.read()
data = json.loads(d) 

#model = load_model(h5_file)

model_graph = Graph()
with model_graph.as_default():
	tf_session = Session()
	with tf_session.as_default():
		model = load_model(h5_file)

word_id = {'a': 0, 'about': 1, 'after': 2, 'am': 3, 'amaz': 4, 'any': 5, 'are': 6, 'awesom': 7, 'bor': 8, 'bye': 9, 'can': 10, 'complete': 11, 'could': 12, 'day': 13, 'did': 14, 'disappearing': 15, 'dislike': 16, 'do': 17, 'don': 18, 'egg': 19, 'eggs': 20, 'enjoying': 21, 'find': 22, 'for': 23, 'found': 24, 'fun': 25, 'gam': 26, 'game': 27, 'go': 28, 'good': 29, 'goodby': 30, 'great': 31, 'hagrid': 32, 'hate': 33, 'have': 34, 'hello': 35, 'help': 36, 'hey': 37, 'hi': 38, 'how': 39, 'i': 40, 'info': 41, 'inform': 42, 'is': 43, 'just': 44, 'like': 45, 'love': 46, 'me': 47, 'need': 48, 'newer': 49, 'next': 50, 'nic': 51, 'not': 52, 'object': 53, 'objective': 54, 'of': 55, 'place': 56, 'play': 57, 'quest': 58, 'region': 59, 'see': 60, 'should': 61, 'suck': 62, 'supposed': 63, 'tell': 64, 'thank': 65, 'thanks': 66, 'that': 67, 'the': 68, 'thi': 69, 'this': 70, 'tir': 71, 'to': 72, 'upd': 73, 'vanish': 74, 'vers': 75, 'what': 76, 'where': 77, 'with': 78, 'wond': 79, 'would': 80, 'you': 81, 'your': 82}
labels = ['call_hagrid', 'egg_details', 'egg_disappear', 'game_details', 'game_dev', 'gen_help', 'goodbye', 'landscape_details', 'neg_reviews', 'pos_reviews']
max_len = 7

def sentence_to_indices(X, word_id, max_len):
	m = X.shape[0]
	X_indices = np.zeros((m, max_len), dtype = np.float64)
	for i in range(m):
		sentence_words = [word for word in X[i].split()]
		j = 0
		for w in sentence_words:
			if w in word_id:
				X_indices[i,j] = word_id[w]
				j+=1
	return X_indices

def cleaner(questions):
	clean_questions = []
	for sent in questions:
		sent = sent.lower()
		sent = re.sub(r"[^'a-zA-Z0-9]"," ", sent)
		clean_questions.append(sent)
	return clean_questions

def stemmer(clean_questions):
	lemmatizer = WordNetLemmatizer()
	stemmer = PorterStemmer()
	stemmed_questions = [stemmer.stem(word) for word in clean_questions]
	lemmatized_questions = [lemmatizer.lemmatize(word, pos = 'n') for word in stemmed_questions]
	lemmatized_questions = [lemmatizer.lemmatize(word, pos = 'v') for word in lemmatized_questions]
	return lemmatized_questions
'''
def tts(text):
	tts = pyttsx3.init() 
	tts.say(text)  
	tts.runAndWait()
'''
# Create your views here.
def index(request):
	context = {'a': 1}
	return render(request, 'index.html', context)

def question_answering(request):
	#print(request)
	#print(request.POST.dict()['question'])
	question = request.POST.dict()['question']
	print(question)

	x = []
	x.append(question.lower())
	x = cleaner(x)
	x = stemmer(x)
	s = np.array(x)
	x = sentence_to_indices(s, word_id, max_len)
	#result = model.predict(x)

	with model_graph.as_default():
		with tf_session.as_default():
			result = model.predict(x)

	result_query_index = np.argmax(result)
	result_query_class = labels[result_query_index]
	for dictionary in data['QA_database']:
		if(dictionary['Query_class'] == result_query_class):
			query_answer = np.random.choice(dictionary['Answers'])
			
	context = {'a': 1, 'predictedLabel': query_answer}
	#context = {'a': 1}
	return render(request, 'index.html', context)


	