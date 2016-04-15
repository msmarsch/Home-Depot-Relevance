## Matthew Smarsch and Yacine Manseur
## NLP Final Project
## Home Depot Product Search Relevance

import pandas as pd
import sklearn
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import requests
import re
import time
from random import randint
import SpellCheck

'''
START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (
		("'", '&#39;'),
		('"', '&quot;'),
		('>', '&gt;'),
		('<', '&lt;'),
		('&', '&amp;'),
)

def spell_check(s):
	query = '+'.join(s.split())
	time.sleep(randint(0,2))
	request= requests.get("https://www.google.co.uk/search?q=" + query)
	content = request.text
	start=content.find(START_SPELL_CHECK)

	if start > -1:
		start = start + len(START_SPELL_CHECK)
		end = content.find(END_SPELL_CHECK)
		search = content[start:end]
		search = re.sub(r'<[^>]+>', '', search)
		for code in HTML_Codes:
			search = search.replace(code[1], code[0])
		search = search[1:]

	else:
		search = s
	return search
'''

stop = stopwords.words('english')
stemmer = SnowballStemmer('english')

def stem_stop(s):
	return " ".join([stemmer.stem(word) for word in SpellCheck.spell_check(s).split() if word not in stop]) #if word not in stop | spell_check(s)

def get_matches(query, info):
	return sum(int(info.find(word)>=0) for word in query.split())

class Regressor:

	def __init__(self):
		self.load_data()

	def load_data(self):
		#df_attributes = pd.read_csv('../Data/attributes.csv')
		df_prod_desc = pd.read_csv('../Data/product_descriptions.csv')
		df_train = pd.read_csv('../Data/train.csv', encoding="ISO-8859-1")
		df_test = pd.read_csv('../Data/test.csv', encoding="ISO-8859-1")
		#df_prod_comb = pd.merge(df_attributes, df_prod_desc, how = 'right', on = 'product_uid')
		self.df_train_all = pd.merge(df_train, df_prod_desc, how = 'left', on = 'product_uid')
		self.df_test_all = pd.merge(df_test, df_prod_desc, how = 'left', on = 'product_uid')

	def preprocess(self):
		self.df_train_all['search_term'] = self.df_train_all['search_term'].map(lambda x: stem_stop(x))
		self.df_train_all['combined_info'] = self.df_train_all['search_term'] + "\t" + self.df_train_all['product_title'] + "\t" + self.df_train_all['product_description']
	 	self.df_train_all['query_length'] = self.df_train_all['search_term'].map(lambda x: len(x.split()))
	 	self.df_train_all['title_length'] = self.df_train_all['product_title'].map(lambda x: len(x.split()))
	 	self.df_train_all['description_length'] = self.df_train_all['product_description'].map(lambda x: len(x.split()))
	 	#self.df_train_all['attribute_length'] = 
	 	self.df_train_all['title_matches'] = self.df_train_all['combined_info'].map(lambda x: get_matches(x.split('\t')[0], x.split('\t')[1]))
		self.df_train_all['description_matches'] = self.df_train_all['combined_info'].map(lambda x: get_matches(x.split('\t')[0], x.split('\t')[2]))
		self.df_train_all = self.df_train_all.drop(['search_term', 'combined_info', 'product_title', 'product_description'], axis = 1)
		
		self.df_test_all['search_term'] = self.df_test_all['search_term'].map(lambda x: stem_stop(x))
		self.df_test_all['combined_info'] = self.df_test_all['search_term'] + "\t" + self.df_test_all['product_title'] + "\t" + self.df_test_all['product_description']
	 	self.df_test_all['query_length'] = self.df_test_all['search_term'].map(lambda x: len(x.split()))
	 	self.df_test_all['title_length'] = self.df_test_all['product_title'].map(lambda x: len(x.split()))
	 	self.df_test_all['description_length'] = self.df_test_all['product_description'].map(lambda x: len(x.split()))
	 	self.df_test_all['title_matches'] = self.df_test_all['combined_info'].map(lambda x: get_matches(x.split('\t')[0], x.split('\t')[1]))
		self.df_test_all['description_matches'] = self.df_test_all['combined_info'].map(lambda x: get_matches(x.split('\t')[0], x.split('\t')[2]))
		self.df_test_all = self.df_test_all.drop(['search_term', 'combined_info', 'product_title', 'product_description'], axis = 1)

	def build_regressor(self, train_features, train_labels):
		rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
		clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
		clf.fit(train_features, train_labels)
		return clf


if __name__ == '__main__':
	reg = Regressor()
	reg.preprocess()
	test_ids = reg.df_test_all['id']
	train_labels = reg.df_train_all['relevance'].values
	train_features = reg.df_train_all.drop(['id', 'relevance'], axis = 1).values
	test_features = reg.df_test_all.drop(['id'], axis = 1).values
	clf = reg.build_regressor(train_features, train_labels)
	pd.DataFrame({"id": test_ids, "relevance": clf.predict(test_features)}).to_csv('submission.csv', index = False)