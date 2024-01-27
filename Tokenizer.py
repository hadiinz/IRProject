import json
from stopwordsiso import stopwords
from parsivar import Normalizer, Tokenizer, FindStems


#  preprocessing
# normalizer
my_normalizer = Normalizer()
# tokenizer
my_tokenizer = Tokenizer()
# stemmer
my_stemmer = FindStems()
# stop words
persian_stopwords = stopwords("fa")
# print(persian_stopwords)

# opening JSON file
f = open('IR_data_news_12k.json')
# JSON object as a dictionary
documents = json.load(f)
f.close()

positional_index = {}

list_stem_stopword_token_normal = []