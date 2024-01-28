import json
from stopwordsiso import stopwords
from parsivar import Normalizer, Tokenizer, FindStems
from collections import Counter

#  preprocessing
# normalizer
my_normalizer = Normalizer()
# tokenizer
my_tokenizer = Tokenizer()
# stemmer
my_stemmer = FindStems()

# opening JSON file
f = open('dataset/data_100.json')
# JSON object as a dictionary
documents = json.load(f)
f.close()

# Dictionary to store words with their frequency, list of documents, and term frequency in each document
word_dict = {}

# iteration on documents
for docID in documents:
    # normalize
    normal_list = my_normalizer.normalize(documents[docID]["content"])
    # tokenize
    normal_token = my_tokenizer.tokenize_words(normal_list)

    # Stemming
    stemmed_token = [my_stemmer.convert_to_stem(w) for w in normal_token]

    # Count word frequencies after normalization, tokenization, and stemming
    word_frequencies_in_dict = Counter(stemmed_token)

    # Update the word_dict with word frequencies, list of documents, and term frequency in each document
    for word, frequency in word_frequencies_in_dict.items():
        if word not in word_dict:
            word_dict[word] = {'frequency': frequency, 'documents': {docID: {'term_frequency': frequency, 'positions': []}}}
        else:
            word_dict[word]['frequency'] += frequency
            if docID not in word_dict[word]['documents']:
                word_dict[word]['documents'][docID] = {'term_frequency': frequency, 'positions': []}
            else:
                word_dict[word]['documents'][docID]['term_frequency'] += frequency

        # Extract positions of each term in the document
        positions = [pos for pos, term in enumerate(stemmed_token) if term == word]
        word_dict[word]['documents'][docID]['positions'].extend(positions)

# Sort words based on frequency in descending order
sorted_words = sorted(word_dict.items(), key=lambda x: x[1]['frequency'], reverse=True)

# Extract the top 50 most frequent words
top_50_words = [(word, data['frequency']) for word, data in sorted_words[:50]]

# Write the list of words and their frequencies to a file
output_file_path = 'outputFiles/removed_words_frequencies.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write("Words and Frequencies Sorted by Frequency:\n")
    for word, frequency in top_50_words:
        output_file.write(f"{word}: {frequency}\n")

# Remove the top 50 most frequent words from the entire corpus
filtered_list = [{'word': word, 'frequency': data['frequency'], 'documents': data['documents']} for word, data in sorted_words[50:]]

# Write the filtered list to a JSON file
filtered_list_json_path = 'outputFiles/filtered_list.json'
with open(filtered_list_json_path, 'w', encoding='utf-8') as filtered_list_json_file:
    filtered_list_json_file.write(json.dumps(filtered_list, indent=2, ensure_ascii=False))

# Print a message indicating successful writing to files
print(f"Results written to {output_file_path} and {filtered_list_json_path}")
