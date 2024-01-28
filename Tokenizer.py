import json
from parsivar import Normalizer, Tokenizer, FindStems
from collections import Counter
import math

#  preprocessing
# normalizer
my_normalizer = Normalizer()
# tokenizer
my_tokenizer = Tokenizer()
# stemmer
my_stemmer = FindStems()

# Opening JSON file
f = open('dataset/data_5000.json')
# JSON object as a dictionary
documents = json.load(f)
f.close()

# Dictionary to store words with their frequency, list of documents, term frequency, positions, and IDF in each document
word_dict = {}

# Number of documents in the corpus
num_documents = len(documents)

# Lists Champions structure to store related documents for each word
lists_champions = {}

# Iteration on documents
for docID in documents:
    # normalize
    normal_list = my_normalizer.normalize(documents[docID]["content"])
    # tokenize
    normal_token = my_tokenizer.tokenize_words(normal_list)

    # Stemming
    stemmed_token = [my_stemmer.convert_to_stem(w) for w in normal_token]

    # Count word frequencies after normalization, tokenization, and stemming
    word_frequencies_in_dict = Counter(stemmed_token)

    # Update the word_dict with word frequencies, list of documents, term frequency, positions, and IDF in each document
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

        # Update Lists Champions
        if word not in lists_champions:
            lists_champions[word] = [{'docID': docID, 'tf-idf': 0}]  # Initialize with an empty list
        else:
            lists_champions[word].append({'docID': docID, 'tf-idf': 0})  # Initialize with an empty TF-IDF score

# Calculate IDF for each word in word_dict
for word, data in word_dict.items():
    idf = math.log(num_documents / (1 + len(data['documents'])))
    data['idf'] = idf

# Calculate TF-IDF for each term in Lists Champions
for word, docs in lists_champions.items():
    for entry in docs:
        tf = word_dict[word]['documents'][entry['docID']]['term_frequency']
        idf = word_dict[word]['idf']
        entry['tf-idf'] = tf * idf

# Sort documents within each list based on TF-IDF score
for word, docs in lists_champions.items():
    lists_champions[word] = sorted(docs, key=lambda x: x['tf-idf'], reverse=True)


# Sort words based on frequency in descending order
sorted_words = sorted(word_dict.items(), key=lambda x: x[1]['frequency'], reverse=True)

# Extract the top 50 most frequent words
top_50_words = [(word, data['frequency'], data['idf']) for word, data in sorted_words[:50]]

# Write the list of words, their frequencies, and IDF values to a file
output_file_path = 'outputFiles/removed_words_frequencies.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write("Words and Frequencies Sorted by Frequency:\n")
    for word, frequency, idf in top_50_words:
        output_file.write(f"{word}: {frequency}, IDF: {idf}\n")

# Remove the top 50 most frequent words from the entire corpus
filtered_list = [{'word': word, 'frequency': data['frequency'], 'documents': data['documents'], 'idf': data['idf']} for word, data in sorted_words[50:]]

# Exclude the top 50 most frequent words from Lists Champions
top_50_words_set = set(word for word, _, _ in top_50_words)
for word in top_50_words_set:
    if word in lists_champions:
        lists_champions.pop(word, None)

# Write the filtered list to a JSON file
filtered_list_json_path = 'outputFiles/filtered_list.json'
with open(filtered_list_json_path, 'w', encoding='utf-8') as filtered_list_json_file:
    filtered_list_json_file.write(json.dumps(filtered_list, indent=2, ensure_ascii=False))

# Write Lists Champions to a JSON file
lists_champions_json_path = 'outputFiles/lists_champions.json'
with open(lists_champions_json_path, 'w', encoding='utf-8') as lists_champions_json_file:
    lists_champions_json_file.write(json.dumps(lists_champions, indent=2, ensure_ascii=False))

# Print a message indicating successful writing to files
print(f"Results written to {output_file_path}, {filtered_list_json_path}, and {lists_champions_json_path}")
