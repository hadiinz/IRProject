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
# stop words
persian_stopwords = stopwords("fa")

# opening JSON file
f = open('IR_data_news_12k.json')
# JSON object as a dictionary
documents = json.load(f)
f.close()

# Lists to store words and their frequencies
words_frequencies = Counter()

# iteration on documents
for docID in documents:
    # normalize
    normal_list = my_normalizer.normalize(documents[docID]["content"])
    # tokenize
    normal_token = my_tokenizer.tokenize_words(normal_list)

    # Count word frequencies after normalization and tokenization
    word_frequencies_in_dict = Counter(normal_token)

    # Add word frequencies to the overall frequencies Counter
    words_frequencies.update(word_frequencies_in_dict)

# Sort words based on frequency in descending order
sorted_words = sorted(words_frequencies.items(), key=lambda x: x[1], reverse=True)
# Extract the top 50 most frequent words
top_50_words = [[word, freq] for word, freq in sorted_words[:50]]

# Write the list of words and their frequencies to a file
output_file_path = 'removed_words_frequencies.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write("Words and Frequencies Sorted by Frequency:\n")
    for word, frequency in top_50_words:
        output_file.write(f"{word}: {frequency}\n")

# Remove the top 50 most frequent words from the entire corpus
filtered_list = [[word, freq] for word, freq in sorted_words[50:]]

# Write the filtered list to a file
filtered_list_file_path = 'filtered_list.txt'
with open(filtered_list_file_path, 'w', encoding='utf-8') as filtered_list_file:
    filtered_list_file.write("Filtered List:\n")
    for word, frequency in filtered_list:
        filtered_list_file.write(f"{word}: {frequency}\n")

# Print a message indicating successful writing to files
print(f"Results written to {output_file_path} and {filtered_list_file_path}")
