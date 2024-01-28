import json
import math
from parsivar import Normalizer, Tokenizer, FindStems
from collections import Counter
import time

def calculate_cosine_similarity(query_vector, document_vector):
    dot_product = sum(query_vector.get(term, 0) * document_vector.get(term, 0) for term in set(query_vector) & set(document_vector))
    query_norm = math.sqrt(sum(value ** 2 for value in query_vector.values()))
    document_norm = math.sqrt(sum(value ** 2 for value in document_vector.values()))

    if query_norm == 0 or document_norm == 0:
        return 0  # To avoid division by zero

    return dot_product / (query_norm * document_norm)

def retrieve_documents(query_vector, tfidf_vectors, idf_threshold, min_query_terms):
    # Filter terms based on IDF threshold
    eligible_terms = [term for term, idf in query_vector.items() if idf > idf_threshold]

    # Filter documents that contain at least one term from the query
    relevant_documents = {doc_id: document_vector for doc_id, document_vector in tfidf_vectors.items() if any(term in document_vector for term in eligible_terms)}

    similarity_scores = {}

    for doc_id, document_vector in relevant_documents.items():
        # Filter documents based on containing many or all query terms
        if len(set(eligible_terms) & set(document_vector.keys())) >= min_query_terms:
            similarity = calculate_cosine_similarity(query_vector, document_vector)
            similarity_scores[doc_id] = similarity

    # Sort documents by similarity scores in descending order
    sorted_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # Eliminate documents with zero similarity
    sorted_documents = [(doc_id, similarity) for doc_id, similarity in sorted_documents if similarity != 0]

    return sorted_documents



def retrieve_documents_using_champions(query_vector, lists_champions, tfidf_vectors, idf_threshold, min_query_terms):
    # Filter terms based on IDF threshold
    eligible_terms = [term for term, idf in query_vector.items() if idf > idf_threshold]

    # Get relevant documents from lists_champions
    relevant_documents = set()
    for term in eligible_terms:
        if term in lists_champions:
            relevant_documents.update(entry['docID'] for entry in lists_champions[term])

    similarity_scores = {}

    for doc_id in relevant_documents:
        document_vector = tfidf_vectors.get(doc_id, {})  # Get the TF-IDF vector for the document
        # Filter documents based on containing many or all query terms
        if len(set(eligible_terms) & set(document_vector.keys())) >= min_query_terms:
            similarity = calculate_cosine_similarity(query_vector, document_vector)
            similarity_scores[doc_id] = similarity

    # Sort documents by similarity scores in descending order
    sorted_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # Eliminate documents with zero similarity
    sorted_documents = [(doc_id, similarity) for doc_id, similarity in sorted_documents if similarity != 0]

    return sorted_documents



start_time = time.time()
# Load the TF-IDF vectors from the file
with open('outputFiles/tfidf_vectors.json', 'r', encoding='utf-8') as tfidf_file:
    tfidf_vectors = json.load(tfidf_file)
# Load the filtered_list.json file to get IDF values
with open('outputFiles/filtered_list.json', 'r', encoding='utf-8') as filtered_list_file:
    filtered_list = json.load(filtered_list_file)


# Load the lists_champions.json file
with open('outputFiles/lists_champions.json', 'r', encoding='utf-8') as lists_champions_file:
    lists_champions = json.load(lists_champions_file)


# Example query
user_query = "ورزش"

# Preprocess the user query similar to the document preprocessing

#  preprocessing
# normalizer
my_normalizer = Normalizer()
# tokenizer
my_tokenizer = Tokenizer()
# stemmer
my_stemmer = FindStems()

normalized_query = my_normalizer.normalize(user_query)
tokenized_query = my_tokenizer.tokenize_words(normalized_query)
stemmed_query = [my_stemmer.convert_to_stem(w) for w in tokenized_query]

# Calculate TF for the query
query_tf = Counter(stemmed_query)

# Calculate IDF for the query terms using the precomputed values from filtered_list.json
query_idf = {term: next((item['idf'] for item in filtered_list if item['word'] == term), 0) for term in query_tf}

# Calculate TF-IDF for the query
query_tfidf = {term: query_tf[term] * query_idf[term] for term in query_tf}
print(query_idf)
# Set IDF threshold and minimum number of query terms required in a document
idf_threshold = 0.2 # Set  desired IDF threshold
min_query_terms = 1  # Set  desired minimum number of query terms


########################################## using normal postings-list ###################################################
# # Retrieve documents based on similarity and index elimination
# retrieved_documents = retrieve_documents(query_tfidf, tfidf_vectors, idf_threshold, min_query_terms)

########################################## using champlion-list ###################################################
# Retrieve documents using champions and index elimination
retrieved_documents = retrieve_documents_using_champions(query_tfidf, lists_champions, tfidf_vectors, idf_threshold, min_query_terms)

# Display the top results
num_results_to_display = 5
for i, (doc_id, similarity) in enumerate(retrieved_documents[:num_results_to_display]):
    print(f"Rank {i+1}: Document ID {doc_id}, Similarity Score: {similarity}")

finish_time = time.time()
print( "time:" )
print(+finish_time - start_time)
