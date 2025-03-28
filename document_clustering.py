import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. Load Documents
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

document_directory = "my_documents"  # Ensure 'my_documents' is in the same directory as this script.
documents = load_documents(document_directory)

# 2. Preprocess Text
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered_tokens)

processed_documents = [preprocess_text(doc) for doc in documents]

# 3. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_documents)

# 4. Clustering (K-Means)
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)
clusters = kmeans.labels_

# 5. Display Results
for i in range(num_clusters):
    print(f"\nCluster {i + 1}:")
    for j, cluster_label in enumerate(clusters):
        if cluster_label == i:
            print(f"- {documents[j]}")