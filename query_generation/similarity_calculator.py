from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityCalculator:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 2))
    
    def find_similar_query(self, queries, negative_queries):
        if len(queries) < 2:
            return None, 0
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(queries)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        sorted_pairs = []
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                if similarity_matrix[i, j] > 0:
                    sorted_pairs.append((i, j, similarity_matrix[i, j]))
        
        sorted_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, similarity in sorted_pairs:
            query1, query2 = queries[i], queries[j]
            if query1 not in negative_queries and query2 not in negative_queries:
                return query1, similarity
        
        return None, 0