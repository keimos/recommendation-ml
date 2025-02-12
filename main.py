import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load dataset
# ratings = pd.read_csv('ratings.csv')

# Create a pivot table: rows = users, columns = items, values = ratings
user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')

# Fill missing values with 0 (or you can use mean imputation)
user_item_matrix = user_item_matrix.fillna(0)

# Convert the user-item matrix into a sparse matrix
sparse_matrix = csr_matrix(user_item_matrix)

# Compute cosine similarity between users
user_similarity = cosine_similarity(sparse_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
