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

def recommend_items(user_id, user_item_matrix, user_similarity_df, top_n=5):
    # Get the similarity scores for the given user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]  # Exclude the user itself

    # Weighted sum of ratings from similar users
    recommendations = user_item_matrix.mul(similar_users, axis=0).sum(axis=0) / similar_users.sum()

    # Recommend items that the user hasn't rated yet
    items_to_recommend = recommendations[user_item_matrix.loc[user_id] == 0].sort_values(ascending=False)

    return items_to_recommend.head(top_n)

# Example: Recommend items for user_id 1
print(recommend_items(user_id=1, user_item_matrix=user_item_matrix, user_similarity_df=user_similarity_df))
