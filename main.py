import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

def recommend_items(user_id, user_item_matrix, user_similarity_df, top_n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    sim_sum = similar_users.sum()
    if sim_sum == 0:
        return pd.Series(dtype=float)
    recommendations = user_item_matrix.mul(similar_users, axis=0).sum(axis=0) / sim_sum
    items_to_recommend = recommendations[user_item_matrix.loc[user_id] == 0].sort_values(ascending=False)
    return items_to_recommend.head(top_n)

def evaluate(user_item_matrix, user_similarity_df):
    actual, predicted = [], []
    for user_id in user_item_matrix.index:
        for item_id in user_item_matrix.columns:
            if user_item_matrix.loc[user_id, item_id] > 0:
                original = user_item_matrix.loc[user_id, item_id]
                user_item_matrix.loc[user_id, item_id] = 0
                recs = recommend_items(user_id, user_item_matrix, user_similarity_df, top_n=user_item_matrix.shape[1])
                pred = recs.get(item_id, 0)
                actual.append(original)
                predicted.append(pred)
                user_item_matrix.loc[user_id, item_id] = original
    return np.sqrt(mean_squared_error(actual, predicted))

def main():
    # Load dataset
    ratings = pd.read_csv('ratings.csv')
    user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix)
    user_similarity = cosine_similarity(sparse_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Example: Recommend items for user_id 1
    print(recommend_items(user_id=1, user_item_matrix=user_item_matrix, user_similarity_df=user_similarity_df))
    print(f"Root Mean Squared Error: {evaluate(user_item_matrix, user_similarity_df)}")

if __name__ == "__main__":
    main()