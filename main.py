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
