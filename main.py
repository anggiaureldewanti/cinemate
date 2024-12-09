import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# 2. Inspect datasets
# print(movies.head())
# print(ratings.head())

# 3. Merge datasets
data = pd.merge(ratings, movies, on='movieId')
print(data)

# 4. Drop missing values
data.dropna(inplace=True)

# 5. Remove outliers using IQR for 'rating'
Q1 = data['rating'].quantile(0.25)
Q3 = data['rating'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
data = data[(data['rating'] >= (Q1 - 1.5 * IQR)) & (data['rating'] <= (Q3 + 1.5 * IQR))]
print(data)

# 6. Feature Engineering: Extract genres
data['genres'] = data['genres'].str.split('|')

# One-hot encode genres
genres = data['genres'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

# Combine genres with original data
data = pd.concat([data, genres], axis=1)

# Drop original 'genres' column
data.drop(columns='genres', inplace=True)

# 7. Standarize numeric features (rating)
scaler = StandardScaler()
data['rating_scaled'] = scaler.fit_transform(data[['rating']])

# 8. Content-Based Filtering
# Prepare features for similarity computation (genres and scaled rating)
features = genres.columns.tolist() + ['rating_scaled']
movie_features = data.groupby('movieId')[features].mean()

# Compute cosine similarity between movies
similarity_matrix = cosine_similarity(movie_features)

# Function to get valid movieId from title
# Function to get valid movieId from title
def get_movie_id_from_title(title):
    matched_movies = movies[movies['title'].str.contains(title, case=False, na=False)]
    if matched_movies.empty:
        return None
    matched_movies = matched_movies[matched_movies['movieId'].isin(movie_features.index)]
    if matched_movies.empty:
        return None
    return matched_movies['movieId'].tolist()

# Modified recommend_movies_based_on_favorites function
def recommend_movies_based_on_favorites(favorite_movie_ids, n_recommendations=5):
    similarity_scores = np.zeros(similarity_matrix.shape[0])
    
    for movie_id in favorite_movie_ids:
        if movie_id in movie_features.index:
            movie_idx = movie_features.index.tolist().index(movie_id)
            similarity_scores += similarity_matrix[movie_idx]
        else:
            print(f"MovieId {movie_id} tidak ditemukan dalam matriks fitur.")
    
    # Sort by similarity scores
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    # Exclude favorite movies from recommendations
    recommendations = [movie_features.index[i] for i in sorted_indices if movie_features.index[i] not in favorite_movie_ids]
    
    # Retrieve top-n recommendations
    top_recommendations = recommendations[:n_recommendations]
    
    # Include movie titles, genres, and average ratings
    result = pd.DataFrame({
        "movieId": top_recommendations,
        "title": [movies[movies['movieId'] == mid]['title'].values[0] for mid in top_recommendations],
        "genres": [movies[movies['movieId'] == mid]['genres'].values[0] for mid in top_recommendations],
        "rating": [ratings[ratings['movieId'] == mid]['rating'].mean() for mid in top_recommendations],
        "score": [similarity_scores[movie_features.index.tolist().index(mid)] for mid in top_recommendations]
    })
    
    return result

# Example: Input favorite movie titles
print("Masukkan beberapa judul film favorit Anda (pisahkan dengan koma, contoh: Iron Man, Jumanji):")
favorite_movies_input = input().strip().split(',')

# Get movie IDs from titles
favorite_movie_ids = []
for title in favorite_movies_input:
    movie_ids = get_movie_id_from_title(title.strip())
    if movie_ids:
        if len(movie_ids) > 1:
            print(f"Terdapat beberapa film dengan judul '{title.strip()}':")
            for i, mid in enumerate(movie_ids):
                movie_title = movies[movies['movieId'] == mid]['title'].values[0]
                print(f"{i + 1}. {movie_title}")
            choice = int(input(f"Pilih nomor film untuk '{title.strip()}': ")) - 1
            favorite_movie_ids.append(movie_ids[choice])
        else:
            favorite_movie_ids.extend(movie_ids)
    else:
        print(f"Judul '{title.strip()}' tidak ditemukan atau tidak memiliki data rating yang cukup.")

# Check if we have valid movie IDs
if not favorite_movie_ids:
    print("Tidak ada film valid yang ditemukan dari input Anda. Program selesai.")
else:
    # Generate recommendations
    recommended_movies = recommend_movies_based_on_favorites(favorite_movie_ids, n_recommendations=5)
    print("\nRekomendasi film berdasarkan preferensi Anda:")
    print(recommended_movies)