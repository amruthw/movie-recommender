import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess dataset
movies = pd.read_csv('movies.csv')
movies = movies.dropna(subset=['genres']).reset_index(drop=True)
movies_subset = movies.head(1000)

# TF-IDF vectorization of genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_subset['genres'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to index
indices = pd.Series(movies_subset.index, index=movies_subset['title'])

# Recommendation function
def recommend_movie(title):
    if title not in indices:
        print(f"\nMovie '{title}' not found. Please try another title.\n")
        return
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]

    print(f"\nTop 10 movies similar to '{title}':\n")
    for i, movie in enumerate(movies_subset['title'].iloc[movie_indices]):
        print(f"{i + 1}. {movie}")
    print()

# User interaction
print("Movie Recommender System")
print("Sample movies you can try:\n")
print(movies_subset['title'].sample(10).to_string(index=False), "\n")

while True:
    user_input = input("Enter a movie title (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    recommend_movie(user_input)
