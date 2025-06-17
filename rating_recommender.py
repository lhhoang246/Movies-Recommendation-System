import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RatingRecommender:
    def __init__(self, ratings_path, movies_path):
        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)

        # ✅ Chỉ giữ các dòng có movieId nằm trong danh sách phim hợp lệ
        ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

        self.movies = movies
        self.ratings = ratings

        # Ánh xạ STT ↔ userId
        self.stt_to_user_id = pd.Series(ratings['userId'].values, index=ratings['stt']).to_dict()
        self.user_id_to_stt = pd.Series(ratings['stt'].values, index=ratings['userId']).to_dict()

        # Ma trận user - movie
        self.user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
        self.user_movie_matrix.fillna(0, inplace=True)

        # Tính độ tương đồng giữa người dùng
        self.similarity = cosine_similarity(self.user_movie_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

    def get_user_id_from_stt(self, stt):
        return self.stt_to_user_id.get(stt)

    def recommend(self, stt, n=10):
        user_id = self.get_user_id_from_stt(stt)
        if user_id is None or user_id not in self.user_movie_matrix.index:
            return [("STT không tồn tại", 0, "")]

        user_ratings = self.user_movie_matrix.loc[user_id]
        unseen_movies = user_ratings[user_ratings == 0].index

        scores = {}
        for movie in unseen_movies:
            sim_scores = self.similarity_df[user_id]
            movie_raters = self.user_movie_matrix[movie] > 0
            num_raters = movie_raters.sum()

            # ✅ Chỉ lấy các phim có ít nhất 5 người đã đánh giá
            if num_raters >= 5:
                weighted_sum = (sim_scores * self.user_movie_matrix[movie]).sum()
                sim_total = sim_scores[movie_raters].sum()
                if sim_total > 0:
                    scores[movie] = weighted_sum / sim_total

        top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for mid, score in top_n:
            movie_info = self.movies[self.movies['movieId'] == mid]
            if not movie_info.empty:
                title = movie_info['title'].values[0]
                poster = movie_info['poster_url'].values[0] if 'poster_url' in movie_info else ""
                results.append((title, round(score, 2), poster))
        return results
