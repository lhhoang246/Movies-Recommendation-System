from flask import Flask, render_template, request, jsonify
from rating_recommender import RatingRecommender
from movie_recommender import MovieRecommender

app = Flask(__name__)

# Khởi tạo 2 hệ thống gợi ý
rec_rating = RatingRecommender("filter_cleaned.csv", "movies_last.csv")
rec_content = MovieRecommender("movies_last.csv", "tmdb_5000_credits.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None
    user_id_display = ""
    method = "rating"  # mặc định

    if request.method == "POST":
        method = request.form.get("method")
        password = request.form.get("password")

        # ✅ Chỉ yêu cầu mật khẩu nếu chọn hệ thống rating
        if method == "rating" and password != "123":
            error = "Sai mật khẩu!"
        elif method == "rating":
            login_type = request.form.get("login_type")
            user_input = request.form.get("user_input")
            try:
                user_input = int(user_input)
                if login_type == "stt":
                    stt = user_input
                    results = rec_rating.recommend(stt)
                    if results and results[0][0] == "STT không tồn tại":
                        error = results[0][0]
                        results = []
                    else:
                        uid = rec_rating.get_user_id_from_stt(stt)
                        user_id_display = f"Gợi ý theo rating cho User ID: {uid}"
                elif login_type == "user_id":
                    stt = rec_rating.user_id_to_stt.get(user_input)
                    if stt is None:
                        error = "User ID không tồn tại!"
                    else:
                        results = rec_rating.recommend(stt)
                        user_id_display = f"Gợi ý theo rating cho User ID: {user_input}"
            except:
                error = "Lỗi: STT/User ID không hợp lệ!"
        elif method == "content":
            title = request.form.get("movie_title")
            results = rec_content.recommend(title)
            if not results:
                error = f"Không tìm thấy phim '{title}'!"
            else:
                user_id_display = f"Gợi ý theo nội dung cho: {title}"

    return render_template("index_combined.html", results=results, error=error, user_id_display=user_id_display, method=method)

# ✅ API autocomplete cho gợi ý tên phim động
@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("q", "").lower()
    suggestions = []
    if query:
        titles = rec_content.movies['title'].dropna().tolist()
        suggestions = [t for t in titles if t.lower().startswith(query)]
    return jsonify(suggestions[:5])

if __name__ == "__main__":
    app.run(debug=True)
