# Import các module cần thiết từ Flask và các thư viện khác
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import spacy
from scipy.spatial.distance import cdist

# Tạo một ứng dụng Flask
app = Flask(__name__)

# Load dữ liệu từ tệp CSV
nx_df = pd.read_csv("D:/KhoaHocDuLieu/KhoaHocDuLieu/data/netflix_titles.csv")

# Lemmatization (chuẩn hóa từ)
nlp = spacy.load('en_core_web_sm')  # Tải mô hình ngôn ngữ spaCy
lemmatok = lambda doc: [token.lemma_ for token in nlp(doc) if token.lemma_ != '-PRON-' and not token.is_punct]  # Hàm chuẩn hóa từ
nx_df_lem = nx_df.copy()  # Sao chép dataframe gốc
nx_df_lem['description'] = nx_df_lem['description'].apply(lambda x: " ".join(lemmatok(x)))  # Áp dụng chuẩn hóa từ cho cột 'description'

# Vectorization (biến đổi văn bản thành vectors số)
tf_unibi = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords.words('english'), min_df=2, max_features=4000)  # Tạo vectorizer với n-grams (1,2), loại bỏ stop words và giới hạn số lượng từ
tf_tr_ub = tf_unibi.fit_transform(nx_df_lem['description'])  # Fit và transform cột 'description'

# Tạo DataFrame với các đặc trưng TF-IDF
tf_df = pd.DataFrame(tf_tr_ub.toarray(), columns=tf_unibi.get_feature_names_out(), index=nx_df_lem['show_id'])  # Chuyển đổi ma trận TF-IDF thành DataFrame

# Định nghĩa route và hàm xử lý cho trang chủ
@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':  # Kiểm tra nếu request là POST
        query = request.form['query']  # Lấy truy vấn từ form
        if query:
            # Lemmatize the query (chuẩn hóa từ truy vấn)
            query_lem = " ".join(lemmatok(query))
            # Vectorize the query (biến đổi truy vấn thành vector)
            query_vec = tf_unibi.transform([query_lem])
            # Calculate cosine similarity between query and all descriptions (tính độ tương tự cosine giữa truy vấn và tất cả mô tả)
            similarities = 1 - cdist(query_vec.toarray(), tf_df, 'cosine')
            # Sort indices based on similarity scores (sắp xếp các chỉ số dựa trên điểm số tương tự)
            similar_indices = similarities.argsort()[0][::-1]
            # Extract titles based on indices (lấy tiêu đề dựa trên các chỉ số)
            similar_titles = nx_df.loc[similar_indices[:10], 'title'].tolist()
            return render_template('results.html', query=query, results=similar_titles)  # Trả về trang kết quả với truy vấn và các kết quả
    return render_template('index.html')  # Trả về trang tìm kiếm nếu không có truy vấn

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)  # Chạy ứng dụng ở chế độ debug
