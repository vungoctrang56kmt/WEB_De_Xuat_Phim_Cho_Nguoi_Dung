from flask import Flask, request, render_template  # Import các module từ Flask để tạo ứng dụng web
import pandas as pd  # Import module pandas để làm việc với dữ liệu dạng DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer từ scikit-learn để vector hóa văn bản
from nltk.corpus import stopwords  # Import stopwords từ nltk để loại bỏ các từ phổ biến không quan trọng
import spacy  # Import spacy để thực hiện lemmatization
from scipy.spatial.distance import cdist  # Import cdist từ scipy để tính khoảng cách cosine

app = Flask(__name__)  # Khởi tạo ứng dụng Flask

# Load data
nx_df = pd.read_csv("D:/KhoaHocDuLieu/KhoaHocDuLieu/data/netflix_titles.csv")  # Đọc dữ liệu từ file CSV

# Lemmatization
nlp = spacy.load('en_core_web_sm')  # Tải mô hình ngôn ngữ tiếng Anh từ spaCy
lemmatok = lambda doc: [token.lemma_ for token in nlp(doc) if token.lemma_ != '-PRON-' and not token.is_punct]  # Hàm lambda để thực hiện lemmatization cho các từ trong văn bản
nx_df_lem = nx_df.copy()  # Sao chép DataFrame gốc
nx_df_lem['description'] = nx_df_lem['description'].apply(lambda x: " ".join(lemmatok(x)))  # Áp dụng lemmatization cho cột 'description'

# Vectorization
tf_unibi = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords.words('english'), min_df=2, max_features=4000)  # Khởi tạo TfidfVectorizer với các tham số nhất định
tf_tr_ub = tf_unibi.fit_transform(nx_df_lem['description'])  # Vector hóa dữ liệu văn bản

# DataFrame with TF-IDF features
tf_df = pd.DataFrame(tf_tr_ub.toarray(), columns=tf_unibi.get_feature_names_out(), index=nx_df_lem['show_id'])  # Tạo DataFrame chứa các đặc trưng TF-IDF

@app.route('/', methods=['GET', 'POST'])  # Định nghĩa route '/' cho trang chính của ứng dụng, cho phép phương thức GET và POST
def search():
    if request.method == 'POST':  # Nếu phương thức là POST
        query = request.form['query']  # Lấy query từ form gửi đi
        if query:  # Nếu query không rỗng
            # Lemmatize the query
            query_lem = " ".join(lemmatok(query))  # Thực hiện lemmatization cho query
            # Vectorize the query
            query_vec = tf_unibi.transform([query_lem])  # Vector hóa query
            # Calculate cosine similarity between query and all descriptions
            similarities = 1 - cdist(query_vec.toarray(), tf_df, 'cosine')  # Tính cosine similarity giữa query và tất cả các văn bản
            # Sort indices based on similarity scores
            similar_indices = similarities.argsort()[0][::-1]  # Sắp xếp các chỉ số dựa trên điểm tương đồng
            # Extract titles based on indices
            similar_titles = nx_df.loc[similar_indices[:10], 'title'].tolist()  # Lấy danh sách các tiêu đề tương tự
            return render_template('results.html', query=query, results=similar_titles)  # Trả về template kết quả tìm kiếm
    return render_template('index.html')  # Trả về template trang chính nếu phương thức không phải là POST

if __name__ == '__main__':  # Nếu đây là file chính được chạy
    app.run(debug=True)  # Khởi động ứng dụng Flask ở chế độ debug
