from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi aplikasi Flask dengan direktori template
app = Flask(__name__, static_url_path='/static')

# Baca data lagu dari CSV
data_lagu = pd.read_csv("data_lagu_bersih.csv")

# Gabungkan kolom 'lyrics' dan 'deskripsi' menjadi satu teks
data_lagu['teks'] = data_lagu['lyrics'] + " " + data_lagu['deskripsi']

# Inisialisasi TF-IDF Vectorizer dari model yang telah disimpan
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Fitur ekstraksi dari kolom 'teks'
tfidf_matrix = tfidf_vectorizer.transform(data_lagu['teks'])

@app.route('/', methods=['GET', 'POST'])
def home():
    cerita_pengguna = None

    if request.method == 'POST':
        cerita = request.form['cerita']
        rekomendasi = rekomendasikan_lagu(cerita)
        cerita_pengguna = cerita
        

        return render_template('hasil_rekomendasi.html', cerita_pengguna=cerita_pengguna, rekomendasi=rekomendasi)

    return render_template('index.html', cerita_pengguna=cerita_pengguna)


# Fungsi untuk merekomendasikan lagu berdasarkan cerita
def rekomendasikan_lagu(cerita):
    cerita_vector = tfidf_vectorizer.transform([cerita])
    cosine_similarities = cosine_similarity(cerita_vector, tfidf_matrix)
    lagu_rekomendasi = data_lagu.copy()
    lagu_rekomendasi['similarity_score'] = cosine_similarities[0]
    lagu_rekomendasi = lagu_rekomendasi.sort_values(by='similarity_score', ascending=False)
    rekomendasi_lagu = lagu_rekomendasi.head(3)
    tampilan_rekomendasi = rekomendasi_lagu[['track_name', 'artist_name']]
    
    # Kembalikan hasil rekomendasi dalam bentuk daftar
    return tampilan_rekomendasi.values.tolist()



if __name__ == '__main__':
    app.run(debug=True)
