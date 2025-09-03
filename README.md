🛡️ Spam Email Detection

Machine Learning tabanlı email spam tespit sistemi. FastAPI ile geliştirilmiş web uygulaması.

🎯 Proje Hakkında

Bu proje, email dosyalarını analiz ederek spam/ham sınıflandırması yapan bir web uygulamasıdır. İki farklı ML algoritması (Naive Bayes ve SVM) kullanarak yüksek doğruluk oranı sağlar.
Geliştirici: Fatih Mehmet Çıldır 

⚡ Özellikler

🤖 İki farklı ML modeli (Naive Bayes & SVM)
📧 .eml dosya formatı desteği
📊 Detaylı analiz sonuçları (olasılık oranları)
🚀 Hızlı ve kolay kullanım
💻 Modern web arayüzü

🛠️ Kurulum

Projeyi klonlayın:
bash git clone https://github.com/fatihcldr/Spam-detector-project.git

cd Spam-detector-project

Gerekli paketleri yükleyin:
bashpip install -r requirements.txt

Uygulamayı çalıştırın:
bashpython main.py

Tarayıcıda açın: http://localhost:8000

🎮 Kullanım

Ana sayfaya gidin
"Dosya Seç" butonuyla .eml dosyanızı yükleyin
"Analiz Et" butonuna tıklayın
İki farklı algoritmanın sonuçlarını karşılaştırın

📊 Veri Seti
CSV dosyası, kişisel email koleksiyonu ve Kaggle platformundaki açık kaynak veri setlerinin birleştirilmiş halidir.
Format:

text: Email içeriği
label: 'spam' veya 'ham' etiketi

🗂️ Proje Yapısı
├── main.py              # Ana FastAPI uygulaması
├── requirements.txt     # Python bağımlılıkları
├── data/               # Veri dosyaları
├── static/             # CSS
└── templates/          # HTML şablonları

🔧 Teknik Detaylar
Backend: FastAPI
ML: scikit-learn, imbalanced-learn
Data Processing: pandas, numpy
Frontend: HTML/CSS


Bu proje eğitim amaçlı geliştirilmiştir.
