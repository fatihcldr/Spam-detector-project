from fastapi import FastAPI, Request, File, UploadFile, HTTPException,Form
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import joblib
from email import message_from_bytes
import os
from pathlib import Path

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
model2 = None
tfidf_vectorizer = None


def train_models():
    # Veri yükleme
    
    current_dir = Path(__file__).parent
    csv_file = current_dir / "data" / "cleaned_combined_data.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    def preprocess_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = ' '.join(text.split())
            return text
        return ''
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Veriyi ayırma kısmı
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['label'].map({'ham': 0, 'spam': 1}),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # TF-IDF ile verileri vektörize ettim
    global tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Verileri dönüştürme
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # SMOTE ile dengeleme
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
    
    # Model eğitme
    global model, model2
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_balanced, y_train_balanced)
    
    model2 = SVC(kernel='linear', probability=True)
    model2.fit(X_train_balanced, y_train_balanced)


@app.on_event("startup")
async def startup_event():
    train_models()

def analyze_text(text: str):
    # Metin ön işleme
    text = text.lower()
    text = ' '.join(text.split())
    
    # Vektörize ettigimiz yer
    text_tfidf = tfidf_vectorizer.transform([text])
    
    # Tahmin
    prediction_NB = model.predict(text_tfidf)[0]
    prediction_SVC = model2.predict(text_tfidf)[0]
    probability_NB = model.predict_proba(text_tfidf)[0]
    probability_SVC = model2.predict_proba(text_tfidf)[0]
    
    return prediction_NB, prediction_SVC, probability_NB, probability_SVC

@app.get("/", response_class=HTMLResponse)
async def index(request: Request ):
    return templates.TemplateResponse("index.html", {"request": request,})

@app.post ("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.eml'):
        return templates.TemplateResponse("errors.html", {
            "request": request,
            "error_message": "Lütfen sadece .eml uzantılı dosyalar yükleyin."
        })
    
    try:
        content = await file.read()
        content = content.decode('utf-8', errors='ignore')
        prediction_NB, prediction_SVC, probabilities_NB, probabilities_SVC = analyze_text(content)
        filename = file.filename
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "filename": filename,
            "prediction_NB": prediction_NB,
            "prediction_SVC": prediction_SVC,
            "spam_probability": probabilities_NB[1] * 100,
            "ham_probability": probabilities_NB[0] * 100,
            "spam_probability2": probabilities_SVC[1] * 100,
            "ham_probability2": probabilities_SVC[0] * 100
        })
        
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": f"Hata: {str(e)}"
        })

@app.post ("/feedback") #kullanicidan geri bildirim almak icin
async def feedback(request: Request,feedback : str = Form (...)):
    feedback = feedback
    return templates.TemplateResponse ("feedback.html",{
        
        "request" : request,
        "feedback": feedback
    })
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
