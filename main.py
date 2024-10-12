from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

app = FastAPI()

# Cargar el modelo y el vectorizador
nb_model = joblib.load('naive_bayes_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

class TextData(BaseModel):
    Textos_espanol: List[str]

class RetrainData(BaseModel):
    Textos_espanol: List[str]
    sdg: List[int]

# Endpoint 1: Realizar predicciones y devolver probabilidades
@app.post("/predict/")
async def predict(data: TextData):
    try:
        preprocessed_texts = [preprocess_text(text) for text in data.Textos_espanol]
        X_tfidf = tfidf_vectorizer.transform(preprocessed_texts)
        predictions = nb_model.predict(X_tfidf)
        probabilities = nb_model.predict_proba(X_tfidf)

        predictions = predictions.tolist()  
        probabilities = probabilities.tolist() 

        response = [{"prediction": int(pred), "probability": prob} for pred, prob in zip(predictions, probabilities)]
        return {"predictions": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Re-entrenar el modelo con nuevos datos
@app.post("/retrain/")
async def retrain(data: RetrainData):
    try:
        # Preprocesar los textos
        preprocessed_texts = [preprocess_text(text) for text in data.Textos_espanol]
        X_tfidf = tfidf_vectorizer.transform(preprocessed_texts)

        # Combina las clases existentes y las nuevas
        unique_classes = np.unique(np.concatenate((data.sdg, nb_model.classes_)))  
        nb_model.partial_fit(X_tfidf, data.sdg, classes=unique_classes)

        # Evaluar el nuevo modelo
        y_pred = nb_model.predict(X_tfidf)
        precision = precision_score(data.sdg, y_pred, average='weighted')
        recall = recall_score(data.sdg, y_pred, average='weighted')
        f1 = f1_score(data.sdg, y_pred, average='weighted')

        # Guardar el nuevo modelo
        joblib.dump(nb_model, 'naive_bayes_model.joblib')
        
        # Devolver las métricas de rendimiento
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Función para preprocesar el texto
def preprocess_text(text):
    return text