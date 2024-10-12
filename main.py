from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Función para cargar y entrenar el modelo con data_train.xlsx
def load_and_train_initial_model():
    try:
        df = pd.read_excel('data_train.xlsx')
        texts = df['Textos_espanol'].tolist()
        labels = df['sdg'].tolist()
        
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        
        model.fit(texts, labels)
        dump(model, "modelo_clasificacion_texto.joblib")
        print("Modelo inicial entrenado y guardado con éxito.")
        return model
    except Exception as e:
        print(f"Error al cargar y entrenar el modelo inicial: {e}")
        return None

# Cargar o entrenar el modelo inicial
model = load_and_train_initial_model()

class TextInput(BaseModel):
    Textos_espanol: str

class PredictionInput(BaseModel):
    texts: List[TextInput]

@app.post("/predict")
async def predict(input: PredictionInput):
    try:
        texts = [item.Textos_espanol for item in input.texts]
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts).max(axis=1)
        
        return [
            {"prediction": int(pred), "probability": float(prob)}
            for pred, prob in zip(predictions, probabilities)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.post("/retrain_replace")
async def retrain_replace(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        texts = df['Textos_espanol'].tolist()
        labels = df['sdg'].tolist()
        
        global model
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        model.fit(texts, labels)
        dump(model, "modelo_clasificacion_texto.joblib")
        
        return {"message": "Modelo reentrenado y reemplazado con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el reentrenamiento: {str(e)}")

@app.post("/retrain_concatenate")
async def retrain_concatenate(file: UploadFile = File(...)):
    try:
        new_df = pd.read_excel(io.BytesIO(await file.read()))
        original_df = pd.read_excel('data_train.xlsx')
        
        combined_df = pd.concat([original_df, new_df], ignore_index=True)
        texts = combined_df['Textos_espanol'].tolist()
        labels = combined_df['sdg'].tolist()
        
        global model
        model.fit(texts, labels)
        dump(model, "modelo_clasificacion_texto.joblib")
        
        return {"message": "Modelo reentrenado con datos concatenados con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el reentrenamiento: {str(e)}")

@app.post("/retrain_weighted")
async def retrain_weighted(file: UploadFile = File(...)):
    try:
        new_df = pd.read_excel(io.BytesIO(await file.read()))
        original_df = pd.read_excel('data_train.xlsx')
        
        # Dar más peso a los nuevos datos
        new_df['weight'] = 2
        original_df['weight'] = 1
        
        combined_df = pd.concat([original_df, new_df], ignore_index=True)
        texts = combined_df['Textos_espanol'].tolist()
        labels = combined_df['sdg'].tolist()
        weights = combined_df['weight'].tolist()
        
        global model
        model.fit(texts, labels, clf__sample_weight=weights)
        dump(model, "modelo_clasificacion_texto.joblib")
        
        return {"message": "Modelo reentrenado con ponderación de datos con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el reentrenamiento: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)