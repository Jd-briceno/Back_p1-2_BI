from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Ajusta según la URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = load("modelo_clasificacion_texto.joblib")

class TextInput(BaseModel):
    Textos_espanol: str

class PredictionInput(BaseModel):
    texts: List[TextInput]

class TrainingInput(BaseModel):
    texts: List[TextInput]
    labels: List[int]

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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain(input: TrainingInput):
    try:
        texts = [item.Textos_espanol for item in input.texts]
        
        # Reentrenar el modelo
        model.fit(texts, input.labels)
        
        # Guardar el modelo reentrenado
        dump(model, "modelo_clasificacion_texto.joblib")
        
        # Evaluar el modelo reentrenado
        predictions = model.predict(texts)
        f1 = f1_score(input.labels, predictions, average='weighted')
        precision = precision_score(input.labels, predictions, average='weighted')
        recall = recall_score(input.labels, predictions, average='weighted')
        
        return {
            "message": "Modelo reentrenado exitosamente",
            "metrics": {
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)