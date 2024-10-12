Juan David Briceño - 201812887
Daniel Clavijo - 202122209

# Backend para Clasificación de Texto

Este es el backend para la aplicación de clasificación de texto.

## Configuración

1. Asegúrate de tener Python 3.7+ instalado.

2. Crea un entorno virtual:
   
   python -m venv venv
   

3. Activa el entorno virtual:
   - En Windows:
     
     venv\Scripts\activate
     
   - En macOS y Linux:
     
     source venv/bin/activate
     

4. Instala las dependencias:
   
   pip install -r requirements.txt
   

## Ejecución

Para iniciar el servidor:


uvicorn main:app --reload


El servidor estará disponible en `http://localhost:8000`.

## Endpoints

- POST `/predict`: Para hacer predicciones
- POST `/retrain`: Para reentrenar el modelo