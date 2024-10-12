# Backend para Clasificación de Texto

Este es el backend para la aplicación de clasificación de texto.

## Autores
- Juan David Briceño - 201812887
- Daniel Clavijo - 202122209
- Carlos Medina - 202112046

## Requisitos previos

Asegúrate de tener Python 3.7+ instalado en tu sistema.

## Configuración

1. Clona este repositorio o descarga los archivos del backend.

2. Navega al directorio del proyecto en tu terminal.

3. (Opcional pero recomendado) Crea un entorno virtual:
   ```
   python -m venv venv
   ```

4. Activa el entorno virtual:
   - En Windows:
     ```
     venv\Scripts\activate
     ```
   - En macOS y Linux:
     ```
     source venv/bin/activate
     ```

5. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Ejecución

Para iniciar el servidor:

```
uvicorn main:app --reload
```

El servidor estará disponible en `http://localhost:8000`.

## Endpoints

- POST `/predict`: Para hacer predicciones
- POST `/retrain_replace`: Para reentrenar el modelo reemplazando los datos anteriores
- POST `/retrain_concatenate`: Para reentrenar el modelo concatenando nuevos datos
- POST `/retrain_weighted`: Para reentrenar el modelo con ponderación de nuevos datos

## Solución de problemas

Si encuentras algún error relacionado con la falta de alguna librería, asegúrate de que todas las dependencias se hayan instalado correctamente. Si el problema persiste, puedes intentar instalar manualmente las librerías faltantes usando pip:

```
pip install openpyxl 
python-multipart
```

## Notas adicionales

- Asegúrate de que el archivo `data_train.xlsx` esté en el mismo directorio que `main.py` para el entrenamiento inicial del modelo.