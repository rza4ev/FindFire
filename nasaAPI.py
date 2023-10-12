from fastapi import FastAPI
import numpy as np
import pickle

# FastAPI uygulamasını başlatın
app = FastAPI()

# Eğitilmiş modeli yükle
with open("nasa2.pkl", "rb") as model_file:
    lr = pickle.load(model_file)

@app.get('/')
def index():
    return 'Prediction API'

@app.post("/predict/")
async def predict(data: dict):
    try:
        # JSON verisini bir liste içinde alın
        features = data.get("features", [])

        # Liste içindeki veriyi NumPy dizisine dönüştürün
        features_array = np.array(features).reshape(1, -1)
        pred = lr.predict(features_array)
        # Modeli kullanarak tahmin yapın
        prediction_prob = lr.predict_proba(features_array)
        probability_fire = prediction_prob[0][0] * 100  # Olasılığı buradan alın
        result = {"Class": pred.tolist()}  # Örneğin, tahmin sonucunu bir liste içinde JSON'a dönüştürebilirsiniz.
        if pred==0:
            return {"prediction_probability": 100-probability_fire,'Class':'Class 1'}
        # Tahmin sonucunu JSON olarak döndürün
        else:
            return {"prediction_probability": probability_fire,'Class':pred.tolist()}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
