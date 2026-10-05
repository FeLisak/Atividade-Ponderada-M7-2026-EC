import joblib
import pandas as pd
import uvicorn
from fastapi import Body, FastAPI


app = FastAPI()
modelo = joblib.load("artifacts/model.joblib")
modelo_sem_livros = joblib.load("artifacts/baseline.joblib")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    open: float = Body(gt=0, allow_inf_nan=False),
    close: float = Body(gt=0, allow_inf_nan=False),
    book_release: int = Body(ge=0, le=1),
):
    entrada = pd.DataFrame([{"open": open, "close": close, "book_release": book_release}])
    previsao = float(modelo.predict(entrada)[0])
    previsao_sem_livros = float(modelo_sem_livros.predict(entrada[["open", "close"]])[0])
    return {
        "with_releases": round(previsao, 2),
        "prices_only": round(previsao_sem_livros, 2),
        "difference_usd": round(previsao - previsao_sem_livros, 2),
        "horizon_days": 1,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
