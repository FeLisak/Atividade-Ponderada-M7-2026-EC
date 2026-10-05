import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


dados = pd.read_csv("data/btc_daily.csv", parse_dates=["date"])
livros = pd.read_json("data/book_releases.json")
dados["book_release"] = dados["date"].isin(pd.to_datetime(livros["date"])).astype(int)
dados["next_close"] = dados["close"].shift(-1)
dados["next_date"] = dados["date"].shift(-1)
dados = dados.dropna()

corte = int(len(dados) * 0.8)
treino = dados.iloc[:corte]
teste = dados.iloc[corte:]

modelo = LinearRegression()
modelo.fit(treino[["open", "close", "book_release"]], treino["next_close"])
modelo_sem_livros = LinearRegression()
modelo_sem_livros.fit(treino[["open", "close"]], treino["next_close"])

previsoes = modelo.predict(teste[["open", "close", "book_release"]])
previsoes_sem_livros = modelo_sem_livros.predict(teste[["open", "close"]])
metricas = {
    "train_samples": len(treino),
    "test_samples": len(teste),
    "train_target_start": treino["next_date"].iloc[0].strftime("%Y-%m-%d"),
    "train_target_end": treino["next_date"].iloc[-1].strftime("%Y-%m-%d"),
    "test_target_start": teste["next_date"].iloc[0].strftime("%Y-%m-%d"),
    "test_target_end": teste["next_date"].iloc[-1].strftime("%Y-%m-%d"),
    "mae_usd": {
        "with_releases": mean_absolute_error(teste["next_close"], previsoes),
        "prices_only": mean_absolute_error(teste["next_close"], previsoes_sem_livros),
        "last_close": mean_absolute_error(teste["next_close"], teste["close"]),
    },
    "rmse_usd": {
        "with_releases": root_mean_squared_error(teste["next_close"], previsoes),
        "prices_only": root_mean_squared_error(teste["next_close"], previsoes_sem_livros),
        "last_close": root_mean_squared_error(teste["next_close"], teste["close"]),
    },
    "r2": {
        "with_releases": r2_score(teste["next_close"], previsoes),
        "prices_only": r2_score(teste["next_close"], previsoes_sem_livros),
        "last_close": r2_score(teste["next_close"], teste["close"]),
    },
}

Path("artifacts").mkdir(exist_ok=True)
dados.to_csv("artifacts/dataset.csv", index=False)
treino.to_csv("artifacts/train.csv", index=False)
teste.to_csv("artifacts/test.csv", index=False)
joblib.dump(modelo, "artifacts/model.joblib")
joblib.dump(modelo_sem_livros, "artifacts/baseline.joblib")
with open("artifacts/metrics.json", "w") as arquivo:
    json.dump(metricas, arquivo, indent=2)

teste[["next_date", "next_close"]].assign(
    with_releases=previsoes, prices_only=previsoes_sem_livros, last_close=teste["close"]
).rename(columns={"next_date": "date", "next_close": "actual_close"}).to_csv("artifacts/test_predictions.csv", index=False)

exemplo = dados.loc[dados["date"] == "2024-10-22", ["open", "close", "book_release"]].iloc[0].to_dict()
with open("artifacts/sample_request.json", "w") as arquivo:
    json.dump(exemplo, arquivo, indent=2)

comparacao = teste["next_date"] >= "2024-05-28"
registro = pd.DataFrame([{
    "version": "Linear, 1 dia",
    "test_start": "2024-05-28",
    "test_end": "2024-12-31",
    "samples": int(comparacao.sum()),
    "with_releases": mean_absolute_error(teste.loc[comparacao, "next_close"], previsoes[comparacao]),
    "prices_only": mean_absolute_error(teste.loc[comparacao, "next_close"], previsoes_sem_livros[comparacao]),
    "last_close": mean_absolute_error(teste.loc[comparacao, "next_close"], teste.loc[comparacao, "close"]),
}])
historico = Path("artifacts/training_history.csv")
if historico.exists():
    registro = pd.concat([pd.read_csv(historico), registro], ignore_index=True)
registro.to_csv(historico, index=False)

print(json.dumps(metricas, indent=2))
