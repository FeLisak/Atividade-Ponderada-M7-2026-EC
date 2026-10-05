import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


previsoes = pd.read_csv("artifacts/test_predictions.csv", parse_dates=["date"])
historico = pd.read_csv("artifacts/training_history.csv")
metricas = json.loads(Path("artifacts/metrics.json").read_text())
pasta = Path("artifacts/graficos")
pasta.mkdir(exist_ok=True)

plt.figure(figsize=(11, 5))
plt.plot(previsoes["date"], previsoes["actual_close"], label="Fechamento real", color="black")
plt.plot(previsoes["date"], previsoes["with_releases"], label="Com lançamentos", linestyle="--", zorder=3)
plt.plot(previsoes["date"], previsoes["prices_only"], label="Só com preços", alpha=0.7)
plt.xlim(previsoes["date"].min(), previsoes["date"].max())
plt.title("Fechamento real e previsões no teste")
plt.ylabel("Preço em USD")
plt.xlabel("Dia previsto")
plt.grid(alpha=0.2)
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(pasta / "previsoes.png", dpi=160)
plt.close()

nomes = ["Com lançamentos", "Só com preços", "Último fechamento"]
colunas = ["with_releases", "prices_only", "last_close"]
cores = ["#2563eb", "#f97316", "#64748b"]
figura, eixos = plt.subplots(1, 3, figsize=(15, 5))
for eixo, chave, titulo in zip(eixos, ["mae_usd", "rmse_usd", "r2"], ["MAE: menor é melhor", "RMSE: menor é melhor", "R²: maior é melhor"]):
    valores = [metricas[chave][coluna] for coluna in colunas]
    barras = eixo.bar(nomes, valores, color=cores)
    casas = 4 if chave == "r2" else 2
    eixo.bar_label(barras, labels=[f"{valor:.{casas}f}".replace(".", ",") for valor in valores], padding=5)
    eixo.set_ylim(min(0, min(valores) * 1.15), max(valores) * 1.15)
    eixo.set_title(titulo)
    eixo.set_ylabel("Sem unidade" if chave == "r2" else "USD")
    eixo.tick_params(axis="x", labelrotation=20)
    eixo.grid(axis="y", alpha=0.2)
figura.suptitle("Métricas da versão atual: 219 dias de teste")
figura.tight_layout()
figura.savefig(pasta / "erros.png", dpi=160)
plt.close(figura)

etapas = [f"{indice + 1}. {versao}" for indice, versao in enumerate(historico["version"])]
plt.figure(figsize=(11, 5))
for coluna, nome, cor in zip(colunas, nomes, cores):
    plt.plot(etapas, historico[coluna], marker="o", label=nome, color=cor)
plt.ylim(0, historico[colunas].max().max() * 1.15)
plt.title("Evolução dos treinamentos no mesmo período de teste")
plt.ylabel("Erro absoluto médio em USD")
plt.xlabel("Execução do treinamento")
plt.grid(alpha=0.2)
plt.legend()
plt.figtext(0.5, 0.01, "Comparação: 28/05/2024 a 31/12/2024, com 218 dias", ha="center")
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(pasta / "evolucao.png", dpi=160)
plt.close()

print("Gráficos salvos em artifacts/graficos")
