import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


url = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_d.csv"
Path("data").mkdir(exist_ok=True)
original = Path("data/Bitstamp_BTCUSD_d.csv")
if not original.exists():
    urlretrieve(url, original)
dados = pd.read_csv(original, skiprows=1)
dados["date"] = pd.to_datetime(dados["date"].str[:10])
dados = dados[dados["date"].between("2022-01-01", "2024-12-31")]
if not dados["symbol"].eq("BTC/USD").all():
    raise ValueError("O arquivo deve conter preços de BTC/USD.")
dados = dados[["date", "open", "close"]].sort_values("date")

if not pd.DatetimeIndex(dados["date"]).equals(pd.date_range("2022-01-01", "2024-12-31")):
    raise ValueError("O período deve estar completo, sem datas repetidas.")
if not np.isfinite(dados[["open", "close"]]).all().all() or (dados[["open", "close"]] <= 0).any().any():
    raise ValueError("Os preços devem ser positivos e finitos.")

Path("data").mkdir(exist_ok=True)
dados.to_csv("data/btc_daily.csv", index=False, date_format="%Y-%m-%d")
with open("data/source.json", "w") as arquivo:
    json.dump({"source": url, "start": "2022-01-01", "end": "2024-12-31", "rows": len(dados)}, arquivo, indent=2)

print(f"Salvei {len(dados)} dias em data/btc_daily.csv")
