import json
from pathlib import Path
from urllib.request import Request, urlopen


with urlopen("http://localhost:8000/health", timeout=10) as resposta:
    print("Backend:", json.load(resposta)["status"])

entrada = json.loads(Path("artifacts/sample_request.json").read_text())
resultados = {}
for lancamento in [1, 0]:
    entrada["book_release"] = lancamento
    pedido = Request(
        "http://localhost:8000/predict",
        data=json.dumps(entrada).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(pedido, timeout=10) as resposta:
        resultado = json.load(resposta)
    resultados[str(lancamento)] = resultado
    print(f"Lançamento = {lancamento}:", json.dumps(resultado, ensure_ascii=False))

Path("artifacts/client_response.json").write_text(json.dumps(resultados, indent=2))
