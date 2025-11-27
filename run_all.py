import json
import subprocess
import itertools
import os

# Valores que você pediu
p_values = [4, 6, 8]
alpha_values = [0.001, 0.01, 0.1]
radius_values = [600, 700, 800]

# Arquivo de parâmetros (o que seu programa usa)
PARAM_FILE = "params.json"

# Onde salvar os logs de execução
os.makedirs("resultados", exist_ok=True)

# Gera todas as combinações cartesianas
experimentos = list(itertools.product(p_values, alpha_values, radius_values))

print(f"Rodando {len(experimentos)} experimentos...\n")

for p, alpha, radius in experimentos:
    # Monta o conteúdo do params.json
    params = {
        "instance_file": "kroA100.tsp",
        "plot_graph": False,
        "p": p,
        "coverage_radius": radius,
        "max_iterations": 10,
        "alpha": alpha
    }

    # Salva o params.json antes da execução
    with open(PARAM_FILE, "w") as f:
        json.dump(params, f, indent=4)

    # Nome do arquivo de saída
    out_name = f"resultados/p{p}_a{alpha}_r{radius}.txt"

    print(f"Executando p={p}, alpha={alpha}, raio={radius}...")

    # Chama seu programa principal
    # Caso seu arquivo principal tenha outro nome, altere "main.py"
    with open(out_name, "w") as outfile:
        subprocess.run(["python3", "main.py"], stdout=outfile, stderr=outfile)

print("\nExperimentos concluídos! Resultados em ./resultados/")
