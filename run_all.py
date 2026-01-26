import os
import itertools
import pandas as pd
from main import run_instance   # importa sua função

# Usar all_intances.py
# CONFIGURAÇÃO DOS PARÂMETROS
instance_file = "kroA100.tsp"

p_values = [4, 6, 8]
radius_values = [600, 700, 800]
alpha_values = [0.001, 0.01, 0.1]
max_iterations = 100   # pode ajustar

# NOME DO CSV FINAL
inst_name = os.path.splitext(os.path.basename(instance_file))[0]
csv_path = f"{inst_name}_resultados.csv"

results = []

print("\n======================================")
print(f" Rodando combinações para {instance_file}")
print("======================================\n")

# LOOP DAS 27 COMBINAÇÕES
for p, radius, alpha in itertools.product(p_values, radius_values, alpha_values):

    print(f"=== Rodando p={p}, R={radius}, alpha={alpha} ===")

    res = run_instance(
        instance_file=instance_file,
        p=p,
        radius=radius,
        max_iter=max_iterations,
        alpha=alpha,
        plot=False  # evitar abrir 27 gráficos
    )

    results.append(res)

# SALVAR CSV FINAL
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)

print("\n======================================")
print(" Execução concluída!")
print(f" CSV salvo como: {csv_path}")
print("======================================")
