import os
import itertools
import pandas as pd
from main import run_instance   # importa sua função

# CONFIGURAÇÃO DOS PARÂMETROS
instance_file = "kroA100.tsp"

p_values = [4, 6, 8]
radius_values = [600, 700, 800]
alpha_values = [0.001, 0.01, 0.1]
max_iterations = 100   # ou o que você usar normalmente

# CRIAR PASTA BASEADA NA INSTÂNCIA
inst_name = os.path.splitext(os.path.basename(instance_file))[0]
output_dir = f"resultados_{inst_name}"
os.makedirs(output_dir, exist_ok=True)

# LISTA PARA GUARDAR TODOS OS RESULTADOS
results = []

# LOOP DAS 27 COMBINAÇÕES
for p, radius, alpha in itertools.product(p_values, radius_values, alpha_values):

    print(f"\n=== Rodando combinação p={p}, R={radius}, alpha={alpha} ===")

    res = run_instance(
        instance_file=instance_file,
        p=p,
        radius=radius,
        max_iter=max_iterations,
        alpha=alpha,
        plot=False   # não queremos 27 gráficos abrindo
    )

    results.append(res)

# SALVAR CSV FINAL
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, f"{inst_name}_resultados.csv")
df.to_csv(csv_path, index=False)

print("\n======================================")
print(" Execução concluída!")
print(f" CSV salvo em: {csv_path}")
print("======================================")
