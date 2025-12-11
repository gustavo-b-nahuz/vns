import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
from main import run_instance


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
instance_file = "kroA100.tsp"

p_values = [4, 6, 8]
radius_values = [600, 700, 800]
alpha_values = [0.001, 0.01, 0.1]

max_iterations = 50   # defina como quiser

inst_name = os.path.splitext(os.path.basename(instance_file))[0]
csv_path = f"{inst_name}_resultados.csv"


# ==========================================================
# FUNÇÃO AUXILIAR PARA RODAR UMA COMBINAÇÃO
# (necessária porque multiprocessing não aceita lambdas com closures complexas)
# ==========================================================
def run_combo(params):
    p, radius, alpha = params
    print(f"[PID] Rodando p={p}, R={radius}, alpha={alpha}")
    
    result = run_instance(
        instance_file=instance_file,
        p=p,
        radius=radius,
        max_iter=max_iterations,
        alpha=alpha,
        plot=False   # evita 27 plots abrindo
    )
    return result


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================
if __name__ == "__main__":
    combos = list(itertools.product(p_values, radius_values, alpha_values))
    n_processes = cpu_count()

    print("\n======================================")
    print(" Rodando as 27 combinações em paralelo")
    print(f" CPUs disponíveis: {n_processes}")
    print("======================================\n")

    # Pool de processos
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_combo, combos)

    # Cria DataFrame final
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    print("\n======================================")
    print(" Execução paralela concluída!")
    print(f" Resultados salvos em: {csv_path}")
    print("======================================")
