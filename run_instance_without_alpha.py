import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
from main import run_instance


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
INSTANCE_FILE = "kroA100.tsp"

p_values = [4, 6, 8]
radius_values = [600, 700, 800]
max_iterations = 30

OUTPUT_CSV = "kroA100_vns_coverage_only.csv"


# ==========================================================
# Função para rodar UMA combinação
# ==========================================================
def run_combo(args):
    instance_file, p, radius = args

    print(f"[PID {os.getpid()}] {instance_file}   p={p}, R={radius}")

    result = run_instance(
        instance_file=instance_file,
        p=p,
        radius=radius,
        max_iter=max_iterations,
        plot=False
    )

    return result


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================
if __name__ == "__main__":
    n_cpus = cpu_count()

    print("\n============================================")
    print(" Rodando VNS (coverage-first) em paralelo")
    print(f" Instância: {INSTANCE_FILE}")
    print(f" CPUs disponíveis: {n_cpus}")
    print("============================================\n")

    # gera todas as combinações (9 no seu caso: 3 p × 3 radius)
    combos = list(itertools.product(
        [INSTANCE_FILE],
        p_values,
        radius_values
    ))

    print(f"Total de execuções: {len(combos)}\n")

    # pool de processos
    with Pool(processes=n_cpus) as pool:
        results = pool.map(run_combo, combos)

    # salva CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n============================================")
    print(f" Arquivo salvo: {OUTPUT_CSV}")
    print("============================================\n")
