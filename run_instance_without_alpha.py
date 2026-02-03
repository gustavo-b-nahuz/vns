import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
from main import run_instance


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
INSTANCES = [
    "kroA100.tsp",
    "kroA200.tsp",
    "kroB100.tsp",
    "kroB200.tsp",
    "kroC100.tsp",
    "kroD100.tsp",
]

p_values = [4, 6, 8]
radius_values = [600, 700, 800]
max_iterations = 30


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
    print(f" CPUs disponíveis: {n_cpus}")
    print("============================================\n")

    for instance in INSTANCES:
        print(f"\n>>> Rodando instância: {instance}\n")

        # gera combinações (p × radius)
        combos = list(itertools.product(
            [instance],
            p_values,
            radius_values
        ))

        print(f"Total de execuções: {len(combos)}")

        with Pool(processes=n_cpus) as pool:
            results = pool.map(run_combo, combos)

        # salva CSV da instância
        inst_name = os.path.splitext(instance)[0]
        output_csv = f"{inst_name}_vns_coverage_only.csv"

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        print(f"\nArquivo salvo: {output_csv}")
        print("--------------------------------------------")

    print("\n===== TODAS AS INSTÂNCIAS FORAM PROCESSADAS =====\n")
