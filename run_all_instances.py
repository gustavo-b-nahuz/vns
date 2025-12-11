import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
from main import run_instance


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
p_values = [4, 6, 8]
radius_values = [600, 700, 800]
alpha_values = [0.001, 0.01, 0.1]
max_iterations = 10

# Pasta onde estão as instâncias .tsp
INSTANCE_DIR = "."
# Filtra apenas arquivos .tsp
instances = [f for f in os.listdir(INSTANCE_DIR) if f.lower().endswith(".tsp")]


# ==========================================================
# Função para rodar UMA INSTÂNCIA com UMA COMBINAÇÃO
# ==========================================================
def run_combo(args):
    instance_file, p, radius, alpha = args

    print(f"[PID] {instance_file}   p={p}, R={radius}, alpha={alpha}")

    result = run_instance(
        instance_file=instance_file,
        p=p,
        radius=radius,
        max_iter=max_iterations,
        alpha=alpha,
        plot=False
    )

    return result


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================
if __name__ == "__main__":
    n_cpus = cpu_count()

    print("\n============================================")
    print(f" Rodando todas as instâncias em paralelo")
    print(f" CPUs disponíveis: {n_cpus}")
    print("============================================\n")

    for inst in instances:
        print(f"\n >>> Rodando instância: {inst}\n")

        inst_name = os.path.splitext(inst)[0]
        csv_path = f"{inst_name}_resultados.csv"

        # gera as 27 combinações
        combos = list(itertools.product([inst], p_values, radius_values, alpha_values))

        # cria pool de processos
        with Pool(processes=n_cpus) as pool:
            results = pool.map(run_combo, combos)

        # salva CSV da instância
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

        print(f"\nArquivo salvo: {csv_path}")
        print("-------------------------------------------------------")

    print("\n===== TODAS AS INSTÂNCIAS FORAM PROCESSADAS =====\n")
