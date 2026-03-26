import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from one_swap_main import run_instance

INST_DIR = "instancias"

INSTANCES = sorted([
    f for f in os.listdir(INST_DIR)
    if f.lower().endswith(".tsp")
])

OUTPUT_CSV = "autor_aceitacao_results.csv"
REPEATS = 1

# escolha um max_iter real
MAX_ITER = 100  # ou 10**9 se você quer só por tempo (600s)

def run_one(args):
    instance_file, rep = args
    full_path = os.path.join(INST_DIR, instance_file)

    pid = os.getpid()
    print(f"[PID {pid}] {instance_file}  rep={rep}")

    result = run_instance(
        instance_file=full_path,
        p=0,
        radius=0,
        max_iter=MAX_ITER,
        plot=False,
        auto_parameters=True
    )

    result["instance_name"] = instance_file
    result["rep"] = rep
    return result

if __name__ == "__main__":
    n_cpus = cpu_count()

    print("\n============================================")
    print(" Rodando Codigo do autor com criterio de aceitacao em paralelo (auto_parameters=True)")
    print(f" CPUs disponíveis: {n_cpus}")
    print("============================================\n")

    tasks = [(inst, rep) for inst in INSTANCES for rep in range(1, REPEATS + 1)]
    print(f"Total de execuções: {len(tasks)}\n")

    with Pool(processes=n_cpus) as pool:
        results = pool.map(run_one, tasks)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nArquivo salvo: {OUTPUT_CSV}")
    print("\n===== TODAS AS INSTÂNCIAS FORAM PROCESSADAS =====\n")