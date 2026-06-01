import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from grasp_main import run_instance

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
INST_DIR = "instancias"

INSTANCES = sorted([
    f for f in os.listdir(INST_DIR)
    if f.lower().endswith(".tsp")
])

OUTPUT_CSV = "grasp_results_sem_limite_p.csv"

# Quantas vezes rodar cada instância (recomendado >1 por causa do random do GRASP)
REPEATS = 1

# ==========================================================
# Função para rodar UMA execução (1 instância, 1 seed)
# ==========================================================
def run_one(args):
    instance_file, rep = args
    full_path = os.path.join(INST_DIR, instance_file)

    pid = os.getpid()
    print(f"[PID {pid}] {instance_file}  rep={rep}")

    # seed diferente por repetição (se você quiser controlar isso melhor, dá pra usar time ou hash)
    # OBS: seu grasp usa seed=123 fixo dentro do run_instance -> se quiser variabilidade real,
    # você precisa passar esse seed pra dentro (ou remover o fixo).
    result = run_instance(
        instance_file=full_path,
        p=0,                 # ignorado quando auto_parameters=True
        radius=0,            # ignorado quando auto_parameters=True
        max_iter=0,          # ignorado no seu GRASP
        plot=False,
        auto_parameters=True
    )

    # Guardar também o nome "limpo" e o rep
    result["instance_name"] = instance_file
    result["rep"] = rep

    return result


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================
if __name__ == "__main__":
    n_cpus = cpu_count()

    print("\n============================================")
    print(" Rodando GRASP em paralelo (auto_parameters=True)")
    print(f" CPUs disponíveis: {n_cpus}")
    print("============================================\n")

    print(f"Instâncias encontradas: {len(INSTANCES)}")
    print(INSTANCES)
    print()

    # monta lista de tarefas: (instância, repetição)
    tasks = [(inst, rep) for inst in INSTANCES for rep in range(1, REPEATS + 1)]
    print(f"Total de execuções: {len(tasks)}\n")

    with Pool(processes=n_cpus) as pool:
        results = pool.map(run_one, tasks)

    # salva tudo em um único CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nArquivo salvo: {OUTPUT_CSV}")
    print("\n===== TODAS AS INSTÂNCIAS FORAM PROCESSADAS =====\n")