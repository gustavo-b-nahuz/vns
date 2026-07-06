import os
import pandas as pd
from multiprocessing import Pool, cpu_count

from one_swap_main import run_instance as run_one_swap
from vns_vnd_main import run_instance as run_vns_vnd
from grasp_main import run_instance as run_grasp


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

INST_DIR = "instancias"

OUTPUT_FINAL = "resultados_finais_multiseed.csv"
OUTPUT_HISTORY = "historico_multiseed.csv"

# Primeiro teste com [0, 1].
# Depois, quando estiver tudo certo, troque para list(range(10)).
SEEDS = [0, 1]

ALGORITHMS = [
    {
        "name": "Lexicografico 1-swap",
        "runner": run_one_swap,
        "max_iter": 100
    },
    {
        "name": "VNS com VND",
        "runner": run_vns_vnd,
        "max_iter": 100
    },
    {
        "name": "GRASP com VND",
        "runner": run_grasp,
        "max_iter": 1000
    }
]


def list_instances():
    return sorted([
        f for f in os.listdir(INST_DIR)
        if f.lower().endswith(".tsp")
    ])


def run_task(task):
    instance_name, algorithm_name, max_iter, seed = task

    full_path = os.path.join(INST_DIR, instance_name)

    if algorithm_name == "Lexicografico 1-swap":
        runner = run_one_swap
    elif algorithm_name == "VNS com VND":
        runner = run_vns_vnd
    elif algorithm_name == "GRASP com VND":
        runner = run_grasp
    else:
        raise ValueError(f"Abordagem desconhecida: {algorithm_name}")

    print(f"[RUN] {instance_name} | {algorithm_name} | seed={seed}")

    try:
        result = runner(
            instance_file=full_path,
            p=0,
            radius=0,
            max_iter=max_iter,
            plot=False,
            auto_parameters=True,
            seed=seed,
            return_history=True
        )

        history = result.pop("history", [])

        final_row = {
            "instancia": instance_name,
            "abordagem": algorithm_name,
            "semente": seed,
            "n_vertices": result.get("n_vertices"),
            "p": result.get("p"),
            "r": result.get("radius"),
            "iteracoes_planejadas": result.get("iterations_requested"),
            "iteracoes_executadas": result.get("iterations_executed"),
            "cobertura_final": result.get("final_cov"),
            "distancia_final": result.get("final_dist"),
            "tempo_execucao": result.get("total_time"),
            "iter_best_found": result.get("iter_best_found"),
            "time_best_found": result.get("time_best_found"),
            "erro": ""
        }

        history_rows = []

        for h in history:
            history_rows.append({
                "instancia": instance_name,
                "abordagem": algorithm_name,
                "semente": seed,
                "n_vertices": result.get("n_vertices"),
                "p": result.get("p"),
                "r": result.get("radius"),
                "iteracao": h.get("iteracao"),
                "tempo_acumulado": h.get("tempo_acumulado"),
                "melhor_cobertura": h.get("melhor_cobertura"),
                "melhor_distancia": h.get("melhor_distancia"),
            })

        return final_row, history_rows

    except Exception as e:
        final_row = {
            "instancia": instance_name,
            "abordagem": algorithm_name,
            "semente": seed,
            "n_vertices": None,
            "p": None,
            "r": None,
            "iteracoes_planejadas": max_iter,
            "iteracoes_executadas": None,
            "cobertura_final": None,
            "distancia_final": None,
            "tempo_execucao": None,
            "iter_best_found": None,
            "time_best_found": None,
            "erro": str(e)
        }

        return final_row, []


if __name__ == "__main__":
    instances = list_instances()

    tasks = []

    for inst in instances:
        for alg in ALGORITHMS:
            for seed in SEEDS:
                tasks.append((
                    inst,
                    alg["name"],
                    alg["max_iter"],
                    seed
                ))

    print("\n============================================")
    print(" Experimentos multiseed")
    print(f" Instâncias: {len(instances)}")
    print(f" Sementes: {SEEDS}")
    print(f" Total de execuções: {len(tasks)}")
    print("============================================\n")

    n_processes = max(1, min(cpu_count() - 1, 6))
    print(f"Processos usados: {n_processes}\n")

    final_rows = []
    history_rows = []

    with Pool(processes=n_processes) as pool:
        for final_row, hist in pool.imap_unordered(run_task, tasks):
            final_rows.append(final_row)
            history_rows.extend(hist)

            if final_row["erro"]:
                print(
                    f"[ERRO] {final_row['instancia']} | "
                    f"{final_row['abordagem']} | "
                    f"seed={final_row['semente']} -> {final_row['erro']}"
                )
            else:
                print(
                    f"[OK] {final_row['instancia']} | "
                    f"{final_row['abordagem']} | "
                    f"seed={final_row['semente']} | "
                    f"cov={final_row['cobertura_final']} | "
                    f"dist={final_row['distancia_final']}"
                )

    df_final = pd.DataFrame(final_rows)
    df_history = pd.DataFrame(history_rows)

    df_final.sort_values(
        by=["instancia", "abordagem", "semente"],
        inplace=True
    )

    if not df_history.empty:
        df_history.sort_values(
            by=["instancia", "abordagem", "semente", "iteracao"],
            inplace=True
        )

    df_final.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8-sig")
    df_history.to_csv(OUTPUT_HISTORY, index=False, encoding="utf-8-sig")

    print("\n============================================")
    print(f"Arquivo salvo: {OUTPUT_FINAL}")
    print(f"Arquivo salvo: {OUTPUT_HISTORY}")
    print("============================================\n")