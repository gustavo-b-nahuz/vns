import json, random, time, pandas as pd
import networkx as nx
from pathlib import Path
from main import (      #  ← todas as funções já existentes
    read_tsplib_instance,
    initialize_solution,
    tsp_nearest_insertion,
    calculate_coverage,
    objective,
    local_search,
    load_params)

def run_instance(tsp_file: str, params: dict):
    """Roda o algoritmo completo para 1 instância e devolve métricas."""
    n, edges = read_tsplib_instance(tsp_file)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_weighted_edges_from(edges)

    p       = params["p"]
    radius  = params["coverage_radius"]
    alpha   = params["alpha"]
    k_min, k_max = 1, p
    max_iter     = params["max_iterations"]

    # --- solução inicial ---------------
    sol = initialize_solution(g, n, p, radius)
    best_sol = sol[:]
    best_tour, best_dist = tsp_nearest_insertion(g, best_sol)
    best_cov  = calculate_coverage(g, best_sol, radius)
    best_obj  = objective(alpha, best_dist, best_cov)

    # --- ciclo VNS ---------------------
    t0 = time.time()
    for it in range(1, max_iter + 1):
        k = k_min
        while k <= k_max:
            pert = best_sol[:]
            for _ in range(k):
                v_rm = random.choice(pert)
                pert.remove(v_rm)
                cand  = [v for v in range(n) if v not in pert]
                pert.append(random.choice(cand))

            pert_tour, pert_dist = tsp_nearest_insertion(g, pert)
            pert_cov = calculate_coverage(g, pert, radius)
            pert_obj = objective(alpha, pert_dist, pert_cov)

            pert_sol, pert_tour, pert_dist, pert_cov, pert_obj = \
                local_search(g, pert, radius, alpha)

            if pert_obj < best_obj:
                best_sol, best_tour  = pert_sol[:], pert_tour[:]
                best_dist, best_cov, best_obj = pert_dist, pert_cov, pert_obj
                k = k_min
            else:
                k += 1
    elapsed = time.time() - t0

    return {
        "instance"  : Path(tsp_file).stem,
        "n"         : n,
        "p"         : p,
        "radius"    : radius,
        "obj"       : round(best_obj, 2),
        "dist"      : round(best_dist, 1),
        "covered"   : best_cov,
        "cov_pct"   : f"{100*best_cov/n:.1f}%",
        "time_s"    : round(elapsed, 2)
    }


if __name__ == "__main__":
    # --------- parâmetros globais -----------------
    params = load_params("params.json")
    tsp_files = [
        "kroA100.tsp", "kroA200.tsp", "kroB100.tsp",
        "kroB200.tsp", "kroC100.tsp", "kroD100.tsp"
    ]

    results = [run_instance(f, params) for f in tsp_files]

    df = pd.DataFrame(results,
        columns=["instance","n","p","radius",
                 "obj","dist","covered","cov_pct","time_s"])
    print("\n==== Resultados ====\n")
    print(df.to_string(index=False))
    df.to_csv("resumo_resultados.csv", index=False)
