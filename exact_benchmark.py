import itertools
import math
import time
import os
import csv
from multiprocessing import Pool, cpu_count
import networkx as nx


# ==========================================================
# UTILIDADES
# ==========================================================
def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def read_tsplib_instance(filename):
    with open(filename, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and ln.strip() != "EOF"]

    idx = lines.index("NODE_COORD_SECTION") + 1

    coords = []
    for ln in lines[idx:]:
        parts = ln.split()
        _, x, y = parts[:3]
        coords.append((float(x), float(y)))

    n = len(coords)
    edges = []
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = euclidean_distance(xi, yi, xj, yj)
            edges.append((i, j, d))
            edges.append((j, i, d))

    return n, edges, coords


def calculate_coverage(graph, solution, radius):
    covered = set(solution)
    for v in solution:
        covered.update(
            w for w in graph.neighbors(v)
            if graph[v][w]["weight"] <= radius
        )
    return len(covered)


def tsp_exact(graph, nodes):
    if len(nodes) <= 1:
        return nodes[:], 0.0

    start = nodes[0]
    others = nodes[1:]

    best_dist = float("inf")
    best_tour = None

    for perm in itertools.permutations(others):
        tour = [start] + list(perm)
        dist = 0.0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            dist += graph[u][v]["weight"]

        if dist < best_dist:
            best_dist = dist
            best_tour = tour[:]

    return best_tour, best_dist


# ==========================================================
# WORKER PARALELO
# ==========================================================
def exact_worker(args):
    graph, combs, radius = args

    best_cov = -1
    best_dist = float("inf")
    best_sol = None

    for comb in combs:
        sol = list(comb)
        cov = calculate_coverage(graph, sol, radius)

        if cov < best_cov:
            continue

        _, dist = tsp_exact(graph, sol)

        if cov > best_cov or (cov == best_cov and dist < best_dist):
            best_cov = cov
            best_dist = dist
            best_sol = sol

    return best_cov, best_dist, best_sol


# ==========================================================
# SOLUÇÃO EXATA PARA UMA INSTÂNCIA + RAIO
# ==========================================================
def exact_solution_parallel(instance_file, p, radius, print_interval=15):
    n, edges, coords = read_tsplib_instance(instance_file)

    g = nx.Graph()
    for i, (x, y) in enumerate(coords):
        g.add_node(i, pos=(x, y))
    g.add_weighted_edges_from(edges)

    all_combs = list(itertools.combinations(range(n), p))
    total = len(all_combs)

    n_proc = cpu_count()
    chunk_size = math.ceil(total / n_proc)

    chunks = [
        all_combs[i:i + chunk_size]
        for i in range(0, total, chunk_size)
    ]

    print(f"\n[{instance_file} | R={radius}] combinações: {total}")
    print(f"[{instance_file} | R={radius}] processos: {n_proc}")

    start = time.time()
    last_print = start

    best_cov = -1
    best_dist = float("inf")
    best_sol = None

    finished = 0

    with Pool(processes=n_proc) as pool:
        results = pool.imap_unordered(
            exact_worker,
            [(g, chunk, radius) for chunk in chunks]
        )

        for cov, dist, sol in results:
            finished += 1

            if cov > best_cov or (cov == best_cov and dist < best_dist):
                best_cov = cov
                best_dist = dist
                best_sol = sol

                elapsed = time.time() - start
                print(
                    f"[{instance_file} | R={radius}] NOVO MELHOR  "
                    f"cov={best_cov}  dist={best_dist:.2f}  "
                    f"time={elapsed:.1f}s"
                )

            if time.time() - last_print > print_interval:
                perc = 100 * finished / len(chunks)
                elapsed = time.time() - start
                print(
                    f"[{instance_file} | R={radius}] progresso: {perc:.1f}% "
                    f"tempo={elapsed:.1f}s"
                )
                last_print = time.time()

    elapsed = time.time() - start

    return {
        "instance": instance_file,
        "p": p,
        "radius": radius,
        "best_cov": best_cov,
        "best_dist": best_dist,
        "time": elapsed,
    }


# ==========================================================
# EXECUÇÃO PARA kroA100 ... kroD100
# ==========================================================
if __name__ == "__main__":

    INSTANCES = [
        "kroA100.tsp",
        "kroB100.tsp",
        "kroC100.tsp",
        "kroD100.tsp",
    ]

    RADII = [600, 700, 800]
    p = 4

    for inst in INSTANCES:
        print(f"\n==============================")
        print(f"Rodando instância {inst}")
        print(f"==============================")

        csv_name = inst.replace(".tsp", "_exact.csv")

        with open(csv_name, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "instance",
                    "p",
                    "radius",
                    "best_cov",
                    "best_dist",
                    "time",
                ]
            )
            writer.writeheader()

            for radius in RADII:
                result = exact_solution_parallel(
                    instance_file=inst,
                    p=p,
                    radius=radius,
                    print_interval=20
                )

                writer.writerow(result)

        print(f"\nArquivo salvo: {csv_name}")

    print("\n=== TODAS AS INSTÂNCIAS E RAIOS PROCESSADOS ===")
