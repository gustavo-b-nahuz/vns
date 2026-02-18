import random, math, json, time
import networkx as nx
import matplotlib.pyplot as plt
import os
import itertools
from matplotlib.patches import Circle
os.makedirs("frames", exist_ok=True)

def draw_iteration(graph, picked, covered, tour_edges, iteration):
    # usar posições reais armazenadas no grafo
    pos = {node: graph.nodes[node]["pos"] for node in graph.nodes}

    plt.figure(figsize=(6, 5))

    # nós selecionados (vermelho)
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=list(picked),
        node_color="red",
        node_size=120
    )

    # nós cobertos mas não selecionados (cinza)
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=[v for v in covered if v not in picked],
        node_color="gray",
        node_size=80
    )

    # nós ainda não cobertos (azul-claro)
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=[v for v in graph.nodes if v not in covered],
        node_color="lightblue",
        node_size=80
    )

    # arestas do tour (verde grosso)
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=tour_edges,
        width=2.0,
        edge_color="green"
    )

    # desenhar rótulos
    nx.draw_networkx_labels(graph, pos, font_size=7)

    plt.title(f"Greedy – iteração {iteration}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")  # importante para não distorcer a geometria real
    plt.tight_layout()
    plt.savefig(f"frames/g_{iteration:02d}.png", dpi=150)
    plt.close()


# ---------- utilidades -------------------------------------------------
def plot_graph(graph):
    """Desenha a instância no plano cartesiano usando coordenadas reais."""

    # Pega posições reais de cada nó
    pos = {node: graph.nodes[node]["pos"] for node in graph.nodes}

    plt.figure(figsize=(8, 6))

    # Desenha só pontos, sem arestas
    nx.draw_networkx_nodes(
        graph, pos,
        node_color="lightblue",
        node_size=120
    )

    # Labels opcionais
    nx.draw_networkx_labels(
        graph, pos,
        font_size=8
    )

    plt.title("Instância TSPLIB no Plano Cartesiano (EUC_2D)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


def tsp_exact(graph, nodes):
    """Resolve o TSP exato para um conjunto pequeno de nós (p <= 10)."""

    if len(nodes) <= 1:
        return nodes[:], 0.0

    best_tour = None
    best_dist = float("inf")

    # fixa o primeiro nó (TSP simétrico, podemos fixar início)
    start = nodes[0]
    others = nodes[1:]

    for perm in itertools.permutations(others):
        tour = [start] + list(perm)
        # calcular distância completa do ciclo
        dist = 0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            dist += graph[u][v]["weight"]

        if dist < best_dist:
            best_dist = dist
            best_tour = tour[:]

    return best_tour, best_dist


def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def read_tsplib_instance(filename):
    """
    Lê arquivo TSPLIB (.tsp) e devolve (num_vertices, edges).
    Assume EDGE_WEIGHT_TYPE = EUC_2D.
    """
    with open(filename, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and ln.strip() != "EOF"]

    try:
        idx = lines.index("NODE_COORD_SECTION") + 1
    except ValueError:
        raise ValueError("Arquivo TSPLIB sem NODE_COORD_SECTION")

    coords = []
    for ln in lines[idx:]:
        parts = ln.split()
        if len(parts) >= 3:
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


# ---------- TSP heurística (inserção mais próxima) ---------------------
def tsp_nearest_insertion_optimized(graph, nodes):
    if len(nodes) <= 1:
        return nodes[:], 0.0

    # -----------------------------
    # Inicialização
    # -----------------------------
    remaining = nodes[:]
    start = remaining.pop(0)
    tour = [start]

    # minDist[v] = menor distância de v até o tour atual
    minDist = {}
    for v in remaining:
        minDist[v] = graph[v][start]["weight"]

    # -----------------------------
    # Loop principal
    # -----------------------------
    while remaining:
        # 1) Escolher o nó mais próximo do tour (O(n))
        best_node = min(remaining, key=lambda v: minDist[v])
        remaining.remove(best_node)

        # Caso especial: tour com apenas 1 nó
        if len(tour) == 1:
            tour.append(best_node)
        else:
            # 2) Cheapest insertion (O(n))
            best_pos = None
            best_increase = float("inf")

            for i in range(len(tour)):
                u = tour[i]
                w = tour[(i + 1) % len(tour)]

                increase = (
                    graph[u][best_node]["weight"] +
                    graph[best_node][w]["weight"] -
                    graph[u][w]["weight"]
                )

                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1

            tour.insert(best_pos, best_node)

        # 3) Atualizar distâncias mínimas (O(n))
        for v in remaining:
            d = graph[v][best_node]["weight"]
            if d < minDist[v]:
                minDist[v] = d

    # -----------------------------
    # Cálculo do custo final
    # -----------------------------
    total_dist = sum(
        graph[tour[i]][tour[(i + 1) % len(tour)]]["weight"]
        for i in range(len(tour))
    )

    return tour, total_dist


# ---------- construção inicial -----------------------------------------
def initialize_solution(graph, n, p, radius):
    covered = set()
    solution = []

    print("\n=== INICIALIZAÇÃO GULOSA ===")

    for it in range(p):
        t0 = time.time()

        best_node = None
        best_gain = -1

        for v in range(n):
            if v in solution:
                continue

            gain = sum(
                1 for w in graph.neighbors(v)
                if graph[v][w]["weight"] <= radius and w not in covered
            )

            if gain > best_gain:
                best_node = v
                best_gain = gain

        if best_node is None:
            print("Nenhum nó aumentou cobertura. Selecionando arbitrário.")
            resto = [v for v in range(n) if v not in solution]
            if not resto:
                break
            best_node = resto[0]
            best_gain = 0

        covered_before = len(covered)

        solution.append(best_node)

        covered.update(
            w for w in graph.neighbors(best_node)
            if graph[best_node][w]["weight"] <= radius
        )
        covered.add(best_node)

        covered_after = len(covered)

        print(f"\nIteração {it+1}")
        print(f"Escolhido: {best_node}")
        print(f"Ganho marginal: {best_gain}")
        print(f"Cobertura total: {covered_after}")
        print(f"Tamanho solução: {len(solution)}")

        # ---- usar heurística TSP, NÃO exato ----
        tour_edges = []
        if len(solution) > 1:
            tour, _ = tsp_nearest_insertion_optimized(graph, solution)
            tour_edges = [
                (tour[i], tour[i+1])
                for i in range(len(tour)-1)
            ]

        draw_iteration(graph, solution, covered, tour_edges, it+1)

        print(f"Tempo iteração: {time.time() - t0:.3f}s")

    print("\n=== FIM DA INICIALIZAÇÃO ===")
    print(f"Solução final: {solution}")
    print(f"Cobertura final: {len(covered)}")

    return solution


# ---------- shaking da solução ----------------------------------------
import random

def shake_random(solution, n, k):
    """
    Implementação do ShakeRandom(x, k) igual ao descrito no artigo:

    - Remove k vértices aleatórios da solução
    - Adiciona k vértices aleatórios diferentes dos removidos
    - Mantém tamanho p
    """

    p = len(solution)
    sol = solution[:]                 # cópia da solução atual

    # 1) Escolhe k índices aleatórios distintos
    remove_idxs = random.sample(range(p), k)

    # 2) Guarda os nós removidos
    removed_nodes = [sol[i] for i in remove_idxs]

    # 3) Remove esses nós da solução
    for idx in sorted(remove_idxs, reverse=True):
        sol.pop(idx)

    # 4) Nós candidatos = todos que NÃO estão na sol e NÃO estão nos removidos
    candidates = [v for v in range(n)
                  if v not in sol and v not in removed_nodes]

    # Se por acaso faltar candidatos (instâncias pequenas), permite recolocar removidos
    if len(candidates) < k:
        candidates = [v for v in range(n) if v not in sol]

    # 5) Escolhe k novos nós aleatórios
    new_nodes = random.sample(candidates, k)

    # 6) Insere cada novo nó em posições aleatórias no tour
    for nd in new_nodes:
        pos = random.randrange(len(sol) + 1)
        sol.insert(pos, nd)

    return sol


# ---------- cobertura e objetivo ---------------------------------------
def calculate_coverage(graph, solution, radius):
    covered = set(solution)  # cada estação cobre a si própria
    for v in solution:
        covered.update({w for w in graph.neighbors(v)
                        if graph[v][w]["weight"] <= radius})
    return len(covered)


def calculate_coverage_set(graph, solution, radius):
    covered = set(solution)
    for v in solution:
        covered.update(
            w for w in graph.neighbors(v)
            if graph[v][w]["weight"] <= radius
        )
    return covered


def build_cover_sets(graph, radius):
    n = graph.number_of_nodes()
    cover_sets = []
    for v in range(n):
        covered = set()
        for w in graph.neighbors(v):
            if graph[v][w]["weight"] <= radius:
                covered.add(w)
        covered.add(v)
        cover_sets.append(covered)
    return cover_sets


def calculate_coverage_from_precomputed(solution, cover_sets):
    covered = set()
    for v in solution:
        covered |= cover_sets[v]
    return covered



# ---------- busca local -------------------------------------------------
def local_search(graph, sol, radius, cover_sets):
    # solução e valor inicial
    best_sol = sol[:]
    best_tour, best_dist = tsp_nearest_insertion_optimized(graph, best_sol)
    best_cov = calculate_coverage(graph, best_sol, radius)

    improved = True

    while improved:
        improved = False

        # guarda o melhor vizinho encontrado nesta iteração
        best_neighbor_sol = None
        best_neighbor_tour = None
        best_neighbor_dist = float("inf")
        best_neighbor_cov = 0

        # percorre todos os possíveis 1-swaps
        for i in range(len(best_sol)):
            for v_new in graph.nodes:
                if v_new in best_sol:
                    continue

                cand = best_sol[:]
                cand[i] = v_new

                # TSP heurístico (igual artigo)
                cand_cov_set = calculate_coverage_from_precomputed(cand, cover_sets)
                cand_cov = len(cand_cov_set)

                if cand_cov < best_cov:
                    continue

                tour, dist = tsp_nearest_insertion_optimized(graph, cand)

                # busca o MELHOR vizinho (não o primeiro)
                if (
                    cand_cov > best_neighbor_cov or
                    (cand_cov == best_neighbor_cov and dist < best_neighbor_dist)
                ):
                    best_neighbor_sol = cand
                    best_neighbor_tour = tour
                    best_neighbor_dist = dist
                    best_neighbor_cov = cand_cov

        # após examinar todos os vizinhos:
        if (
            best_neighbor_cov > best_cov or
            (best_neighbor_cov == best_cov and best_neighbor_dist < best_dist)
        ):
            # aceita o melhor vizinho
            best_sol = best_neighbor_sol
            best_tour = best_neighbor_tour
            best_dist = best_neighbor_dist
            best_cov = best_neighbor_cov
            improved = True

    return best_sol, best_tour, best_dist, best_cov


def local_search_2swap(graph, sol, cover_sets):
    best_sol = sol[:]
    best_cov_set = calculate_coverage_from_precomputed(best_sol, cover_sets)
    best_cov = len(best_cov_set)
    best_tour, best_dist = tsp_nearest_insertion_optimized(graph, best_sol)

    n = graph.number_of_nodes()
    p = len(best_sol)
    max_tries_per_pair = min(200, max(40, int(40000 / n)))

    improved = True
    while improved:
        improved = False

        best_sol_set = set(best_sol)
        outside = [v for v in range(n) if v not in best_sol_set]


        for i in range(p):
            for j in range(i + 1, p):
                if len(outside) < 2:
                    continue

                for _ in range(min(max_tries_per_pair, len(outside))):
                    v1, v2 = random.sample(outside, 2)

                    cand = best_sol[:]
                    cand[i] = v1
                    cand[j] = v2

                    # ---- cobertura primeiro (BARATO) ----
                    cand_cov_set = calculate_coverage_from_precomputed(cand, cover_sets)
                    cand_cov = len(cand_cov_set)

                    if cand_cov < best_cov:
                        continue

                    # ---- TSP só se fizer sentido ----
                    tour, dist = tsp_nearest_insertion_optimized(graph, cand)

                    if (
                        cand_cov > best_cov or
                        (cand_cov == best_cov and dist < best_dist)
                    ):
                        best_sol = cand
                        best_cov = cand_cov
                        best_cov_set = cand_cov_set
                        best_tour = tour
                        best_dist = dist
                        improved = True
                        # print(f"  -> 2-swap melhorou: dist={best_dist:.1f} cov={best_cov} ")
                        break

                if improved:
                    break
            if improved:
                break

    return best_sol, best_tour, best_dist, best_cov


def tour_distance(graph, tour):
    return sum(
        graph[tour[i]][tour[(i + 1) % len(tour)]]["weight"]
        for i in range(len(tour))
    )


def two_opt_swap(tour, i, k):
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]


def two_opt(graph, tour):
    best = tour[:]
    best_dist = tour_distance(graph, best)

    improved = True
    while improved:
        improved = False
        # i começa em 1 para não mexer no primeiro nó (opcional, mas comum)
        for i in range(1, len(best) - 1):
            for k in range(i + 1, len(best)):
                new_tour = two_opt_swap(best, i, k)
                new_dist = tour_distance(graph, new_tour)
                if new_dist < best_dist:
                    best = new_tour
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break

    return best, best_dist


def local_search_2opt(graph, sol, radius, tour=None):
    cov = calculate_coverage(graph, sol, radius)
    if tour is None:
        tour, _ = tsp_nearest_insertion_optimized(graph, sol)
    tour, dist = two_opt(graph, tour)
    return sol[:], tour, dist, cov


def vnd(graph, sol, radius, cover_sets):
    best_sol = sol[:]
    best_tour, best_dist = tsp_nearest_insertion_optimized(graph, best_sol)
    best_cov = len(calculate_coverage_from_precomputed(best_sol, cover_sets))

    k = 1
    k_max = 2   # duas vizinhanças: 1-swap e 2-swap
    last_improvement = None

    while k <= k_max:

        if k == 1:
            cand_sol, cand_tour, cand_dist, cand_cov = local_search(graph, best_sol, radius, cover_sets)
            neigh_name = "1-swap"

        elif k == 2:
            cand_sol, cand_tour, cand_dist, cand_cov = local_search_2swap(graph, best_sol, cover_sets)
            neigh_name = "2-swap"

        if (
            cand_cov > best_cov or
            (cand_cov == best_cov and cand_dist < best_dist)
        ):
            best_sol = cand_sol[:]
            best_tour = cand_tour[:]
            best_dist = cand_dist
            best_cov = cand_cov
            last_improvement = neigh_name 
            k = 1
        else:
            k += 1

    return best_sol, best_tour, best_dist, best_cov, last_improvement


# ---------- carregar parâmetros ----------------------------------------
def load_params(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def plot_final_solution(graph, best_sol, best_tour, radius):
    """Plota solução final com círculos de cobertura reais."""

    pos = {node: graph.nodes[node]["pos"] for node in graph.nodes}

    # calcular nós cobertos
    covered = set(best_sol)
    for v in best_sol:
        covered.update({w for w in graph.neighbors(v)
                        if graph[v][w]["weight"] <= radius})

    not_covered = [v for v in graph.nodes if v not in covered]
    covered_not_selected = [v for v in covered if v not in best_sol]

    fig, ax = plt.subplots(figsize=(8, 6))

    # -------------------------
    # DESENHAR CÍRCULOS DE COBERTURA
    # -------------------------
    for v in best_sol:
        x, y = pos[v]
        circle = Circle(
            (x, y),
            radius,
            edgecolor='red',
            facecolor='red',
            alpha=0.08,
            linewidth=1.5
        )
        ax.add_patch(circle)

    # -------------------------
    # nós
    # -------------------------
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=best_sol,
        node_color="red",
        node_size=140,
        label="Depósitos",
        ax=ax
    )

    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=covered_not_selected,
        node_color="gray",
        node_size=100,
        label="Cobertos",
        ax=ax
    )

    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=not_covered,
        node_color="lightblue",
        node_size=100,
        label="Não cobertos",
        ax=ax
    )

    # -------------------------
    # rota
    # -------------------------
    tour_edges = [(best_tour[i], best_tour[i + 1]) for i in range(len(best_tour) - 1)]
    tour_edges.append((best_tour[-1], best_tour[0]))

    nx.draw_networkx_edges(
        graph, pos,
        edgelist=tour_edges,
        width=2.5,
        edge_color="green",
        ax=ax
    )

    # labels
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)

    # -------------------------
    # escala real
    # -------------------------
    xs = [pos[v][0] for v in graph.nodes]
    ys = [pos[v][1] for v in graph.nodes]

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_aspect("equal")

    ax.set_title("Solução Final com Raio de Cobertura")

    plt.tight_layout()
    plt.savefig("resultado_final.png", dpi=200)
    plt.show()


def run_instance(instance_file, p, radius, max_iter, plot=True):

    # ----- Carrega instância -----
    n, edges, coords = read_tsplib_instance(instance_file)

    g = nx.Graph()
    for i, (x, y) in enumerate(coords):
        g.add_node(i, pos=(x, y))

    g.add_weighted_edges_from(edges)
    
    cover_sets = build_cover_sets(g, radius)
    print("Construiu cover sets")

    # ----- Solução inicial (gulosa) -----
    sol = initialize_solution(g, n, p, radius)
    print("Solução inicial construída")

    init_tour, init_dist = tsp_nearest_insertion_optimized(g, sol)
    init_cov = calculate_coverage(g, sol, radius)

    print(f"Solução inicial  dist={init_dist:.1f} "
          f"cov={init_cov}  estações={sol}")

    # ----- Inicialização do VNS -----
    best_sol = sol[:]
    best_tour = init_tour[:]
    best_dist = init_dist
    best_cov  = init_cov

    k_min = 1
    k_max = math.floor((2/3)*p)

    start = time.time()
    time_best_found = 0.0
    iter_best_found = 0

    # ----- Loop principal do VNS -----
    for it in range(1, max_iter + 1):
        print(f"\nIteração {it}/{max_iter}")
        k = k_min

        while k <= k_max:
            # print(f"  Vizinhança k={k}")
            # ---- Shaking ----
            pert = shake_random(best_sol[:], n, k)

            # ---- VND interno ----
            cand_sol, cand_tour, cand_dist, cand_cov, which_neigh = vnd(g, pert, radius, cover_sets)

            # ---- Critério de aceitação (lexicográfico) ----
            if (
                cand_cov > best_cov or
                (cand_cov == best_cov and cand_dist < best_dist)
            ):
                best_sol  = cand_sol[:]
                best_tour = cand_tour[:]
                best_dist = cand_dist
                best_cov  = cand_cov

                k = k_min

                time_best_found = time.time() - start
                iter_best_found = it
                # print(f"  -> Nova melhor solução! dist={best_dist:.1f} "
                #       f"cov={best_cov} estações={best_sol} "
                #       f"(encontrada na vizinhança {which_neigh})")

            else:
                k += 1

    elapsed = time.time() - start

    if plot:
        plot_final_solution(g, best_sol, best_tour, radius)

    return {
        "instance": instance_file,
        "p": p,
        "radius": radius,
        "final_dist": best_dist,
        "final_cov": best_cov,
        "vns_time": elapsed,
        "time_best_found": time_best_found,
        "iter_best_found": iter_best_found,
        "init_dist": init_dist,
        "init_cov": init_cov,
    }



# ---------- programa principal -----------------------------------------
def main():
    params = load_params("params.json")

    results = run_instance(
        instance_file=params["instance_file"],
        p=params["p"],
        radius=params["coverage_radius"],
        max_iter=params["max_iterations"],
        plot=True,   # deixa True pra manter o gráfico na execução normal
    )

    # Se quiser já ver o dicionário retornado:
    print("\nResumo (para debug / futuro CSV):")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
