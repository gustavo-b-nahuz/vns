import random, math, json, time
import networkx as nx
import matplotlib.pyplot as plt
import os
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
def tsp_nearest_insertion(graph, nodes):
    if len(nodes) <= 1:
        return nodes[:], 0.0

    remaining = nodes[:]
    tour = [remaining.pop(0)]  # começa com um nó arbitrário

    # Enquanto houver nós para inserir
    while remaining:
        # -----------------------------
        # 1) Selecionar o nó mais próximo do tour (NEAREST)
        # -----------------------------
        best_node = None
        best_dist = float("inf")

        for v in remaining:
            # menor distância v → qualquer nó do tour
            d = min(graph[v][u]["weight"] for u in tour)
            if d < best_dist:
                best_dist = d
                best_node = v

        # Remove o nó selecionado
        remaining.remove(best_node)
        
        # Caso especial: tour com apenas 1 vértice
        if len(tour) == 1:
            tour.append(best_node)
            continue

        # -----------------------------
        # 2) Inserir na melhor posição (CHEAPEST INSERTION)
        # -----------------------------
        best_pos = None
        best_increase = float("inf")
        
        # Testando todas as posições do tour (entre cada par consecutivo)
        for i in range(len(tour)):
            u = tour[i]
            w = tour[(i + 1) % len(tour)]  # próximo nó, com ciclo fechado

            increase = (
                    graph[u][best_node]["weight"] +
                    graph[best_node][w]["weight"] -
                    graph[u][w]["weight"]
            )

            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        tour.insert(best_pos, best_node)

    # -----------------------------
    # 3) Calcular a distância final do tour
    # -----------------------------
    total_dist = sum(graph[tour[i]][tour[(i + 1) % len(tour)]]["weight"]
                     for i in range(len(tour)))

    return tour, total_dist


# ---------- construção inicial -----------------------------------------
def initialize_solution(graph, n, p, radius):
    covered, solution = set(), []
    for it in range(p):
        best_node, best_cov = None, 0
        for v in range(n):
            if v in solution:
                continue
            cov = {w for w in graph.neighbors(v)
                   if graph[v][w]["weight"] <= radius and w not in covered}
            if len(cov) > best_cov:
                best_node, best_cov = v, len(cov)
        if best_node is None:                   # ninguém aumenta cobertura
            resto = [v for v in range(n) if v not in solution]
            if not resto:
                break
            best_node = resto[0]
        print(f"\nIteração {it+1}")
        print(f"  escolhido: {best_node}")          # +1 para id humano
        covered_before = len(covered)
        solution.append(best_node)
        covered.update({w for w in graph.neighbors(best_node)
                        if graph[best_node][w]["weight"] <= radius})
        covered.add(best_node)
        print(f"  nova cobertura (ganho): {len(covered) - covered_before}")
        print(f"  cobertos depois: {sorted(w for w in covered)}")
        # ----- gerar tour parcial p/ desenho -------
        tour_edges = []
        if len(solution) > 1:
            tour, _ = tsp_nearest_insertion(graph, solution)
            tour_edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]

        # ----- desenhar / salvar figura ------------
        draw_iteration(graph, solution, covered, tour_edges, it+1)
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


def objective(alpha, dist, cov):
    return alpha * dist - (1 - alpha) * cov


# ---------- busca local -------------------------------------------------
def local_search(graph, sol, radius, alpha):
    best_sol = sol[:]
    best_tour, best_dist = tsp_nearest_insertion(graph, best_sol)
    best_cov = calculate_coverage(graph, best_sol, radius)
    best_obj = objective(alpha, best_dist, best_cov)

    improved = True
    while improved:
        improved = False
        for i in range(len(best_sol)):
            v_old = best_sol[i]
            for v_new in graph.neighbors(v_old):
                if v_new in best_sol:
                    continue
                cand = best_sol[:]
                cand[i] = v_new
                tour, dist = tsp_nearest_insertion(graph, cand)
                cov = calculate_coverage(graph, cand, radius)
                obj = objective(alpha, dist, cov)
                if obj < best_obj:
                    (best_sol, best_tour, best_dist,
                     best_cov, best_obj) = cand, tour, dist, cov, obj
                    improved = True
                    break
            if improved:
                break
    return best_sol, best_tour, best_dist, best_cov, best_obj


# ---------- carregar parâmetros ----------------------------------------
def load_params(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def plot_final_solution(graph, best_sol, best_tour, radius):
    """Plota o resultado final da solução no plano cartesiano real."""

    pos = {node: graph.nodes[node]["pos"] for node in graph.nodes}

    # calcular nós cobertos
    covered = set(best_sol)
    for v in best_sol:
        covered.update({w for w in graph.neighbors(v)
                        if graph[v][w]["weight"] <= radius})

    # nós não cobertos
    not_covered = [v for v in graph.nodes if v not in covered]
    covered_not_selected = [v for v in covered if v not in best_sol]

    plt.figure(figsize=(8, 6))

    # 1. estações selecionadas (vermelho)
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=best_sol,
        node_color="red",
        node_size=140,
        label="Estações selecionadas"
    )

    # 2. cobertos mas não selecionados (cinza)
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=covered_not_selected,
        node_color="gray",
        node_size=100,
        label="Nós cobertos"
    )

    # 3. não cobertos (azul claro)
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=not_covered,
        node_color="lightblue",
        node_size=100,
        label="Não cobertos"
    )

    # 4. rota final (arestas verdes)
    tour_edges = [(best_tour[i], best_tour[i + 1]) for i in range(len(best_tour) - 1)]
    tour_edges.append((best_tour[-1], best_tour[0]))  # fechar ciclo

    nx.draw_networkx_edges(
        graph, pos,
        edgelist=tour_edges,
        width=2.5,
        edge_color="green",
        label="Tour final"
    )

    # labels dos nós
    nx.draw_networkx_labels(graph, pos, font_size=7)

    plt.title("Solução Final – Estações, Cobertura e Rota")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.legend(loc="upper left")

    plt.savefig("resultado_final.png", dpi=200)
    plt.show()


# ---------- programa principal -----------------------------------------
def main():
    params = load_params("params.json")
    n, edges, coords = read_tsplib_instance(params["instance_file"])

    g = nx.Graph()
    for i, (x, y) in enumerate(coords):
        g.add_node(i, pos=(x, y))

    g.add_weighted_edges_from(edges)

    if params.get("plot_graph", False):
        plot_graph(g)

    p              = params["p"]
    radius         = params["coverage_radius"]
    max_iter       = params["max_iterations"]
    alpha          = params["alpha"]
    k_min, k_max   = 1, p

    sol = initialize_solution(g, n, p, radius)
    best_sol = sol[:]
    best_tour, best_dist = tsp_nearest_insertion(g, best_sol)
    best_cov   = calculate_coverage(g, best_sol, radius)
    best_obj   = objective(alpha, best_dist, best_cov)

    print(f"Solução inicial  obj={best_obj:.2f}  dist={best_dist:.1f} "
          f"cov={best_cov}  estações={best_sol}")

    start = time.time()
    for it in range(1, max_iter + 1):
        print(it)
        k = k_min
        while k <= k_max:
            #escolhe outra solução aleatória removendo k vértices e adicionando k do resto
            pert = shake_random(best_sol[:], n, k)

            #calcula rota ótima, distancia, cobertura e objetivo
            # pert_tour, pert_dist = tsp_nearest_insertion(g, pert)
            # pert_cov = calculate_coverage(g, pert, radius)
            # pert_obj = objective(alpha, pert_dist, pert_cov)

            # busca local
            (pert_sol, pert_tour, pert_dist,
             pert_cov, pert_obj) = local_search(g, pert, radius, alpha)

            if pert_obj < best_obj:
                best_sol, best_tour = pert_sol[:], pert_tour[:]
                best_dist, best_cov, best_obj = pert_dist, pert_cov, pert_obj
                k = k_min
            else:
                k += 1

        if it % 50 == 0:
            print(f"iter {it:4d}  obj={best_obj:.2f}")

    elapsed = time.time() - start
    print("\n=== Resultado Final ===")
    print("Estações selecionadas :", best_sol)
    print("Tour                 :", best_tour)
    print(f"Distância tour       : {best_dist:.1f}")
    print(f"Cobertura            : {best_cov}")
    print(f"Objetivo final       : {best_obj:.2f}")
    print(f"Tempo (s)            : {elapsed:.2f}")

    plot_final_solution(g, best_sol, best_tour, radius)


if __name__ == "__main__":
    main()
