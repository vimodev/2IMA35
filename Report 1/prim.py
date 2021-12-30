import math
import random
from datetime import datetime

from helpers import create_distance_matrix, plot_mst


def prim(dataset):
    vertex_count = len(dataset[0][0])
    dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])

    used = set()
    unused = set()
    start = random.randint(0, vertex_count)
    used.add(start)

    mst = []
    for i in range(0, vertex_count):
        if i != start:
            unused.add(i)

    print("[PRIM] Start creating MST...")
    timestamp = datetime.now()
    while len(unused) != 0:
        if len(unused) % 100 == 0:
            print(len(used), vertex_count)
        min_weight = math.inf
        s = -1
        t = -1
        for vertex in used:
            for target in unused:
                cost = dm[vertex][target]
                if min_weight > cost:
                    min_weight = cost
                    s = vertex
                    t = target
        mst.append((s, t, dm[s][t]))
        unused.remove(t)
        used.add(t)
    print("[PRIM] Found MST in: ", datetime.now() - timestamp)
    score = 0
    for edge in mst:
        score = score+edge[2]
    print("[PRIM] Score: ", score)
    plot_mst(dataset[0][0], mst, False, False)