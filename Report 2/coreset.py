import math
import random

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

"""
 Use kMeans++ algorithm to determine initial centroids for kMeans, 
 based on pseudocode from lecture.
 :param P: Set of points like [[x, y], [x, y], [x, y]]
 :param k: the number of centroids
"""
def k_means_plus_plus(P, k):
    n = len(P)
    x = random.random()
    centroids = []
    centroids.append(P[math.floor(x * n)])
    minDistances = []
    for j in range(n):
        minDistances.append(float('inf'))
    for i in range(1, k):
        for j in range(n):
            x = distance(P[j], centroids[i-1])
            if (minDistances[j] > x):
                minDistances[j] = x
        cumulative = []
        cumulative.append(minDistances[0] ** 2)
        for j in range(1, n):
            cumulative.append(cumulative[j-1] + minDistances[j] ** 2)
        x = random.random()
        x = x * cumulative[n-1]
        if (x <= cumulative[0]):
            index = 1
        else:
            for j in range(1, n):
                if (x > cumulative[j-1] and x <= cumulative[j]):
                    index = j
        centroids.append(P[index])
    return centroids

"""
 Given the centroids, compute the clustering
"""
def compute_labels(P, centroids):
    labels = [-1 for i in range(len(P))]
    for i in range(len(P)):
        minDistance = float('inf')
        for j in range(len(centroids)):
            d = distance(P[i], centroids[j])
            if (d < minDistance):
                minDistance = d
                labels[i] = j
    return labels

"""
 Get the centroids for all clusters
"""
def compute_centroids(P, labels, k):
    cluster_sizes = [0 for i in range(k)]
    point_sum = [[0, 0] for i in range(k)]
    for i in range(len(P)):
        cluster_sizes[labels[i]] += 1
        point_sum[labels[i]][0] += P[i][0]
        point_sum[labels[i]][1] += P[i][1]
    centroids = [[0,0] for i in range(k)]
    for i in range(k):
        centroids[i][0] = point_sum[i][0] / cluster_sizes[i]
        centroids[i][1] = point_sum[i][1] / cluster_sizes[i]
    return centroids

"""
 Have the centroids changed?
"""
def centroids_changed(old, new):
    for i in range(len(old)):
        if (old[i][0] != new[i][0] or old[i][1] != new[i][1]):
            return True
    return False

"""
 Use kMeans algorithm to cluster points, 
 based on pseudocode from lecture.
 :param P: Set of points like [[x, y], [x, y], [x, y]]
 :param k: the number of centroids
"""
def k_means(P, k):
    centroids = k_means_plus_plus(P, k)
    changed = True
    while changed:
        labels = compute_labels(P, centroids)
        old_centroids = centroids
        centroids = compute_centroids(P, labels, k)
        changed = centroids_changed(old_centroids, centroids)
    return centroids, compute_labels(P, centroids)


"""
 Compute the total cost of the assignments
"""
def cost(P, centroids, labels):
    sum = 0
    for i in range(len(P)):
        sum += distance(P[i], centroids[labels[i]])
    return sum

"""
 Is the point inside the circle at center
 with radius?
"""
def inside_ball(point, center, radius):
    return (distance(point, center) <= radius)

"""
 Get all the cells from a ball
"""
def get_ball_cells(center, radius, x):
    # Next to the center cell, how many needed to cover
    # Ball on each side of the center? Multiply by 2 and then add center back
    dim = math.ceil((radius - (x / 2)) / x) * 2 + 1
    # Coords are of the left bottom
    # Cell format [x, y, d]
    cells = [[-1, -1, -1] for i in range(dim ** 2)]
    for i in range(len(cells)):
        # Cell coordinates on the grid, integral
        # index is from left to right bottom to scan
        cx = i % dim
        cy = math.floor(i / dim)
        # Set the cell properties
        cells[i][0] = center[0] - (dim * x / 2) + cx * x
        cells[i][1] = center[1] - (dim * x / 2) + cy * x
        cells[i][2] = x
    return cells

"""
 Return all points indexes that lie within the ball
"""
def get_ball_candidates(center, radius, P, handled):
    candidates = []
    for i in range(len(P)):
        if (inside_ball(P[i], center, radius) and not handled[i]):
            candidates.append(i)
    return candidates

"""
 Does the point lie in the cell ([x, y, dimension])
"""
def inside_cell(cell, point):
    return (point[0] >= cell[0] and point[1] >= cell[1] and point[0] < cell[0] + cell[2] and point[1] < cell[1] + cell[2])

"""
 Given a cell belonging to a ball, get the Points indexes
 that lie within
"""
def get_cell_points(cell, candidates, P):
    result = []
    for i in range(len(candidates)):
        if (inside_cell(cell, P[candidates[i]])):
            result.append(candidates[i])
    return result


"""
 Construct a coreset for the given P, k and eps
 Follows pseudocode from the course notes
 Each point in P is [x, y, weight=1]
"""
def coreset_construction(P, k, eps):
    # 2-Dimensional, and the a in a log (n)
    d = 2
    a = 1

    n = len(P)
    handled = [False for i in range(n)]
    S = []
    z = math.log(n) * math.log(a * math.log(n))
    C, labels = k_means(P, k)
    r = math.sqrt(cost(P, C, labels) / (a * math.log(n) * n))
    x = (eps * r) / math.sqrt(d)

    # For each center
    for c in range(len(C)):
        # Create a ball with a grid
        center = C[c]
        cells = get_ball_cells(center, r, x)
        candidates = get_ball_candidates(center, r, P, handled)
        # Go over all cells
        for i in range(len(cells)):
            # And make a representative if necessary
            cell = cells[i]
            points = get_cell_points(cell, candidates, P)
            # If no points, continue
            if (len(points) == 0):
                continue
            else:
                # Otherwise, get a representative
                rep = P[points[0]]
                # Set all points to handled
                w = 0
                for j in range(len(points)):
                    handled[points[j]] = True
                    w += P[points[j]][2]
                # And add it to S with weight
                S.append([rep[0], rep[1], w])
    
    # Now do the donuts for all centers
    for j in range(1, math.ceil(z) + 1):
        for c in range(len(C)):
            center = C[c]
            # Get cells, radius is 2^j * r, dimension of cells is eps * 2^j * r / sqrt(d)
            cells = get_ball_cells(center, (2**j) * r, (eps * (2 ** j) * r) / math.sqrt(d))
            candidates = get_ball_candidates(center, (2**j) * r, P, handled)
            for i in range(len(cells)):
                cell = cells[i]
                points = get_cell_points(cell, candidates, P)
                # If no points, continue
                if (len(points) == 0):
                    continue
                else:
                    # Otherwise, get a representative
                    rep = P[points[0]]
                    # Set all points to handled
                    w = 0
                    for u in range(len(points)):
                        handled[points[u]] = True
                        w += P[points[u]][2]
                    # And add it to S with weight
                    S.append([rep[0], rep[1], w])
    
    return S