import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Parameters
N_SAMPLE = 500
N_KNN = 10
MAX_EDGE_LEN = 30.0
show_animation = True


class Node:
    """Node class for Dijkstra search"""
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return f"{self.x},{self.y},{self.cost},{self.parent_index}"


def is_collision(sx, sy, gx, gy, rr, obstacle_list):
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(dy, dx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)
    x, y = sx, sy

    for _ in range(n_step):
        for (ox, oy, size) in obstacle_list:
            dist = math.hypot(ox - x, oy - y)
            if dist <= size + rr:
                return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    for (ox, oy, size) in obstacle_list:
        dist = math.hypot(ox - gx, oy - gy)
        if dist <= size + rr:
            return True  

    return False 


def generate_road_map(sample_x, sample_y, rr, obstacle_list):
    """Road map generation"""
    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_list):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    return road_map


def sample_points(sx, sy, gx, gy, rr, obstacle_list, rng, rand_area):
    min_x, max_x = rand_area[0], rand_area[1]
    min_y, max_y = rand_area[0], rand_area[1]

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = rng.random() * (max_x - min_x) + min_x
        ty = rng.random() * (max_y - min_y) + min_y

        in_collision = False
        for (ox, oy, size) in obstacle_list:
            dist = math.hypot(ox - tx, oy - ty)
            if dist <= size + rr:
                in_collision = True
                break

        if not in_collision:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.extend([sx, gx])
    sample_y.extend([sy, gy])

    return sample_x, sample_y


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    """Dijkstra path planning"""
    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node
    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        if show_animation and len(closed_set.keys()) % 2 == 0:
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None]
            )
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("Goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        del open_set[c_id]
        closed_set[c_id] = current

        for n_id in road_map[c_id]:
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)

            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue

            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if not path_found:
        return [], []

    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index

    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def plot_circle(x, y, size, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)


def prm_planning(sx, sy, gx, gy, obstacle_list, robot_radius, rand_area, *, rng=None):
    """Run PRM planning with circular obstacles"""

    sample_x, sample_y = sample_points(sx, sy, gx, gy,
                                        robot_radius, obstacle_list,
                                        rng, rand_area)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_list)

    rx, ry = dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y)

    return rx, ry


def main(rng=None):
    print(__file__ + " start!!")

    sx, sy = 0.0, 0.0
    gx, gy = 6.0, 10.0
    robot_radius = 0.3
    rand_area = [-2, 15]

    obstacle_list = [
        (1, 10, 1), (3, 10, 1), (3, 8, 1), (3, 6, 1),
        (5, 5, 1), (7, 5, 1), (9, 5, 1), (9, 3, 1),
        (8, 10, 1), (14, 8, 2)
    ]

    if show_animation:
        plt.figure(figsize=(8, 8))
        for (ox, oy, size) in obstacle_list:
            plot_circle(ox, oy, size)
        plt.plot(sx, sy, "^r", label="Start")
        plt.plot(gx, gy, "^c", label="Goal")
        plt.grid(True)
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.legend()

    rx, ry = prm_planning(sx, sy, gx, gy, obstacle_list, robot_radius, rand_area, rng=rng)
    assert rx, 'Cannot find path'

    if show_animation:
        plt.plot(rx, ry, "-r", linewidth=2, label="Path")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()