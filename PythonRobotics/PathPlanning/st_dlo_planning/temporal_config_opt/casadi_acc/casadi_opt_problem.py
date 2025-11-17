import casadi as ca
import numpy as np


# Smooth sigmoid function with sharpness control
def ca_sigmoid(x, k):
    return 1 / (1 + ca.exp(-k * x))


# compute smooth transitions for each segment
def ca_weight_function(t, t_i, t_next, k):
    return ca_sigmoid(t - t_i, k) * ca_sigmoid(-(t - t_next), k)


def ca_query_point_from_path(t, waypoints, k=120):
    # Step 1: Calculate distances between consecutive waypoints
    diffs = waypoints[1:] - waypoints[:-1]
    distances = ca.sqrt(ca.sum1(ca.power(diffs, 2), axis=1))

    # Step 2: Compute cumulative distance (chord length)
    cumulative_distances = [ca.MX(0)]
    for d in distances:
        cumulative_distances.append(cumulative_distances[-1] + d)
    cumulative_distances = ca.vertcat(*cumulative_distances)

    # Step 3: Normalize to [0, 1]
    total_length = cumulative_distances[-1]
    normalized_distances = cumulative_distances / total_length

    # Step 4: Interpolate point based on t
    point_dim = waypoints.size2()
    point_smooth = ca.MX.zeros(point_dim)
    weight_sum = ca.MX(0)

    for i in range(len(normalized_distances) - 1):
        t_i = normalized_distances[i]
        t_next = normalized_distances[i + 1]
        point_i = waypoints[i] + (waypoints[i + 1] - waypoints[i]) * (t - t_i) / (
            t_next - t_i
        )
        w_i = ca_weight_function(t, t_i, t_next, k)
        point_smooth += w_i * point_i
        weight_sum += w_i

    return point_smooth / weight_sum


def ca_compute_energy(dlo_shape, k1, k2, segment_len):
    num_feature = dlo_shape.shape[0]
    U = 0.0
    for offset, k in zip(range(1, 5), [k1, k2, k2, k2]):
        for j in range(num_feature - offset):
            dist = ca.norm_2(dlo_shape[j + offset, :] - dlo_shape[j, :])
            target = offset * segment_len
            U += k / 2 * (dist - target) ** 2
    return U


class CasadiPathSet:
    def __init__(self, all_path, T: int, seg_len: float):
        self.all_path = all_path  # list of numpy arrays (n_path, n_waypoints, 2 or 3)
        self.num_path = len(all_path)
        self.T = T
        self.seg_len = seg_len

    def query_dlo_shape(self, sigma):
        dlo_shape = []
        for i in range(self.num_path):
            path = ca.MX(self.all_path[i])  # convert to MX for symbolic graph
            t = sigma[i]
            pt = ca_query_point_from_path(t, path)
            dlo_shape.append(pt)
        return ca.vertcat(*dlo_shape).reshape((self.num_path, -1))


class CasadiDloOptProblem:
    def __init__(self, pathset, k1: float, k2: float):
        self.pathset = pathset
        self.k1 = k1
        self.k2 = k2

        self.T = pathset.T
        self.num_path = pathset.num_path
        self.seg_len = pathset.seg_len
        self.N = (self.T + 1) * self.num_path

        self._build_problem()

    def _build_problem(self):
        x = ca.MX.sym("x", self.N)
        sigmas = ca.reshape(x, (self.T + 1, self.num_path))

        energy = 0
        for t in range(self.T):
            sigma_t = sigmas[t, :]
            dlo_shape = self.pathset.query_dlo_shape(sigma_t)
            u = ca_compute_energy(dlo_shape, self.k1, self.k2, self.seg_len)
            energy += u

        diff2 = sigmas[2:] - 2 * sigmas[1:-1] + sigmas[:-2]
        reg = ca.sumsqr(diff2)

        self.obj = energy + 0.5 * reg

        cons = []
        cons.append(sigmas[0, :] - ca.DM.zeros(self.num_path))
        cons.append(sigmas[self.T, :] - ca.DM.ones(self.num_path))
        for t in range(self.T):
            cons.append(sigmas[t + 1, :] - sigmas[t, :])

        self.g = ca.vertcat(*cons)
        self.nlp = {"x": x, "f": self.obj, "g": self.g}
        self.solver = ca.nlpsol(
            "solver",
            "ipopt",
            self.nlp,
            {
                "ipopt.print_level": 0,
                "print_time": False,
                "ipopt.tol": 1e-4,
                "ipopt.max_iter": 500,
            },
        )

    def solve(self, x0):
        lbx = [0] * self.N
        ubx = [1] * self.N

        lbg = [0.0] * self.g.shape[0]  # lower bounds
        ubg = [0.0] * self.g.shape[0]  # exact equality and >= 0

        sol = self.solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        return sol["x"].full().reshape((self.T + 1, self.num_path))
