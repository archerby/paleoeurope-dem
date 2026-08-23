"""Connected-path minimax (the elevation of the lowest saddle on the best route)
between two cells, at a stated connectivity. Bisection on the threshold, with
scipy connected-component labelling as the connectivity oracle."""
import numpy as np, scipy.ndimage as nd

S4 = np.array([[0,1,0],[1,1,1],[0,1,0]], bool)
S8 = np.ones((3,3), bool)

def connected_at(dem, t, a, b, struct):
    m = dem <= t
    if not (m[a] and m[b]): return False
    lab, n = nd.label(m, structure=struct)
    return lab[a] != 0 and lab[a] == lab[b]

def minimax(dem, a, b, struct, lo=None, hi=None, tol=1e-4, log=None):
    lo = float(max(dem[a], dem[b])) if lo is None else lo
    hi = float(dem.max()) if hi is None else hi
    if not connected_at(dem, hi, a, b, struct):
        return None, 0
    it = 0
    while hi - lo > tol:
        mid = 0.5*(lo+hi); it += 1
        if connected_at(dem, mid, a, b, struct): hi = mid
        else: lo = mid
    return hi, it
