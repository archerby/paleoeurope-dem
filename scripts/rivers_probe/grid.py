import numpy as np, math, rasterio
R_KM = 6371.0088
def row_cell_km(transform, nrows):
    dlat = -transform.e; dlon = transform.a
    lats = np.array([(transform*(0.5, i+0.5))[1] for i in range(nrows)])
    dy = np.full(nrows, R_KM*math.pi/180.0*dlat)
    dx = R_KM*math.pi/180.0*dlon*np.cos(np.radians(lats))
    return dx, dy, lats
def row_area_km2(transform, nrows):
    dx, dy, _ = row_cell_km(transform, nrows)
    return dx*dy
# WBT D8Pointer encoding
DIAG = (2,8,32,128); EW = (1,16); NS = (4,64)
def d8_step_len_km(pntr, transform, nrows):
    dx, dy, _ = row_cell_km(transform, nrows)
    out = np.zeros(pntr.shape, np.float64)
    p = pntr.astype(np.int32)
    DX = dx[:,None]; DY = dy[:,None]
    m = np.isin(p, EW); out[m] = np.broadcast_to(DX, p.shape)[m]
    m = np.isin(p, NS); out[m] = np.broadcast_to(DY, p.shape)[m]
    m = np.isin(p, DIAG); out[m] = np.broadcast_to(np.hypot(DX, DY), p.shape)[m]
    return out
