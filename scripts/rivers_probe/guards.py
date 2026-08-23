"""Every guard here raises. Each is proved to fire by job3_guardproof.py, which
plants the exact defect the guard exists to catch."""
import numpy as np
WATER_BIT = 1; UNOBS_BIT = 2

class GuardFailure(AssertionError): pass

def guard_no_channel_on_water_or_unobservable(stream_mask, water_state, label):
    bad_w = int((stream_mask & ((water_state & WATER_BIT) > 0)).sum())
    bad_u = int((stream_mask & ((water_state & UNOBS_BIT) > 0)).sum())
    if bad_w or bad_u:
        raise GuardFailure("%s: %d channel cells on WATER, %d on UNOBSERVABLE" % (label, bad_w, bad_u))
    return bad_w, bad_u

def guard_third_state_not_collapsed(masked_dem, nodata, water_state, label):
    """UNOBSERVABLE must be neither land nor water: it must be absent from the DEM
    handed to the router, so no flow direction is invented for it."""
    unobs = (water_state & UNOBS_BIT) > 0
    leaked = int((unobs & (masked_dem != nodata)).sum())
    if leaked:
        raise GuardFailure("%s: %d UNOBSERVABLE cells carry a finite elevation -> collapsed into land" % (label, leaked))
    return leaked

def guard_mask_partition(masked_dem, nodata, water_state, label):
    expect = int((((water_state & WATER_BIT) > 0) | ((water_state & UNOBS_BIT) > 0)).sum())
    got = int((masked_dem == nodata).sum())
    if expect != got:
        raise GuardFailure("%s: nodata cells %d != water|unobservable cells %d" % (label, got, expect))
    return got
