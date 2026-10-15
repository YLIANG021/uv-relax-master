# numpy-only UV math.  No bpy types: every function runs on plain arrays, so the exact
# same code is testable outside Blender.

import hashlib
import time

import numpy as np


# Small, bounded caches for data whose identity is determined by the mesh arrays.  Blender
# recreates the arrays on edit, so the byte signatures keep stale topology or 3D metrics from being
# reused while still avoiding repeated global numpy work during ordinary UV-only edits.
_TOPOLOGY_CACHE = {}
_GEOMETRY_CACHE = {}
_ISLAND_REFERENCE_CACHE = {}
_OVERLAP_INDEX_CACHE = {}


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.blake2b(memoryview(array).cast("B"), digest_size=16).digest()
    return (array.shape, array.dtype.str, digest)


def cached_uv_topology(tri, tuv, limit=1, profile=None):
    """Return UV point ids and island labels, reusing unchanged mesh/UV seam topology."""
    total_t0 = time.perf_counter()
    profile = profile if profile is not None else {}
    profile.update(cache="miss", signature_ms=0.0, build_ms=0.0)
    tri = np.asarray(tri)
    tuv = np.asarray(tuv)
    quantized_uv = np.rint(tuv * 1e5).astype(np.int64)
    key = _array_digest(tri)
    uv_signature = _array_digest(quantized_uv)
    signature_t1 = time.perf_counter()
    profile["signature_ms"] = (signature_t1 - total_t0) * 1000.0
    entries = _TOPOLOGY_CACHE.setdefault(key, [])
    corner_uv = quantized_uv.reshape(-1, 2)
    corner_vertex = tri.reshape(-1)
    for entry_index, entry in enumerate(entries):
        inv, point_isl, point_vertex, first_corner, metadata, cached_uv_signature = entry
        if cached_uv_signature == uv_signature:
            profile["cache"] = "hit"
            profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
            return inv, point_isl, metadata
        representative_uv = corner_uv[first_corner]
        # A changed value is harmless when every corner belonging to a cached UV point changed
        # together.  This catches a split point without rebuilding the global unique table.
        if not np.array_equal(corner_uv, representative_uv[inv.reshape(-1)]):
            continue
        # Conversely, two previously separate points must not have become coincident on the same
        # model vertex.  The check is over one representative per UV point, not every corner.
        point_keys = np.concatenate([point_vertex[:, None], representative_uv], axis=1)
        if len(np.unique(point_keys, axis=0)) != len(point_vertex):
            continue
        entries[entry_index] = (inv, point_isl, point_vertex, first_corner,
                                metadata, uv_signature)
        profile["cache"] = "uv_refresh"
        profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        return inv, point_isl, metadata

    build_t0 = time.perf_counter()
    inv = uvvert_ids(tri, tuv)
    U = int(inv.max()) + 1
    point_isl = islands(inv, U)
    inv_flat = inv.reshape(-1)
    # np.unique returns the point ids in its first result and corner indices in its second.
    point_ids, first = np.unique(inv_flat, return_index=True)
    first_corner = np.zeros(U, np.int64)
    first_corner[point_ids] = first
    point_vertex = np.zeros(U, np.int64)
    point_vertex[point_ids] = corner_vertex[first]
    metadata = _build_uv_topology_metadata(inv, point_isl)
    entries.append((inv, point_isl, point_vertex, first_corner, metadata, uv_signature))
    while len(entries) > limit:
        entries.pop(0)
    while len(_TOPOLOGY_CACHE) > 1:
        _TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))
    profile["build_ms"] = (time.perf_counter() - build_t0) * 1000.0
    profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
    return inv, point_isl, metadata


def _group_offsets(labels, count):
    labels = np.asarray(labels, dtype=np.int64)
    sizes = np.bincount(labels, minlength=count)
    offsets = np.empty(count + 1, np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    return np.argsort(labels, kind="stable"), offsets


def _build_uv_topology_metadata(inv, point_isl):
    """Build reusable island groups, point-to-face adjacency, and island boundary edges."""
    face_isl = point_isl[inv[:, 0]]
    island_count = int(point_isl.max()) + 1 if len(point_isl) else 0
    face_order, face_offsets = _group_offsets(face_isl, island_count)
    point_order, point_offsets = _group_offsets(point_isl, island_count)

    face_ids = np.repeat(np.arange(len(inv), dtype=np.int64), 3)
    point_face = np.unique(np.stack([inv.ravel(), face_ids], axis=1), axis=0)
    incident_order = np.argsort(point_face[:, 0], kind="stable")
    incident_face = point_face[incident_order, 1]
    incident_count = np.bincount(point_face[:, 0], minlength=len(point_isl))
    incident_offsets = np.empty(len(point_isl) + 1, np.int64)
    incident_offsets[0] = 0
    np.cumsum(incident_count, out=incident_offsets[1:])

    corner_pairs = ((0, 1), (1, 2), (2, 0))
    edge_points = np.concatenate([inv[:, list(pair)] for pair in corner_pairs], axis=0)
    edge_points = np.sort(edge_points, axis=1)
    edge_labels = np.tile(face_isl, 3)
    edge_faces = np.tile(np.arange(len(inv), dtype=np.int64), 3)
    edge_corners = np.concatenate([
        np.broadcast_to(np.asarray(pair, np.int8), (len(inv), 2))
        for pair in corner_pairs
    ], axis=0)
    edge_keys = np.column_stack([edge_labels, edge_points])
    unique_edges, first_edge, edge_count = np.unique(
        edge_keys, axis=0, return_index=True, return_counts=True)
    boundary = edge_count == 1
    boundary_labels = unique_edges[boundary, 0]
    boundary_edges = unique_edges[boundary, 1:]
    boundary_order, boundary_offsets = _group_offsets(boundary_labels, island_count)
    boundary_faces = edge_faces[first_edge[boundary]]
    boundary_corners = edge_corners[first_edge[boundary]]
    return {
        "face_isl": face_isl,
        "face_order": face_order,
        "face_offsets": face_offsets,
        "point_order": point_order,
        "point_offsets": point_offsets,
        "incident_offsets": incident_offsets,
        "incident_faces": incident_face,
        "boundary_order": boundary_order,
        "boundary_offsets": boundary_offsets,
        "boundary_edges": boundary_edges,
        "boundary_isl": boundary_labels,
        "boundary_faces": boundary_faces,
        "boundary_corners": boundary_corners,
        "island_count": island_count,
    }


def grouped_indices(metadata, group, kind="face"):
    order = metadata[kind + "_order"]
    offsets = metadata[kind + "_offsets"]
    return order[offsets[group]:offsets[group + 1]]


def incident_faces(metadata, point_ids):
    point_ids = np.asarray(point_ids, dtype=np.int64)
    if not len(point_ids):
        return np.zeros(0, np.int64)
    offsets = metadata["incident_offsets"]
    lengths = offsets[point_ids + 1] - offsets[point_ids]
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    starts = offsets[point_ids]
    repeat_starts = np.repeat(starts, lengths)
    within = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.unique(metadata["incident_faces"][repeat_starts + within])


def cached_island_references(co, tri, inv, point_isl, tuv, pos, metadata,
                             geom_p, geom_a3, limit=1, profile=None):
    """Cache island-wide area, density, orientation, scale, center, and boundary data."""
    total_t0 = time.perf_counter()
    profile = profile if profile is not None else {}
    profile.update(cache="miss", signature_ms=0.0, build_ms=0.0)
    key = (_array_digest(tri), _array_digest(inv), _array_digest(tuv), _array_digest(co))
    profile["signature_ms"] = (time.perf_counter() - total_t0) * 1000.0
    cached = _ISLAND_REFERENCE_CACHE.get(key)
    if cached is not None:
        profile["cache"] = "hit"
        profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        return cached

    build_t0 = time.perf_counter()
    face_isl = metadata["face_isl"]
    point_isl = np.asarray(point_isl, dtype=np.int64)
    island_count = metadata["island_count"]
    q = signed_uv_area(tuv)
    abs_area2 = np.bincount(face_isl, weights=np.abs(q), minlength=island_count)
    total_a3 = np.bincount(face_isl, weights=geom_a3, minlength=island_count)
    density_target = np.zeros(island_count, dtype=float)
    np.divide(0.5 * abs_area2, total_a3, out=density_target, where=total_a3 > 1e-30)
    positive = np.bincount(face_isl, weights=(q > 0.0), minlength=island_count)
    negative = np.bincount(face_isl, weights=(q < 0.0), minlength=island_count)
    orientation = np.where(positive >= negative, 1.0, -1.0)

    uv_center = np.zeros((island_count, 2), dtype=float)
    np.add.at(uv_center, point_isl, pos)
    point_count = np.bincount(point_isl, minlength=island_count)
    np.divide(uv_center, point_count[:, None], out=uv_center,
              where=point_count[:, None] > 0)
    uv_min = np.full((island_count, 2), np.inf)
    uv_max = np.full((island_count, 2), -np.inf)
    np.minimum.at(uv_min, point_isl, pos)
    np.maximum.at(uv_max, point_isl, pos)
    spread = np.linalg.norm(uv_max - uv_min, axis=1) + 1e-30

    boundary_faces = metadata["boundary_faces"]
    boundary_corners = metadata["boundary_corners"]
    boundary_isl = metadata["boundary_isl"]
    if len(boundary_faces):
        edge_xyz = geom_p[boundary_faces[:, None], boundary_corners]
        boundary_lengths = np.linalg.norm(edge_xyz[:, 1] - edge_xyz[:, 0], axis=1)
    else:
        boundary_lengths = np.zeros(0, dtype=float)
    boundary_order, boundary_offsets = _group_offsets(boundary_isl, island_count)
    cached = {
        "total_a3": total_a3,
        "area2": abs_area2,
        "density_target": density_target,
        "orientation": orientation,
        "center": uv_center,
        "spread": spread,
        "boundary_order": boundary_order,
        "boundary_offsets": boundary_offsets,
        "boundary_edges": metadata["boundary_edges"],
        "boundary_lengths": boundary_lengths,
    }
    _ISLAND_REFERENCE_CACHE[key] = cached
    while len(_ISLAND_REFERENCE_CACHE) > limit:
        _ISLAND_REFERENCE_CACHE.pop(next(iter(_ISLAND_REFERENCE_CACHE)))
    profile["build_ms"] = (time.perf_counter() - build_t0) * 1000.0
    profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
    return cached


def island_boundary(metadata, references, island):
    order = references["boundary_order"]
    offsets = references["boundary_offsets"]
    ids = order[offsets[island]:offsets[island + 1]]
    return (references["boundary_edges"][ids],
            references["boundary_lengths"][ids], len(ids))


def cached_uv_geometry(co, tri, limit=1, profile=None):
    """Return 3D corners, areas, and metric basis for an unchanged triangulated mesh."""
    total_t0 = time.perf_counter()
    profile = profile if profile is not None else {}
    profile.update(cache="miss", signature_ms=0.0, build_ms=0.0)
    co = np.asarray(co)
    tri = np.asarray(tri)
    key = (_array_digest(co), _array_digest(tri))
    profile["signature_ms"] = (time.perf_counter() - total_t0) * 1000.0
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        build_t0 = time.perf_counter()
        p, a3 = uv3d(co, tri)
        cached = (p, a3, face_metric_prep(p))
        _GEOMETRY_CACHE[key] = cached
        profile["build_ms"] = (time.perf_counter() - build_t0) * 1000.0
        while len(_GEOMETRY_CACHE) > limit:
            _GEOMETRY_CACHE.pop(next(iter(_GEOMETRY_CACHE)))
    else:
        profile["cache"] = "hit"
    profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
    return cached


def face_metric_prep(p):
    """Precompute the model-only part of face_metric: Ginv @ TN.

    This depends only on the 3D geometry, so it can be hoisted out of the relax loop.
    Pass the result to face_metric_uv() for each UV candidate.
    Returns (T,2,2) array."""
    t1 = p[:, 1] - p[:, 0]
    t2 = p[:, 2] - p[:, 0]
    la = np.sqrt(np.maximum((t1 * t1).sum(1), 1e-30))
    c = (t1 * t2).sum(1) / la
    lb2 = np.sqrt(np.maximum(((t2 - c[:, None] * (t1 / la[:, None])) ** 2).sum(1), 1e-30))
    a = (t1 * t1).sum(1)
    b = (t1 * t2).sum(1)
    d = (t2 * t2).sum(1)
    detG = np.maximum(a * d - b * b, 1e-30)
    Ginv = np.stack([np.stack([d, -b], 1), np.stack([-b, a], 1)], 1) / detG[:, None, None]
    TN = np.stack([np.stack([la, np.zeros_like(la)], 1), np.stack([c, lb2], 1)], 1)
    return Ginv @ TN


def face_metric_uv(prepped, tuv):
    """Per-face slack from precomputed geometry and UV corners tuv (T,3,2).

    Returns (shape, det):  det = UV area / 3D area, i.e. texel density up to one
    constant shared by the whole mesh;  shape = distortion ratio, 1.0 means conformal."""
    s1 = tuv[:, 1] - tuv[:, 0]
    s2 = tuv[:, 2] - tuv[:, 0]
    S = np.stack([np.stack([s1[:, 0], s2[:, 0]], 1), np.stack([s1[:, 1], s2[:, 1]], 1)], 1)
    K = S @ prepped
    k00, k01, k10, k11 = K[:, 0, 0], K[:, 0, 1], K[:, 1, 0], K[:, 1, 1]
    tr = k00 * k00 + k01 * k01 + k10 * k10 + k11 * k11
    det = k00 * k11 - k01 * k10
    disc = np.sqrt(np.maximum(tr * tr - 4 * det * det, 0.0))
    g1 = np.sqrt(np.maximum((tr + disc) * 0.5, 1e-30))
    g2 = np.sqrt(np.maximum((tr - disc) * 0.5, 1e-30))
    return g1 / np.maximum(g2, 1e-12), np.abs(det)


def face_metric(p, tuv):
    """Per-face slack from model corners p (T,3,3) and UV corners tuv (T,3,2).

    Returns (shape, det):  det = UV area / 3D area, i.e. texel density up to one
    constant shared by the whole mesh;  shape = distortion ratio, 1.0 means conformal."""
    return face_metric_uv(face_metric_prep(p), tuv)


def uv3d(co, tri):
    """Model corners and 3D face areas of triangulation (tri = (T,3) vertex ids)."""
    p = co[tri]
    a3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return p, a3


def boundary_geometry(p, tl):
    """Return UV edge ids and their 3D lengths for the outline of a triangulated region."""
    all_edges = np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]], tl[:, [2, 0]]], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    all_model_edges = np.concatenate([p[:, [0, 1]], p[:, [1, 2]], p[:, [2, 0]]], axis=0)
    edge_order = np.lexsort((all_edges[:, 1], all_edges[:, 0]))
    sorted_edges = all_edges[edge_order]
    sorted_model_edges = all_model_edges[edge_order]
    unique_edges, first_edge, edge_count = np.unique(
        sorted_edges, axis=0, return_index=True, return_counts=True)
    boundary_edges = unique_edges[edge_count == 1]
    boundary_model_edges = sorted_model_edges[first_edge[edge_count == 1]]
    lengths = np.linalg.norm(boundary_model_edges[:, 1] - boundary_model_edges[:, 0], axis=1)
    valid = lengths > 1e-30
    return boundary_edges[valid], lengths[valid]


def uv_area(p, a3, tuv):
    """Summed UV area of faces given model corners, their 3D areas and UV corners."""
    return float((face_metric(p, tuv)[1] * a3).sum())


def uvvert_ids(tri, tuv):
    """Identity of a UV point = same 3D vertex AND same UV coordinate.  Coincident
    corners along one seam merge; corners at different seams of a vertex stay apart."""
    key = np.stack([tri, np.rint(tuv[..., 0] * 1e5).astype(np.int64),
                    np.rint(tuv[..., 1] * 1e5).astype(np.int64)], -1).reshape(-1, 3)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(tri.shape)


def _root(par, i):
    while True:
        p = par[i]
        done = p == i
        if done.all():
            return i
        i = np.where(done, i, p)


def islands(inv, U):
    """UV-space connected components, labelled 0..n-1, one label per UV point."""
    par = np.arange(U)
    for _ in range(60):
        old = par.copy()
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ra = _root(par, inv[:, a])
            rb = _root(par, inv[:, b])
            m = ra != rb
            if m.any():
                hi = np.maximum(ra[m], rb[m])
                lo = np.minimum(ra[m], rb[m])
                o = np.unique(hi, return_index=True)[1]
                par[hi[o]] = lo[o]
        if np.array_equal(par, old):
            break
    _, lab = np.unique(_root(par, np.arange(U)), return_inverse=True)
    return lab


def glued_points(tri, inv, isl):
    """Which UV points something else is matched to, so moving that one alone would show on the model.

    Two kinds of glue.  A 3D vertex that owns more than one UV point is cut by a seam: the two sides
    of the cut sit at different places in the map and each samples its own patch of the texture, so
    moving one of them and not the other is exactly what makes a seam visibly tear.  And a UV point
    that faces of more than one island sit on is the mirror case -- `uvvert_ids` merged two islands
    there, and they are one unknown now, so they cannot be pulled apart.

    What is left is the edge of the model itself: nothing on the mesh continues past it and nothing in
    the map is matched to it, so relaxing may move it.  That is what lets a border that was unwrapped
    badly be repaired instead of being kept as the shape everything else has to fit inside.
    `isl` are per-face island labels.  Returns (U,) bool, one entry per UV point."""
    U = int(inv.max()) + 1
    glued = np.zeros(U, bool)
    v = tri.ravel()
    w = inv.ravel()
    if len(v) == 0:
        return glued
    sides = np.unique(np.stack([v, w], 1), axis=0)
    cnt = np.bincount(sides[:, 0], minlength=int(v.max()) + 1)
    glued[sides[cnt[sides[:, 0]] > 1, 1]] = True
    both = np.unique(np.stack([w, np.repeat(np.asarray(isl), 3)], 1), axis=0)
    cnt = np.bincount(both[:, 0], minlength=U)
    glued[both[cnt[both[:, 0]] > 1, 0]] = True
    return glued


def signed_uv_area(t):
    """Twice each face's UV area, with its sign: negative means the face is inside out.
    Takes the face's three corners (T,3,2)."""
    return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) \
        - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])


def _boundary_loops(tl, p):
    """Return all ordered boundary loops and their 3D edge lengths.

    UV islands normally have one loop.  Choosing the longest connected loop keeps holes from
    becoming the outer circle when a mesh contains them, while the empty result lets callers fall
    back to a projection for closed or non-manifold input.
    """
    edges = np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]], tl[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    _, first, count = np.unique(edges, axis=0, return_index=True, return_counts=True)
    bedges = edges[first[count == 1]]
    if len(bedges) < 3:
        return []

    adjacency = {}
    edge_at = {}
    for i, (a, b) in enumerate(bedges.tolist()):
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        edge_at.setdefault(a, []).append(i)
        edge_at.setdefault(b, []).append(i)
    unseen = set(range(len(bedges)))
    loops = []
    while unseen:
        seed = unseen.pop()
        component = [seed]
        stack = [seed]
        while stack:
            i = stack.pop()
            a, b = bedges[i]
            for vertex in (int(a), int(b)):
                for j in edge_at[vertex]:
                    if j in unseen:
                        unseen.remove(j)
                        component.append(j)
                        stack.append(j)
        comp_edges = bedges[component]
        degree = {}
        for a, b in comp_edges.tolist():
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        if len(component) < 3 or any(v != 2 for v in degree.values()):
            continue
        start = int(comp_edges[0, 0])
        ordered = [start]
        previous = -1
        current = start
        while True:
            nxt = [v for v in adjacency[current] if v != previous]
            if not nxt:
                break
            candidate = int(nxt[0])
            if candidate == start:
                break
            ordered.append(candidate)
            previous, current = current, candidate
            if len(ordered) > len(component):
                break
        if len(ordered) == len(component):
            xyz = np.zeros((len(tl), 3), dtype=float)
            xyz[tl.ravel()] = p.reshape(-1, 3)
            length = np.linalg.norm(xyz[np.roll(ordered, -1)] - xyz[ordered], axis=1)
            loops.append((float(length.sum()), np.asarray(ordered, np.int64), length))
    return loops


def _boundary_loop(tl, p):
    """Return the longest ordered boundary loop of a triangulated disk.

    UV islands normally have one loop.  Choosing the longest connected loop keeps holes from
    becoming the outer circle when a mesh contains them, while the empty result lets callers fall
    back to a projection for closed or non-manifold input.
    """
    loops = _boundary_loops(tl, p)
    if not loops:
        return np.zeros(0, np.int64), np.zeros(0, float)
    _, loop, lengths = max(loops, key=lambda item: item[0])
    return loop, lengths


def _cap_holes(p, tl, uv0):
    """Add temporary 3D triangle fans that make each inner boundary part of the solve.

    The extra center vertices and faces are solver-only.  The caller removes the virtual vertices
    from the returned UV, so the mesh topology and the user's UV selection never change.
    """
    loops = _boundary_loops(tl, p)
    if len(loops) <= 1:
        return p, tl, uv0, 0
    outer = max(loops, key=lambda item: item[0])[1]
    outer_set = set(outer.tolist())
    holes = [item[1] for item in loops if not set(item[1].tolist()).issubset(outer_set)]
    if not holes:
        return p, tl, uv0, 0

    xyz = np.zeros((len(uv0), 3), dtype=float)
    xyz[tl.ravel()] = p.reshape(-1, 3)
    q = signed_uv_area(uv0[tl])
    nz = q[np.abs(q) > 1e-14 * max(float(np.abs(q).max()), 1e-30)]
    majority = 1.0 if len(nz) == 0 or int((nz > 0).sum()) >= int((nz < 0).sum()) else -1.0

    add_p = []
    add_tl = []
    add_uv = []
    for loop in holes:
        center_xyz = xyz[loop].mean(0)
        hole_uv = uv0[loop]
        hole_next = np.roll(hole_uv, -1, axis=0)
        cross = hole_uv[:, 0] * hole_next[:, 1] - hole_next[:, 0] * hole_uv[:, 1]
        area2 = float(cross.sum())
        if abs(area2) > 1e-20:
            center_uv = ((hole_uv + hole_next) * cross[:, None]).sum(0) / (3.0 * area2)
        else:
            center_uv = hole_uv.mean(0)
        if not np.isfinite(center_uv).all():
            center_uv = hole_uv.mean(0)
        center_id = len(uv0) + len(add_uv)
        add_uv.append(center_uv)
        for a, b in zip(loop, np.roll(loop, -1)):
            a, b = int(a), int(b)
            area = ((uv0[b, 0] - uv0[a, 0]) * (center_uv[1] - uv0[a, 1]) -
                    (uv0[b, 1] - uv0[a, 1]) * (center_uv[0] - uv0[a, 0]))
            if area * majority < 0.0:
                a, b = b, a
            add_tl.append((center_id, a, b))
            add_p.append((center_xyz, xyz[a], xyz[b]))
    if not add_tl:
        return p, tl, uv0, 0
    return (np.concatenate([p, np.asarray(add_p, dtype=float)], axis=0),
            np.concatenate([tl, np.asarray(add_tl, dtype=np.int64)], axis=0),
            np.concatenate([uv0, np.asarray(add_uv, dtype=float)], axis=0),
            len(add_uv))


def reinitialize_uv(p, tl, uv0):
    """Build a stable disk parameterization from 3D geometry for a badly damaged full island.

    The boundary is fixed to a circle by 3D edge length and the interior is solved with a positive
    graph Laplacian.  The result is then Procrustes-aligned and area-matched to the incoming UV so
    a repair does not unexpectedly jump across the image editor.
    """
    V = len(uv0)
    loop, lengths = _boundary_loop(tl, p)
    if len(loop) < 3:
        # A closed island has no boundary to anchor a disk map.  PCA still gives a deterministic
        # starting chart, and the subsequent ARAP solve can improve it without using old UV data.
        xyz = np.zeros((V, 3), dtype=float)
        xyz[tl.ravel()] = p.reshape(-1, 3)
        xyz -= xyz.mean(0)
        _, _, vh = np.linalg.svd(xyz, full_matrices=False)
        out = xyz @ vh[:2].T
        if np.linalg.norm(out.max(0) - out.min(0)) <= 1e-20:
            return uv0.copy()
    else:
        xyz = np.zeros((V, 3), dtype=float)
        xyz[tl.ravel()] = p.reshape(-1, 3)
        out = np.zeros((V, 2), dtype=float)
        perimeter = max(float(lengths.sum()), 1e-30)
        angle = 2.0 * np.pi * np.cumsum(np.r_[0.0, lengths[:-1]]) / perimeter
        # Keep the majority orientation of the existing chart when it is meaningful.
        q = signed_uv_area(uv0[tl])
        if int((q < 0.0).sum()) > int((q > 0.0).sum()):
            angle = -angle
        out[loop] = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        boundary = np.zeros(V, bool)
        boundary[loop] = True

        edges = np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]], tl[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        edge_xyz = xyz[edges[:, 1]] - xyz[edges[:, 0]]
        ew = 1.0 / np.maximum(np.linalg.norm(edge_xyz, axis=1), 1e-12)
        diag = np.bincount(edges.ravel(), weights=np.repeat(ew, 2), minlength=V)
        interior = ~boundary
        interior_ids = np.flatnonzero(interior)
        if len(interior_ids):
            local = -np.ones(V, np.int64)
            local[interior_ids] = np.arange(len(interior_ids))
            ii = interior_ids
            rhs = np.zeros((len(ii), 2), dtype=float)
            ea, eb = edges[:, 0], edges[:, 1]
            ab = interior[ea] & boundary[eb]
            ba = interior[eb] & boundary[ea]
            np.add.at(rhs, local[ea[ab]], ew[ab, None] * out[eb[ab]])
            np.add.at(rhs, local[eb[ba]], ew[ba, None] * out[ea[ba]])

            both = interior[ea] & interior[eb]
            ia, ib, iw = local[ea[both]], local[eb[both]], ew[both]

            def mat(x):
                y = diag[ii] * x
                np.add.at(y, ia, -iw * x[ib])
                np.add.at(y, ib, -iw * x[ia])
                return y

            x = np.zeros_like(rhs)
            for axis in (0, 1):
                bvec = rhs[:, axis]
                r = bvec - mat(x[:, axis])
                z = r / np.maximum(diag[ii], 1e-30)
                d = z.copy()
                rz = float(r @ z)
                for _ in range(min(1200, max(80, 4 * len(ii)))):
                    hd = mat(d)
                    den = float(d @ hd)
                    if den <= 1e-30:
                        break
                    alpha = rz / den
                    x[:, axis] += alpha * d
                    r -= alpha * hd
                    if float(np.linalg.norm(r)) <= 1e-7 * max(float(np.linalg.norm(bvec)), 1e-30):
                        break
                    z = r / np.maximum(diag[ii], 1e-30)
                    rz_new = float(r @ z)
                    d = z + (rz_new / max(rz, 1e-30)) * d
                    rz = rz_new
            out[ii] = x

    source_center = out.mean(0)
    target_center = uv0.mean(0)
    src = out - source_center
    dst = uv0 - target_center
    h = src.T @ dst
    uu, _, vv = np.linalg.svd(h)
    rot = uu @ vv
    if np.linalg.det(rot) < 0.0:
        uu[:, -1] *= -1.0
        rot = uu @ vv
    out = src @ rot
    old_area = float(np.abs(signed_uv_area(uv0[tl])).sum())
    new_area = float(np.abs(signed_uv_area(out[tl])).sum())
    if new_area > 1e-30:
        out *= np.sqrt(max(old_area, 1e-30) / new_area)
    return out + target_center


# how far a model triangle's angles are allowed to go before its stiffness is capped: a face
# squashed flat would otherwise ask for an infinite weight, and a face thrown out entirely asks for
# nothing at all, which leaves it dead material the rest of the island flows around.  RizomUV
# clamps the angle it reads a cotangent weight from into [MinAngle, pi - MinAngle] and its
# `#MinAngle` default is the same order as this.
MIN_ANGLE = 1e-5
MIN_ANGLE_SIN = float(np.sin(MIN_ANGLE))


def stretch_terms(p, tuv, e_side=None):
    """Per triangle, the stretch that relaxing is meant to remove.

    The triangle's own map from the model to UV is compared with the nearest rotation plus
    uniform scale of *the orientation it already has*, so what is left over is shear and
    uneven scaling only, and an island that was mirrored on purpose stays mirrored.  Pass
    `e_side` (+1 or -1) to ask every triangle for that orientation instead, which is what an
    island with a few faces folded over wants: see relax_solve.  Returns (A, w): the two
    coefficient blocks A (T,2,3,2) and the weight w (T,).  The island's stretch energy is then
    sum_t w_t * (A_t . its six corners)**2 -- quadratic in the UV coordinates, so walking it
    down is a matter of repeating one simple step.

    The weight is 1/(8 * model area), so a face squashed towards a line wants to be stiffer than
    any of its neighbours without limit.  It is capped instead of dropped: the area the weight is
    taken from is floored at the smallest area the face's own two longest edges allow once every
    angle is held inside [MIN_ANGLE, pi - MIN_ANGLE], which bounds how hard one sliver can pull
    while still letting it take part in the relax.  Only a face with no length in it at all says
    nothing about stretch, and gets weight zero.  And note that this energy is as content with a
    smaller island as with an even one: see relax_solve."""
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    e3 = p[:, 2] - p[:, 1]
    l1 = np.sqrt(np.maximum((e1 * e1).sum(1), 1e-30))
    c = (e1 * e2).sum(1) / l1
    b2 = np.sqrt(np.maximum(((e2 - (c / l1)[:, None] * e1) ** 2).sum(1), 0.0))
    a3 = 0.5 * l1 * b2
    edges = np.sort(np.stack([np.sqrt((e1 * e1).sum(1)), np.sqrt((e2 * e2).sum(1)),
                              np.sqrt((e3 * e3).sum(1))], 1), 1)
    a3_cap = np.maximum(a3, 0.5 * edges[:, 1] * edges[:, 2] * MIN_ANGLE_SIN)
    if e_side is None:
        e = np.sign(signed_uv_area(tuv))
        e[e == 0.0] = 1.0
    else:
        e = np.full(len(l1), float(e_side))
    z = np.zeros_like(b2)
    A1 = np.stack([np.stack([-b2, e * (l1 - c)], -1), np.stack([b2, e * c], -1),
                   np.stack([z, -e * l1], -1)], 1)
    A2 = np.stack([np.stack([c - l1, -e * b2], -1), np.stack([-c, e * b2], -1),
                   np.stack([l1, z], -1)], 1)
    w = np.zeros_like(a3)
    np.divide(1.0, 8.0 * a3_cap, out=w, where=a3_cap > 1e-30)
    return np.stack([A1, A2], 1), w


# how far out the density term is asked to care about a single face: its error is the logarithm of
# the face's density over the island's target, saturated at this factor in both directions
DENSITY_CAP = 8.0
BOUNDARY_LOG_CAP = 2.5
BOUNDARY_WEIGHT_BASE = 0.12
BOUNDARY_WEIGHT_DENSITY = 0.42


def _local_symmetric_dirichlet_solve(p, tl, uv0, pin, maxit, density_weight,
                                     total_a3_override=None, density_target_override=None,
                                     majority_orientation=None, boundary_weight=None,
                                     boundary_edges_override=None,
                                     boundary_3d_length_override=None,
                                     boundary_count_override=None,
                                     scale_center_override=None):
    """Minimize true symmetric-Dirichlet energy over the free UV coordinates.

    Limited-memory BFGS leaves pinned coordinates untouched and rejects trial steps that invert a
    triangle. Folded or degenerate input falls back to the established fold-repair path.
    """
    p = np.asarray(p, dtype=float)
    tl = np.asarray(tl, dtype=np.int64)
    uv0 = np.asarray(uv0, dtype=float).copy()
    pin = np.asarray(pin, dtype=bool)
    point_count = len(uv0)
    face_count = len(tl)
    free_ids = np.flatnonzero(~pin)
    profile = {"model_ms": 0.0, "preconditioner_ms": 0.0,
               "cg_ms": 0.0, "line_search_ms": 0.0}
    if not len(free_ids) or not face_count:
        return None

    model_t0 = time.perf_counter()
    a3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if np.any(a3 <= 1e-30):
        return None
    total_a3 = float(a3.sum()) if total_a3_override is None else float(total_a3_override)
    if total_a3 <= 1e-30:
        return None
    prepped = face_metric_prep(p)
    signs = np.sign(signed_uv_area(uv0[tl]))
    nz = signs[signs != 0.0]
    majority = (1.0 if (nz > 0.0).sum() >= (nz < 0.0).sum() else -1.0
                ) if majority_orientation is None else float(majority_orientation)
    if np.any(signs != majority):
        return None

    face_weight = a3 / total_a3
    q = signed_uv_area(uv0[tl])
    density_target = (float(np.abs(q).sum()) / (2.0 * total_a3)
                      if density_target_override is None else float(density_target_override))
    if density_target <= 1e-30:
        return None
    density_weight = max(float(density_weight), 0.0)
    target_sqrt = np.sqrt(density_target)

    if boundary_edges_override is None:
        boundary_edges, boundary_lengths = boundary_geometry(p, tl)
        boundary_count = len(boundary_edges)
    else:
        boundary_edges = np.asarray(boundary_edges_override, dtype=np.int64)
        boundary_lengths = np.asarray(boundary_3d_length_override, dtype=float)
        boundary_count = (len(boundary_edges) if boundary_count_override is None
                          else int(boundary_count_override))
    if boundary_weight is None:
        boundary_weight = BOUNDARY_WEIGHT_BASE + BOUNDARY_WEIGHT_DENSITY * min(
            density_weight, 1.0)
    boundary_weight = max(float(boundary_weight), 0.0)
    boundary_target = target_sqrt * boundary_lengths
    boundary_weights = (np.full(len(boundary_edges), boundary_weight / boundary_count)
                        if len(boundary_edges) and boundary_count > 0
                        else np.zeros(len(boundary_edges)))
    def evaluate(u, need_gradient=True):
        uv_tri = u[tl]
        s1 = uv_tri[:, 1] - uv_tri[:, 0]
        s2 = uv_tri[:, 2] - uv_tri[:, 0]
        S = np.stack([np.stack([s1[:, 0], s2[:, 0]], axis=1),
                      np.stack([s1[:, 1], s2[:, 1]], axis=1)], axis=1)
        K = S @ prepped
        det = K[:, 0, 0] * K[:, 1, 1] - K[:, 0, 1] * K[:, 1, 0]
        oriented_det = majority * det
        if np.any(oriented_det <= max(density_target * 1e-12, 1e-30)):
            return np.inf, None
        inv = np.empty_like(K)
        inv[:, 0, 0] = K[:, 1, 1] / det
        inv[:, 1, 1] = K[:, 0, 0] / det
        inv[:, 0, 1] = -K[:, 0, 1] / det
        inv[:, 1, 0] = -K[:, 1, 0] / det
        energy_face = (K * K).sum(axis=(1, 2)) / density_target
        energy_face += density_target * (inv * inv).sum(axis=(1, 2))
        value = float(np.dot(face_weight, energy_face))

        grad_uv = None
        if need_gradient:
            grad_k = 2.0 * K / density_target
            grad_k -= (2.0 * density_target *
                       (inv.transpose(0, 2, 1) @ inv @ inv.transpose(0, 2, 1)))

            if density_weight > 0.0:
                log_ratio = np.log(np.maximum(oriented_det / density_target, 1e-30))
                cap = np.log(DENSITY_CAP)
                err = np.clip(log_ratio, -cap, cap)
                value += density_weight * float(np.dot(face_weight, err * err))
                # Match the existing robust density term: residuals saturate, while their
                # derivative keeps a bounded, useful direction for badly scaled faces.
                density_derivative = np.where(np.abs(log_ratio) < cap, err, 0.0)
                inv_t = inv.transpose(0, 2, 1)
                grad_k += (2.0 * density_weight * density_derivative)[:, None, None] * inv_t

            grad_s = grad_k @ prepped.transpose(0, 2, 1)
            grad_tri = np.empty((face_count, 3, 2), dtype=float)
            grad_tri[:, 1] = grad_s[:, :, 0]
            grad_tri[:, 2] = grad_s[:, :, 1]
            grad_tri[:, 0] = -grad_tri[:, 1] - grad_tri[:, 2]
            grad_uv = np.zeros((point_count, 2), dtype=float)
            np.add.at(grad_uv, tl.ravel(), (grad_tri * face_weight[:, None, None]).reshape(-1, 2))

        if len(boundary_edges):
            delta = u[boundary_edges[:, 1]] - u[boundary_edges[:, 0]]
            length2 = np.einsum("ei,ei->e", delta, delta)
            length = np.sqrt(np.maximum(length2, 1e-30))
            ratio = np.maximum(length / np.maximum(boundary_target, 1e-30), 1e-30)
            log_ratio = np.log(ratio)
            cap = BOUNDARY_LOG_CAP
            error = np.clip(log_ratio, -cap, cap)
            value += float(np.dot(boundary_weights, error * error))
            if need_gradient:
                derivative = np.where(np.abs(log_ratio) < cap, error, 0.0)
                edge_grad = (2.0 * boundary_weights * derivative / length2)[:, None] * delta
                np.add.at(grad_uv, boundary_edges[:, 0], -edge_grad)
                np.add.at(grad_uv, boundary_edges[:, 1], edge_grad)

        return value, grad_uv

    start_value, start_gradient = evaluate(uv0)
    if not np.isfinite(start_value) or start_gradient is None:
        return None
    profile["model_ms"] = (time.perf_counter() - model_t0) * 1000.0
    cg_t0 = time.perf_counter()
    x = uv0[free_ids].reshape(-1).copy()
    base = uv0.copy()
    loose = not bool(pin.any())
    if scale_center_override is not None:
        scale_center = np.asarray(scale_center_override, dtype=float).reshape(2)
    elif loose:
        scale_center = uv0.mean(axis=0)
    else:
        pinned_ids = np.flatnonzero(pin)
        scale_center = (uv0[pinned_ids].mean(axis=0) if len(pinned_ids)
                        else uv0.mean(axis=0))

    def project_similarity(v, state):
        """Remove whole-chart similarity motion from free-coordinate vectors."""
        q = state[free_ids] - scale_center
        scale = q.copy()
        if loose:
            basis = np.stack((
                np.broadcast_to((1.0, 0.0), q.shape),
                np.broadcast_to((0.0, 1.0), q.shape),
                np.stack((-q[:, 1], q[:, 0]), axis=1),
                q,
            ), axis=2).reshape(-1, 4)
        else:
            basis = scale.reshape(-1, 1)
        flat_basis = basis
        if not flat_basis.size:
            return v
        orthogonal, singular, _ = np.linalg.svd(flat_basis, full_matrices=False)
        if not len(singular) or singular[0] <= 1e-30:
            return v
        rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
        if not rank:
            return v
        directions = orthogonal[:, :rank]
        return v - directions @ (directions.T @ v)

    history_s = []
    history_y = []
    history_rho = []
    value = start_value
    gradient = project_similarity(start_gradient[free_ids].reshape(-1), base)
    initial_norm = float(np.linalg.norm(gradient))
    converged = initial_norm <= 1e-10
    iterations = 0
    max_iterations = max(1, int(maxit))
    memory = 8
    for iteration in range(max_iterations):
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm <= max(1e-7 * initial_norm, 1e-10):
            converged = True
            break

        direction = gradient.copy()
        alphas = []
        for s_vec, y_vec, rho in zip(reversed(history_s), reversed(history_y),
                                     reversed(history_rho)):
            alpha = rho * float(np.dot(s_vec, direction))
            alphas.append(alpha)
            direction -= alpha * y_vec
        if history_s:
            sy = float(np.dot(history_s[-1], history_y[-1]))
            yy = float(np.dot(history_y[-1], history_y[-1]))
            direction *= sy / max(yy, 1e-30)
        for (s_vec, y_vec, rho), alpha in zip(
                zip(history_s, history_y, history_rho), reversed(alphas)):
            beta = rho * float(np.dot(y_vec, direction))
            direction += s_vec * (alpha - beta)
        direction = -direction
        direction = project_similarity(direction, base)
        slope = float(np.dot(gradient, direction))
        if not np.isfinite(slope) or slope >= -1e-20:
            history_s.clear()
            history_y.clear()
            history_rho.clear()
            direction = -gradient
            slope = -float(np.dot(gradient, gradient))

        line_t0 = time.perf_counter()
        accepted = None
        step = 1.0
        for _ in range(24):
            trial = base.copy()
            trial[free_ids] = (x + step * direction).reshape(-1, 2)
            trial_value, trial_gradient = evaluate(trial)
            if (np.isfinite(trial_value) and
                    trial_value <= value + 1e-4 * step * slope):
                accepted = (trial, trial_value, trial_gradient, x + step * direction)
                break
            step *= 0.5
        profile["line_search_ms"] += (time.perf_counter() - line_t0) * 1000.0
        if accepted is None:
            break

        next_base, next_value, next_gradient_uv, next_x = accepted
        next_gradient = project_similarity(
            next_gradient_uv[free_ids].reshape(-1), next_base)
        s_vec = project_similarity(next_x - x, base)
        y_vec = next_gradient - gradient
        y_vec = project_similarity(y_vec, base)
        sy = float(np.dot(s_vec, y_vec))
        if sy > 1e-12 * np.linalg.norm(s_vec) * max(np.linalg.norm(y_vec), 1e-30):
            history_s.append(s_vec)
            history_y.append(y_vec)
            history_rho.append(1.0 / sy)
            if len(history_s) > memory:
                history_s.pop(0)
                history_y.pop(0)
                history_rho.pop(0)
        x = next_x
        base = next_base
        value = next_value
        gradient = next_gradient
        iterations = iteration + 1

    profile["cg_ms"] = (time.perf_counter() - cg_t0) * 1000.0
    if iterations == 0 and not converged:
        # A failed first line search is outside this optimizer's basin.  Let the established
        # fold-aware solver handle it instead of reporting a false no-op.
        return None
    if iterations == 0:
        return uv0.copy(), {
            "free": len(free_ids), "pass": 0, "solve": 1, "gain": 0.0,
            "why": "already even", "fold": 0, "ask": "open",
            "converged": converged, "fold_left": 0,
            "algorithm": "local_sd_lbfgs",
            "reinitialized": False, "capped_holes": 0, **profile,
        }
    final_value = value
    return base, {
        "free": len(free_ids), "pass": iterations, "solve": 1,
        "gain": start_value - final_value, "why": "relaxed", "fold": 0,
        "ask": "open", "converged": converged, "fold_left": 0,
        "algorithm": "local_sd_lbfgs",
        "reinitialized": False, "capped_holes": 0, **profile,
    }


def relax_solve(p, tl, uv0, pin, maxit=200, knee=2.0, density_weight=0.30,
                fold_rounds=6, density_refine=8, density_tol=0.3,
                density_face_weight=None, boundary_weight=None,
                total_a3_override=None, density_target_override=None,
                area2_override=None, spread_override=None,
                majority_orientation=None, boundary_edges_override=None,
                boundary_3d_length_override=None, boundary_count_override=None,
                scale_center_override=None, reinitialize=False,
                energy_mode="conformal", cap_holes=False, local_solver=False,
                whole_island_solver="CG"):
    """Bring one island to the evenest shape its boundary allows -- by solving for it.

    The conformal part of the energy this relaxes is a quadratic form in the UV coordinates:
    `sum_t w_t * (A_t . its six corners)**2` (see `stretch_terms`), whose coefficients depend only
    on the model and on which orientation each face is asked for, so its Hessian is the constant
    `2 sum_t w_t A_t^T A_t` and finding its bottom is a linear system.  That sum is divided by the
    island's own UV area, which is what makes it a number of the same kind as the density term below
    -- without it the trade-off between the two would be set by however large the author happened to
    pack the island, and one `density_weight` would mean a different balance on every island.  A
    triangle's UV area is quadratic in its corners rather than linear, so the density part is not, and
    it enters as its first-order expansion about the state the solve starts from -- a Gauss-Newton
    step, whose model matches the true energy's value and gradient there, re-expanded by the next solve
    about the state that one reached.  With the density weight at zero the objective is exactly
    quadratic and the first solve is already the global minimum of the problem, so `density_refine`
    costs nothing.

    The system is solved by preconditioned conjugate gradients, matrix free: Hessian times a vector
    is the same two `einsum` products the gradient used to be made of, and the diagonal of that same
    Hessian -- what the old Jacobi step divided by -- is the preconditioner.  Conjugate gradients
    rather than a dense Cholesky because the dense form costs the square of the island's point count
    in memory and time (30 MB for an island of 1900 points, and no room to grow), while this costs
    O(n) memory and its iteration count tracks the island's linear resolution.

    That replaces a walk which spent eight hundred passes, a tenth of a folded face each, crawling
    down a bowl it could have jumped to the bottom of.  It is why no tuning of the trust radius, the
    step count or the restarts ever bought more than about the same wall time per face: the cost
    was in the format of the iteration, not in its parameters.

    Points the caller names in `pin` -- those the selection left out, and those it holds because
    something else is matched to them -- are not a force here but Dirichlet conditions whose
    displacement is held at zero; which outline points belong in that set is the caller's business,
    and this module does not look at borders at all.  Held rather than softly pulled is what makes
    seam alignment a promise instead of a trend: a point no selected face pulls hardly moves, a
    pinned one does not move.  A wholly loose island keeps the four global similarity modes out of
    the Krylov directions, rather than holding any physical UV point; the packing it started with
    is put back after every accepted density round, and once more at the end for the rounds that
    carry no density term to hold it.

    Faces the wrong way round are not priced either.  A flip is not a place on the energy surface,
    it is a change of *which problem the island should be solved for*, so it is handled
    combinatorially, the way LSCM's own inversion step does: solve asking every face for the majority
    orientation, look at what the global solution still leaves inside out, pin the corners of those
    faces where the solution put them, and solve again.  Every round moves the whole island at once
    and each one is the exact answer to a convex problem, so a fold opens by as much as the rest of
    the island's least squares will pay for, rather than by as far as one Jacobi step can carry.
    Once a round comes back clean the pins stay on while the density settles, and the last solve
    releases them, so the opened fold is not left in the shape chosen for the round that opened it.

    What pinning costs is stated rather than hidden: a fold whose own boundary is folded too cannot
    be opened this way, because there is nothing left outside the pins to push from.  That is the
    non-disk case, and the answer to it is a seam, not a longer solve.

    p: (T,3,3) model corners, tl: (T,3) ids into uv0, uv0: (V,2), pin: (V,) bool.
    `maxit` is the conjugate-gradient cap of one solve, `knee` the point where the unevenness score
    stops squaring (see `shape`), `density_weight` the strength of the island-local texel-density
    term against the island's own shape term (see `density_error`), `fold_rounds` how many times the wrong-way-round pins may grow,
    `density_refine` how many
    Gauss-Newton re-expansions a clean island may ask for at most (a round whose step has gone too
    small to matter ends them sooner), and `density_tol` how much of a density round's gradient its
    solve has to remove -- the round only needs the direction, since the length is chosen afterwards
    on the true energy, and asking the solve for more than that costs iterations the step length
    then throws away.
    Returns (new_uv, stats) with stats["pass"] the CG iterations used, stats["solve"] how many solves
    that covers, stats["converged"] whether every one of them reached its residual tolerance instead
    of running out of iterations, stats["why"] ("relaxed", "already even", "nothing to move",
    "left alone"), stats["gain"] the drop of the solved energy, stats["fold"] how many faces were
    inside out to begin with, stats["fold_left"] how many still are in what is handed back, and
    stats["ask"] which of the two questions won: "open" asked for one orientation and worked to turn
    the folds back, "even" left them mirrored and evened them out instead."""
    output_V = len(uv0)
    V = output_V
    uv0 = np.asarray(uv0, dtype=float).copy()
    pin = np.asarray(pin, dtype=bool).copy()
    if local_solver and not reinitialize and not cap_holes:
        local_result = _local_symmetric_dirichlet_solve(
            p, tl, uv0, pin, maxit=maxit, density_weight=density_weight,
            total_a3_override=total_a3_override,
            density_target_override=density_target_override,
            majority_orientation=majority_orientation,
            boundary_weight=boundary_weight,
            boundary_edges_override=boundary_edges_override,
            boundary_3d_length_override=boundary_3d_length_override,
            boundary_count_override=boundary_count_override,
            scale_center_override=scale_center_override)
        if local_result is not None:
            return local_result
    virtual_holes = 0
    real_face_count = len(tl)
    if cap_holes and not pin.any():
        p, tl, uv0, virtual_holes = _cap_holes(p, tl, uv0)
        if virtual_holes:
            pin = np.concatenate([pin, np.zeros(virtual_holes, dtype=bool)])
            V = len(uv0)
    reinitialized = False
    if reinitialize and not np.asarray(pin, dtype=bool).any() and len(tl):
        # Rebuild only when the incoming chart is beyond what a local Newton/ARAP step can
        # sensibly repair.  Ordinary uneven UVs keep their authored layout and therefore their
        # useful placement in the editor.
        q_in = signed_uv_area(uv0[tl])
        nz = q_in[np.abs(q_in) > 1e-14 * max(float(np.abs(q_in).max()), 1e-30)]
        if len(nz):
            majority = 1.0 if int((nz > 0).sum()) >= int((nz < 0).sum()) else -1.0
            fold_ratio = float((nz * majority < 0.0).sum()) / len(nz)
            metric_shape, metric_det = face_metric(p, uv0[tl])
            severe = (fold_ratio > 0.08 or
                      float(np.percentile(metric_shape, 99)) > 25.0 or
                      float(np.percentile(metric_shape, 50)) > 4.0 or
                      float(np.std(np.log(np.maximum(metric_det, 1e-30)))) > 1.4)
            if severe:
                candidate = reinitialize_uv(p, tl, uv0)
                if np.isfinite(candidate).all():
                    uv0 = candidate
                    reinitialized = True
    profile = {"model_ms": 0.0, "preconditioner_ms": 0.0, "cg_ms": 0.0,
               "line_search_ms": 0.0}
    q0 = signed_uv_area(uv0[tl])
    sg = np.sign(q0)
    real_faces = np.arange(len(tl)) < real_face_count
    sg_real = sg[real_faces]
    maj = (1.0 if (sg_real > 0).sum() >= (sg_real < 0).sum() else -1.0
           ) if majority_orientation is None else float(majority_orientation)
    wrong = real_faces & (sg != maj) & (sg != 0.0)
    fold = bool(wrong.any())

    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    valid = a3 > 1e-30
    total_a3 = float(a3.sum()) if total_a3_override is None else float(total_a3_override)
    weight = a3 / total_a3 if total_a3 > 1e-30 else np.zeros_like(a3)
    density_weight = max(float(density_weight), 0.0)
    # Boundary protection is now a separate soft geometric metric.  Keep the old argument in the
    # signature so a reloaded operator from the previous version remains callable.
    # the density target is the island's own starting UV area per 3D area, so the term evens the
    # packing inside the island without asking the island to change size
    density_target = ((0.5 * float(np.abs(q0[valid]).sum()) / total_a3)
                      if total_a3 > 1e-30 else 0.0)
    if density_target_override is not None:
        density_target = float(density_target_override)
    use_d = density_weight > 0.0 and density_target > 1e-30
    area2_0 = float(np.abs(q0).sum()) if area2_override is None else float(area2_override)
    # the density residual is a ratio of areas and has no units, while each conformal term is
    # 1/(8*model area) times coefficients that carry model length squared: the model scale cancels
    # (which is why scaling his mesh changes nothing) and what is left is (UV length)^2, so the sum
    # grows as the SQUARE of however big the author packed the island.  Dividing it by that area is
    # what makes one Density Balance mean the same shape / evenness trade-off on a tightly packed
    # island and a loosely packed one: his mesh's islands span UV areas 0.04 to 0.36, so the single
    # slider value was really nine different ratios (.test/sweep_norm_balance.py).  An island
    # collapsed to a point has no area to divide by and no stretch to minimise either, so there the
    # weights stay as they are.
    conf_norm = 1.0 / area2_0 if area2_0 > 1e-30 else 1.0
    spread = (float(np.linalg.norm(uv0.max(0) - uv0.min(0)) + 1e-30)
              if spread_override is None else float(spread_override))

    prepped = face_metric_prep(p)
    symmetric_dirichlet = str(energy_mode).lower() in {
        "symmetric_dirichlet", "sd", "arap", "global_local"}

    def sd_scale(u):
        """IRLS weight for the inverse-stretch side of symmetric Dirichlet energy."""
        if not symmetric_dirichlet or density_target <= 1e-30:
            return np.ones(len(a3), dtype=float)
        shape_now, det = face_metric_uv(prepped, u[tl])
        ratio = np.clip(det / density_target, 1e-4, 1e4)
        anisotropy = np.clip(0.5 * (shape_now + 1.0 / np.maximum(shape_now, 1e-12)), 1.0, 64.0)
        # The square root keeps the reweighted quadratic well-conditioned while still making
        # collapsed, over-expanded, and highly anisotropic triangles expensive in the next global
        # step.  The conformal residual supplies the directional part; this factor supplies the
        # inverse-stretch side of symmetric Dirichlet.
        return np.sqrt(np.clip(0.5 * (ratio + 1.0 / ratio) * anisotropy, 1.0, 64.0))

    # Recover the outline from topology and measure it in 3D.  The target UV length uses the
    # island's starting average density, so a bad starting outline can move freely without making
    # the whole island change scale just to satisfy the boundary term.
    if boundary_edges_override is None:
        boundary_edges, boundary_3d_length = boundary_geometry(p, tl)
        boundary_count = len(boundary_edges)
    else:
        boundary_edges = np.asarray(boundary_edges_override, dtype=np.int64)
        boundary_3d_length = np.asarray(boundary_3d_length_override, dtype=float)
        boundary_count = (len(boundary_edges) if boundary_count_override is None
                          else int(boundary_count_override))
    boundary_target = np.sqrt(max(density_target, 1e-30)) * boundary_3d_length
    if boundary_weight is None:
        # Normalized below, so this does not depend on the number of outline edges.  Density
        # pressure raises the protection automatically; there is no user-facing B slider to tune.
        boundary_weight = BOUNDARY_WEIGHT_BASE + BOUNDARY_WEIGHT_DENSITY * min(density_weight, 1.0)
    boundary_weight = max(float(boundary_weight), 0.0)
    boundary_weights = (np.full(len(boundary_edges), boundary_weight / boundary_count)
                        if len(boundary_edges) else np.zeros(0, dtype=float))

    # Try L-BFGS only after the existing hole-cap and severe-chart repair steps produce a valid,
    # consistently oriented chart. Unsupported input falls through to the established CG path.
    if (str(whole_island_solver).upper() == "LBFGS" and symmetric_dirichlet and
            not local_solver and not pin.any() and not fold):
        lbfgs_result = _local_symmetric_dirichlet_solve(
            p, tl, uv0, pin, maxit=maxit, density_weight=density_weight,
            total_a3_override=total_a3_override,
            density_target_override=density_target_override,
            majority_orientation=maj,
            boundary_weight=boundary_weight,
            boundary_edges_override=boundary_edges,
            boundary_3d_length_override=boundary_3d_length,
            boundary_count_override=boundary_count,
            scale_center_override=scale_center_override)
        if lbfgs_result is not None:
            result_uv, result_stats = lbfgs_result
            # The symmetric-Dirichlet objective is scale-invariant. Preserve the authored
            # packing size so repeated relax operations do not slowly shrink or grow the island.
            start_area = float(np.abs(signed_uv_area(uv0[tl])).sum())
            result_area = float(np.abs(signed_uv_area(result_uv[tl])).sum())
            if start_area > 1e-30 and result_area > 1e-30:
                center = result_uv.mean(axis=0)
                result_uv = center + (result_uv - center) * np.sqrt(start_area / result_area)
            result_stats.update({
                "algorithm": "whole_island_sd_lbfgs",
                "free": int((~pin[:output_V]).sum()),
                "fold": int(wrong.sum()),
                "fold_left": 0,
                "reinitialized": reinitialized,
                "capped_holes": int(virtual_holes),
            })
            return result_uv[:output_V].copy(), result_stats

    def shape(u):
        """How uneven the island is, the way the overlay sees it: every face's stretch above 1.0
        counted by square up to the knee, then by plain amount, so one ruined face cannot outvote
        the six hundred ordinary ones around it.  Blind to which way round a face lies, which is
        why every comparison below counts the folds first and reaches shape only to break that
        tie."""
        z = np.maximum(face_metric_uv(prepped, u[tl])[0] - 1.0, 0.0)
        z = z * z if knee is None else np.where(z <= knee, z * z, 2.0 * knee * z - knee * knee)
        return float(z.mean())

    flat = tl.ravel()

    def scatter(gs):
        """Sum the per-face corner contributions (T,3,2) onto the island's points."""
        out = np.empty((V, 2))
        for k in (0, 1):
            out[:, k] = np.bincount(flat, weights=gs[:, :, k].ravel(), minlength=V)
        return out

    def scatter_edges(gs):
        """Sum boundary-edge endpoint contributions (E,2,2) onto island points."""
        out = np.zeros((V, 2))
        if len(boundary_edges):
            np.add.at(out, boundary_edges[:, 0], gs[:, 0])
            np.add.at(out, boundary_edges[:, 1], gs[:, 1])
        return out

    def scatter_blocks(gs):
        """Sum per-face 2x2 corner blocks (T,3,2,2) onto island points."""
        out = np.zeros((V, 2, 2))
        np.add.at(out, tl.ravel(), gs.reshape(-1, 2, 2))
        return out

    def scatter_edge_blocks(gs):
        """Sum per-boundary-edge 2x2 endpoint blocks (E,2,2,2) onto island points."""
        out = np.zeros((V, 2, 2))
        if len(boundary_edges):
            ids = boundary_edges.ravel()
            np.add.at(out, ids, gs.reshape(-1, 2, 2))
        return out

    ask_cache = {}

    def ask(e_side):
        """An orientation trial's fixed coefficients: every face asked for `e_side` (+1, -1) or for
        the way round it already is (None)."""
        if e_side not in ask_cache:
            A, w = stretch_terms(p, uv0[tl], e_side=e_side)
            w = w * conf_norm
            if e_side is None:
                signs = np.where(sg == 0.0, 1.0, sg)
            else:
                signs = np.full(len(q0), float(e_side))
            ask_cache[e_side] = (A, w, signs)
        return ask_cache[e_side]

    def area_grad(u):
        """d(UV double area)/d(each corner), (T,3,2) -- linear in the corners, because the area
        itself is quadratic in them."""
        t = u[tl]
        t0, t1, t2 = t[:, 0], t[:, 1], t[:, 2]
        return np.stack([
            np.stack([t1[:, 1] - t2[:, 1], t2[:, 0] - t1[:, 0]], 1),
            np.stack([t2[:, 1] - t0[:, 1], t0[:, 0] - t2[:, 0]], 1),
            np.stack([t0[:, 1] - t1[:, 1], t1[:, 0] - t0[:, 0]], 1),
        ], 1)

    def density_error(signs, u):
        """How far each face sits from the island's own texel density, as the logarithm of the ratio
        of the two, saturated at DENSITY_CAP either way.

        The logarithm is what keeps the term from needling an island.  `ratio - 1` charges a face
        five times too large 25x what it charges one five times too small, so the rounds buy their
        fall by flattening the loudest outlier: on a 1872-face island whose outline wedge started at
        38.4 deg the convex stage left faces at x5.36 and x0.27, and the density stage collapsed the
        wedge to 7.2 deg to even the big one.  `.test/probe_needle_energy.py` scores that needle
        0.00759 against 0.0494 for the state that kept the wedge, so the linear error did mean it --
        but the same probe says the needle was a basin, not a verdict -- the state the logarithm
        settles on (wedge 18.6 deg) is cheaper under BOTH errors.  Faces near the target are hardly
        touched either, since `log(ratio)` and `ratio - 1` share a slope at ratio 1; what changes is
        that crushing a shape no longer pays for an outlier, and DENSITY_CAP stops one ruined face
        outvoting the six hundred tight ones around it.
        """
        rat = np.ones_like(a3)
        np.divide(signs * signed_uv_area(u[tl]), 2.0 * density_target * a3, out=rat, where=valid)
        err = np.log(np.clip(rat, 1.0 / DENSITY_CAP, DENSITY_CAP))
        err[~valid] = 0.0
        return err

    def density_lin(signs, base):
        """First-order expansion of that error at `base`, as the affine function `L . u - r` whose
        value agrees with the error there.  The area is quadratic in the corners, so what the
        expansion leaves out grows with the square of how far the step goes -- which is why each
        round takes the length the true energy will vouch for, not the one the model asks for."""
        err = density_error(signs, base)
        L = np.zeros((len(a3), 3, 2))
        np.divide(signs[:, None, None] * area_grad(base),
                  (2.0 * density_target * a3)[:, None, None], out=L, where=valid[:, None, None])
        # d(log ratio) = d(ratio) / ratio.  The divisor is the saturated ratio rather than the true
        # one, so a face beyond the cap keeps a small slope instead of leaving the model flat there
        L = L / np.exp(err)[:, None, None]
        return L, np.einsum("tcv,tcv->t", L, base[tl]) - err

    def boundary_error(u):
        """Robust log edge-length residual against the 3D metric target."""
        if not len(boundary_edges):
            return np.zeros(0, dtype=float)
        d = u[boundary_edges[:, 1]] - u[boundary_edges[:, 0]]
        length = np.linalg.norm(d, axis=1)
        ratio = np.ones_like(length)
        np.divide(length, boundary_target, out=ratio, where=boundary_target > 1e-30)
        return np.log(np.clip(ratio, np.exp(-BOUNDARY_LOG_CAP), np.exp(BOUNDARY_LOG_CAP)))

    def boundary_lin(base):
        """Linearize log(|uv edge| / target) at the current UV state."""
        if not len(boundary_edges):
            return np.zeros((0, 2, 2)), np.zeros(0, dtype=float)
        d = base[boundary_edges[:, 1]] - base[boundary_edges[:, 0]]
        length2 = np.einsum("ei,ei->e", d, d)
        length = np.sqrt(np.maximum(length2, 1e-30))
        err = boundary_error(base)
        L = np.zeros((len(boundary_edges), 2, 2), dtype=float)
        # d(log |b-a|) = (b-a) . (db-da) / |b-a|^2
        L[:, 0] = -d / length2[:, None]
        L[:, 1] = d / length2[:, None]
        return L, np.einsum("eij,eij->e", L, base[boundary_edges]) - err

    conformal_cache = {}

    def system(A, w, signs, base, dens, boundary_factor=1.0):
        """The quadratic that stands in for the objective at `base`: Hessian-times-vector, the
        Hessian's diagonal, the constant part of its gradient, and a function that scores any state
        by this same quadratic.  `dens` False leaves the density term out, which makes it the
        conformal energy exactly -- the problem the fold rounds want, because that one is convex and
        its bottom is the whole island's answer.  With the density term in, the scoring function is
        what a step promises, the true `energy` being what the step is judged on."""
        model_t0 = time.perf_counter()
        lin = None
        wd = None
        blin = None
        wb = None
        boundary_diag = np.zeros((V, 2))

        def hvp(v):
            face_v = v[tl]
            d = np.einsum("tkcv,tcv->tk", A, face_v)
            gs = np.einsum("tkcv,tk->tcv", A, d) * (2.0 * w)[:, None, None]
            if lin is None:
                out = scatter(gs)
            else:
                L, _ = lin
                out = scatter(gs + (2.0 * wd * np.einsum("tcv,tcv->t", L, face_v))[:, None, None] * L)
            if blin is not None:
                Lb, _ = blin
                db = np.einsum("eij,eij->e", Lb, v[boundary_edges])
                out = out + scatter_edges((2.0 * wb * db)[:, None, None] * Lb)
            return out

        cache_key = (id(A), id(w))
        if cache_key not in conformal_cache:
            conformal_blocks = 2.0 * w[:, None, None, None] * np.einsum(
                "tkci,tkcj->tcij", A, A)
            conformal_cache[cache_key] = (
                conformal_blocks, np.diagonal(conformal_blocks, axis1=2, axis2=3))
        conformal_blocks, conformal_diag = conformal_cache[cache_key]
        face_blocks = conformal_blocks.copy() if dens else conformal_blocks
        dg = conformal_diag.copy() if dens else conformal_diag
        c0 = np.zeros((V, 2))
        if dens:
            L, r = density_lin(signs, base)
            wd = density_weight * weight
            lin = (L, r)
            c0 = scatter((-2.0 * wd * r)[:, None, None] * L)
            density_blocks = (2.0 * wd)[:, None, None, None] * np.einsum(
                "tci,tcj->tcij", L, L)
            face_blocks = face_blocks + density_blocks
            dg = dg + np.diagonal(density_blocks, axis1=2, axis2=3)

        if len(boundary_edges):
            Lb, rb = boundary_lin(base)
            wb = boundary_weights * boundary_factor
            blin = (Lb, rb)
            c0 = c0 + scatter_edges((-2.0 * wb * rb)[:, None, None] * Lb)
            boundary_blocks = (2.0 * wb)[:, None, None, None] * np.einsum(
                "eai,eaj->eaij", Lb, Lb)
            dg_b = np.diagonal(boundary_blocks, axis1=2, axis2=3)
            boundary_diag = scatter_edges(dg_b)
        else:
            boundary_blocks = np.zeros((0, 2, 2, 2))

        def value(u):
            d = np.einsum("tkcv,tcv->tk", A, u[tl])
            e = float((w[:, None] * d * d).sum())
            if lin is not None:
                L, off = lin
                e += float((wd * (np.einsum("tcv,tcv->t", L, u[tl]) - off) ** 2).sum())
            if blin is not None:
                Lb, off_b = blin
                e += float((wb * (np.einsum("eij,eij->e", Lb, u[boundary_edges]) - off_b) ** 2).sum())
            return e

        block_diag = scatter_blocks(face_blocks) + scatter_edge_blocks(boundary_blocks)
        diag = scatter(dg) + boundary_diag
        profile["model_ms"] += (time.perf_counter() - model_t0) * 1000.0
        return hvp, diag, c0, value, block_diag

    def energy(A, w, signs, u, dens):
        """The objective itself rather than its local stand-in, so successive Gauss-Newton steps can
        be compared on the thing they are trying to reduce."""
        d = np.einsum("tkcv,tcv->tk", A, u[tl])
        e = float((w[:, None] * d * d).sum())
        if dens:
            err = density_error(signs, u)
            e += density_weight * float((weight * err * err).sum())
        if len(boundary_edges):
            e += float((boundary_weights * boundary_error(u) ** 2).sum())
        return e

    def objective(A, w, signs, u, dens):
        """Evaluate the current IRLS objective, including its symmetric-Dirichlet weights."""
        return energy(A, w * sd_scale(u), signs, u, dens)

    def inverted(u):
        """Faces of the island lying the way round the majority is not, in the state u."""
        s = np.sign(signed_uv_area(u[tl]))
        return real_faces & (s != maj) & (s != 0.0)

    def inside_out(u):
        return int(inverted(u).sum())

    loose = not bool(pin.any())
    # A loose island has a four-dimensional similarity nullspace.  These are global gauge freedoms,
    # not boundary conditions: fixing two real UV vertices turns them into visible corners.  The
    # solve below projects its directions away from translation, rotation and scale instead, so every
    # island vertex remains free to move.
    gauge = None
    scale_gauge = None
    if loose:
        cen = uv0.mean(0)
        q = uv0 - cen
        gauge = np.stack([
            np.broadcast_to(np.array([1.0, 0.0]), (V, 2)),
            np.broadcast_to(np.array([0.0, 1.0]), (V, 2)),
            np.stack([-q[:, 1], q[:, 0]], axis=1),
            q,
        ], axis=2)
    else:
        # A partial solve has real Dirichlet points, so translation and rotation are already
        # determined.  Its remaining failure mode is a uniform contraction of the free region
        # towards the held part; remove only that one scale direction, leaving local shape changes
        # unconstrained.
        scale_center = uv0.mean(0) if scale_center_override is None else np.asarray(
            scale_center_override, dtype=float)
        scale_gauge = (uv0 - scale_center)[:, :, None]
    # which points may move is worked out once, before any solve, and shared by both questions below:
    # a point no live triangle sits on has no stiffness and nothing to be even about
    _, stiff, _, _, _ = system(*ask(maj), uv0, False)
    held = pin | (stiff.min(1) <= 0.0)

    stats = {"free": int((~held).sum()), "pass": 0, "solve": 0, "gain": 0.0, "why": "relaxed",
             "fold": int(wrong.sum()), "ask": None, "converged": True, "fold_left": 0,
             "algorithm": ("local_fallback_sd_cg" if local_solver else
                           "whole_island_sd_cg" if symmetric_dirichlet else
                           "whole_island_conformal_cg"),
             "reinitialized": reinitialized, "capped_holes": int(virtual_holes), **profile}
    if held.all():
        stats["why"] = "nothing to move"
        stats.update(profile)
        return uv0[:output_V].copy(), stats
    flip0 = inside_out(uv0)

    def restore(v):
        """Put the island's packing back at the area it started with, about its own centre.  The
        energy is as small for a shrunken island as for an even one, and with nothing held there is
        nothing else to settle its scale.  A uniform positive scaling turns no face over."""
        now = float(np.abs(signed_uv_area(v[tl])).sum())
        if now <= 1e-18:
            return v
        cen = v.mean(0)
        return (v - cen) * np.sqrt(area2_0 / now) + cen

    projection_cache = {}

    def solve(A, w, signs, clamp, base, dens, tol=1e-6, model=None):
        """One global move over every point not held by selection or a fold clamp.

        On a loose island, each Krylov vector is projected off the four global similarity modes.
        This fixes the numerical gauge without fixing any physical boundary vertex.
        """
        if model is None:
            model = system(A, w, signs, base, dens)
        hvp, diag, c0, _, block_diag = model
        live = (~held & ~clamp)[:, None]
        active = live[:, 0]
        preconditioner_t0 = time.perf_counter()
        basis = None
        if gauge is not None and not clamp.any():
            basis = gauge
        elif scale_gauge is not None:
            basis = scale_gauge

        # Only the active set and the fixed gauge basis determine this projection. Reuse its
        # indices and QR across density rounds; a different fold clamp gets a separate entry.
        projection_key = (id(basis), active.tobytes())
        if projection_key not in projection_cache:
            active_ids = np.flatnonzero(active)
            inactive_ids = np.flatnonzero(~active)
            projector = None
            if basis is not None and len(active_ids):
                M = basis[active_ids].reshape(-1, basis.shape[2])
                projector, _ = np.linalg.qr(M, mode="reduced")
            projection_cache[projection_key] = (active_ids, inactive_ids, projector)
        active_ids, inactive_ids, projector = projection_cache[projection_key]

        def project(v):
            if projector is None:
                return v * live
            out = np.zeros_like(v)
            flat = v[active_ids].reshape(-1)
            if len(flat):
                out[active_ids] = (flat - projector @ (projector.T @ flat)).reshape(-1, 2)
            return out

        inv_diag = live / np.maximum(diag, 1e-30)
        block_inv = np.zeros_like(block_diag)
        if len(active_ids):
            blocks = block_diag[active_ids]
            scale = np.maximum(np.max(np.abs(blocks), axis=(1, 2)), 1e-30)
            det = blocks[:, 0, 0] * blocks[:, 1, 1] - blocks[:, 0, 1] * blocks[:, 1, 0]
            good = det > 1e-12 * scale * scale
            if good.any():
                selected = blocks[good]
                inv_det = 1.0 / det[good]
                inverse = np.empty_like(selected)
                inverse[:, 0, 0] = selected[:, 1, 1] * inv_det
                inverse[:, 1, 1] = selected[:, 0, 0] * inv_det
                inverse[:, 0, 1] = -selected[:, 0, 1] * inv_det
                inverse[:, 1, 0] = -selected[:, 1, 0] * inv_det
                block_inv[active_ids[good]] = inverse
        fallback = active & (np.max(np.abs(block_inv), axis=(1, 2)) <= 0.0)
        fallback_ids = np.flatnonzero(fallback)

        def precondition(v):
            """Apply a per-UV-point 2x2 Jacobi block, with scalar fallback for flat points."""
            out = np.einsum("vij,vj->vi", block_inv, v)
            if len(fallback_ids):
                out[fallback_ids] = v[fallback_ids] * inv_diag[fallback_ids]
            out[inactive_ids] = 0.0
            return out

        profile["preconditioner_ms"] += (time.perf_counter() - preconditioner_t0) * 1000.0

        cg_t0 = time.perf_counter()
        g = project(hvp(base) + c0)
        g0 = float(np.linalg.norm(g))
        r = project(-g)
        # what the model says is left to win; when that is nothing the island is already standing
        # where this solve would put it, and there is no iteration worth spending
        if 0.5 * float((r * precondition(r)).sum()) <= 1e-12 * max(
                energy(A, w, signs, base, dens), 1e-30):
            profile["cg_ms"] += (time.perf_counter() - cg_t0) * 1000.0
            return base.copy(), 0, False, np.zeros((V, 2))
        x = np.zeros((V, 2))
        z = project(precondition(r))
        dirn = z.copy()
        rz = float((r * z).sum())
        cap = max(1, int(maxit))
        iters = 0
        out = True
        for iteration in range(1, cap + 1):
            hp = project(hvp(dirn))
            ph = float((dirn * hp).sum())
            if ph <= 1e-30 or rz <= 1e-30:
                out = False
                break
            step = rz / ph
            x = project(x + step * dirn)
            r = project(r - step * hp)
            iters = iteration
            # the recycled residual drifts away from the true one once the Krylov space has grown
            # long, and a drift nobody catches says "converged" about a state that is still moving
            rn = float(np.linalg.norm(r))
            check = rn
            if iteration % 20 == 0 or rn <= tol * g0:
                rt = project(-(hvp(base + x) + c0))
                check = float(np.linalg.norm(rt))
                if abs(check - rn) > 0.5 * check:
                    r = rt
                    z = project(precondition(r))
                    dirn = z.copy()
                    rz = float((r * z).sum())
                    continue
            if check <= tol * g0:
                out = False
                break
            z = project(precondition(r))
            rz_new = float((r * z).sum())
            dirn = project(z + (rz_new / rz) * dirn)
            rz = rz_new
        profile["cg_ms"] += (time.perf_counter() - cg_t0) * 1000.0
        return base + x, iters, out, x

    def run(e_side, rounds, pin_folds):
        """Ask the island the `e_side` question, and keep the state worth handing back: fewest faces
        inside out, and the lowest objective among those.

        Two stages, because the two terms want different problems.  The first solves the convex
        conformal quadratic and grows the set of pinned wrong-way-round faces between solves; the
        density term is left out of it, since its linearization is only good near the state it was
        taken at and a fold has to travel through the flat state to open, which a model taken at the
        folded state does not ask for.  Measured on a grid island of 2178 faces with 196 of them
        inside out and a block of them stamped down to a twenty-fifth of their size: the convex
        rounds cleared all 196 in one solve and left no pair of faces over each other.  The second
        stage brings the density back as Gauss-Newton rounds about the opened island."""
        A, w, signs = ask(e_side)
        best = [None]
        base = uv0
        clamp = np.zeros(V, bool)
        count = [0, 0, False]

        def note(iters, hit):
            count[0] += iters
            count[1] += 1
            count[2] = count[2] or hit

        def keep(cand):
            # the objective the solves are actually minimising, folds first: `shape` is blind to
            # density, and keying on it threw every density state the island reached away
            key = (inside_out(cand), objective(A, w, signs, cand, True))
            if best[0] is None or key < best[0][0]:
                best[0] = (key, cand, count[0], count[1], count[2])
            return key

        def step(clamp_at, base_at, dens):
            # Symmetric Dirichlet is handled as a short global/local IRLS loop.  The local step
            # raises the weight of collapsed or over-expanded faces, then the global CG solve moves
            # the UVs under that updated metric.  The loop is deliberately short because density
            # polishing below already re-expands the model at the accepted state.
            current = base_at
            loops = 3 if symmetric_dirichlet else 1
            for _ in range(loops):
                local_w = w * sd_scale(current)
                cand, iters, hit, direction = solve(
                    A, local_w, signs, clamp_at, current, dens)
                note(iters, hit)
                if iters == 0:
                    break
                search_t0 = time.perf_counter()
                old_fold = inside_out(current)
                old_energy = objective(A, w, signs, current, dens)
                accepted = None
                for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    trial = current + fraction * direction
                    if loose:
                        trial = restore(trial)
                    if (inside_out(trial) <= old_fold and
                            objective(A, w, signs, trial, dens) <= old_energy + 1e-12 * max(old_energy, 1.0)):
                        accepted = trial
                        break
                profile["line_search_ms"] += (time.perf_counter() - search_t0) * 1000.0
                if accepted is None:
                    break
                current = accepted
                keep(current)
                if float(np.abs(fraction * direction).max()) <= 1e-7 * spread:
                    break
            return current

        def polish(base_at, clamp_at):
            """Even the packing out, one Gauss-Newton round at a time about where the last round
            left the island.

            The density model is only good near where it was taken: a face's UV area is quadratic in
            its corners, so its affine model goes on promising an area the face can only reach by
            turning itself over.  So each round solves for one direction -- no damping term, only its
            length -- and tries it at t = 1, 1/2, 1/4..., keeping the first t that turns no face over
            and whose fall in the TRUE objective is at least a tenth of what the model's own slope
            promises for that length.  The slope is read off the model at three points of the ray
            rather than taken as the fall at t = 1, because the round's solve is deliberately inexact
            and the parabola's promise is only the model's own at the exact minimiser; on the islands
            measured so far the two tests pick the same length, so that is a correctness argument
            rather than a speed one.  Descent is then guaranteed by construction rather than by hoping
            the linearization reached, and a round costs one conjugate-gradient solve."""
            base = base_at
            for _ in range(density_refine):
                local_w = w * sd_scale(base)
                model = system(A, local_w, signs, base, True)
                value = model[3]
                cand, iters, hit, x = solve(
                    A, local_w, signs, clamp_at, base, True, tol=density_tol, model=model)
                note(iters, hit)
                if iters == 0:
                    break
                line_search_t0 = time.perf_counter()
                e_last = objective(A, w, signs, base, True)
                # m(t) = e_last + a t + b t^2 along the ray, read off the model at t = 1/2 and t = 1
                slope = 4.0 * (value(base + 0.5 * x) - e_last) - (value(cand) - e_last)
                fold_last = inside_out(base)
                moved = None
                for t in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    u = base + t * x
                    if loose:
                        u = restore(u)
                    if (inside_out(u) <= fold_last
                            and objective(A, w, signs, u, True) <= e_last + 0.1 * t * slope):
                        moved = u - base
                        base = u
                        keep(base)
                        break
                profile["line_search_ms"] += (time.perf_counter() - line_search_t0) * 1000.0
                if moved is None or float(np.abs(moved).max()) <= 1e-7 * spread:
                    break
            return base

        if pin_folds:
            while rounds > 0:
                rounds -= 1
                cand = step(clamp, base, False)
                bad = inverted(cand)
                base = cand
                if not bad.any():
                    break
                nxt = clamp.copy()
                nxt[tl[bad].ravel()] = True
                # a fold that is already pinned, or an island with nothing left outside the pins,
                # has no further pinning to do: that is the non-disk case and it is reported, not
                # fought
                if (nxt == clamp).all() or int((~held & ~nxt).sum()) < 3:
                    break
                clamp = nxt
        if not pin_folds:
            # the mirrored question gets no fold rounds: one convex solve is its whole answer
            base = step(clamp, base, False)
        if use_d:
            base = polish(base, clamp)
        if pin_folds and clamp.any():
            # the pins have done their job pushing the good material into the room the fold needs;
            # let go of them and see whether the island settles even with the fold opened
            base = step(np.zeros(V, bool), base, False)
            if use_d:
                base = polish(base, np.zeros(V, bool))
        return best[0][1], best[0][2], best[0][3], best[0][4], \
            objective(A, w, signs, uv0, True) - objective(A, w, signs, best[0][1], True)

    at, used, nsolve, out, gain = run(maj if fold else None, fold_rounds, True)

    def good(f, drop):
        """Worth handing back at all: fewer faces folded over than the island started with, or as
        many and a lower objective.  `drop` is the objective's fall under the question that trial
        asked, so each trial is judged against its own start -- the two questions are different
        problems and their energies do not compare.  Shape alone would rather have a fold flattened
        on top of its neighbours than opened, and it is blind to density."""
        return f < flip0 or (f == flip0 and drop > 1e-15)

    trials = [("open", at, used, nsolve, out, gain)]
    if fold and (inside_out(at) > 0 or not good(inside_out(at), gain)):
        # the mirrored-flat question is one solve, not a second campaign: it says what the island
        # looks like if the folds are left lying where they are and merely evened out, which on an
        # overlap that pinning cannot open is the better answer
        trials.append(("even",) + run(None, 1, False))
    win = None
    for name, t_at, t_used, t_solves, t_out, t_gain in trials:
        if not good(inside_out(t_at), t_gain):
            continue
        rank = (inside_out(t_at), shape(t_at))
        if win is None or rank < win[0]:
            win = (rank, name, t_at, t_used, t_solves, t_out, t_gain)
    if win is None:
        stats["ask"] = "open"
        stats["why"] = "left alone" if fold else "already even"
        stats.update(profile)
        return uv0[:output_V].copy(), stats
    _, stats["ask"], at, stats["pass"], stats["solve"], out, stats["gain"] = win
    if loose:
        # a no-op for anything the density rounds touched -- they keep the area themselves -- but
        # the density-free answers need their scale put back
        at = restore(at)
    if stats.get("algorithm") != "whole_island_sd_lbfgs":
        stats["converged"] = not out
    stats["fold_left"] = inside_out(at)
    stats.update(profile)
    return at[:output_V].copy(), stats


def _refresh_overlap_cache(cache, tl, u, lab, moved):
    """Update cached face data without rebuilding unchanged UV geometry."""
    profile = {name: cache.get(name, 0.0) for name in
               ("refresh_ms", "candidate_ms", "sat_ms", "candidate_count",
                "candidate_mode")}
    key = (_array_digest(tl), _array_digest(lab) if lab is not None else None)
    spatial = _OVERLAP_INDEX_CACHE.get(key)
    current_uv = u[tl]
    moved_ids = (np.arange(len(tl), dtype=np.int64) if moved is None
                 else np.flatnonzero(moved))
    if spatial is None:
        area2 = signed_uv_area(current_uv)
        box = np.stack([current_uv.min(1), current_uv.max(1)], 1)
        edge = np.stack([current_uv[:, 1] - current_uv[:, 0],
                         current_uv[:, 2] - current_uv[:, 1],
                         current_uv[:, 0] - current_uv[:, 2]], 1)
        normals = np.stack([-edge[..., 1], edge[..., 0]], -1)
        spatial = {"box": box, "area2": area2, "normals": normals,
                   "scale": np.sqrt(2.0 * np.abs(area2) / 1.7320508) + 1e-30,
                   "previous_uv": current_uv.copy(), "moved_ids": moved_ids}
        _OVERLAP_INDEX_CACHE[key] = spatial
        while len(_OVERLAP_INDEX_CACHE) > 1:
            _OVERLAP_INDEX_CACHE.pop(next(iter(_OVERLAP_INDEX_CACHE)))
    else:
        changed = np.flatnonzero(np.any(current_uv != spatial["previous_uv"], axis=(1, 2)))
        if len(changed):
            t = current_uv[changed]
            area2 = signed_uv_area(t)
            box = np.stack([t.min(1), t.max(1)], 1)
            edge = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1],
                             t[:, 0] - t[:, 2]], 1)
            spatial["area2"][changed] = area2
            spatial["box"][changed] = box
            spatial["normals"][changed] = np.stack([-edge[..., 1], edge[..., 0]], -1)
            spatial["scale"][changed] = np.sqrt(2.0 * np.abs(area2) / 1.7320508) + 1e-30
            spatial["previous_uv"][changed] = t
        spatial["moved_ids"] = moved_ids
    cache.clear()
    cache.update(spatial)
    cache.update(profile)
    cache.setdefault("candidate_mode", "skipped")


def overlapping_pairs(tl, u, lab=None, moved=None, window=200, block=65536,
                      return_pairs=False, overlap_cache=None):
    """How many pairs of these faces lie on top of each other in UV.

    The relaxed energy cannot see this: it asks each face to be a good similarity of itself and is
    satisfied by two faces sharing one patch of the map, which is why an island can come back with
    every face the right way round and still have material stamped over material.  Overlap is a
    property of pairs, so it is measured on pairs and reported, rather than pulled into the
    objective, where it would need a barrier of its own and a gradient for it.

    Two triangles overlap when no edge normal of either separates them -- the separating-axis test,
    which never has to build the intersection polygon.  A pair counts only where the two overlap by
    more than a twentieth of the smaller face's own edge on *every* one of those six axes, so the
    way a conformal triangulation meets along a shared edge does not count against the island, and
    neither does a block of faces that was itself stamped down to a hundredth of its size.

    Face boxes and SAT axes stay cached between overlap rounds. Dense movement uses a vectorized
    x-sweep; sparse movement sweeps the moving faces and checks them against the stationary boxes.
    Both paths feed the same SAT test in blocks.

    tl: (T,3) point ids, u: (V,2) their UV coordinates, lab: (T,) island label per face, or None to
    compare every face in the batch with every other.  `moved`: (T,) bool, which faces this relax was
    allowed to move -- given, only pairs with a moved face in them are counted, and they come back
    split two ways, because an island stamped over itself and an island that grew over a neighbour
    are different problems: the first is opened by a seam, the second by the packer.
    Returns (pairs within one island, pairs across islands); with `return_pairs`, also returns
    the matching face-index pairs, with the within-island pairs first."""
    if overlap_cache is None:
        overlap_cache = {}
    refresh_t0 = time.perf_counter()
    _refresh_overlap_cache(overlap_cache, tl, u, lab, moved)
    overlap_cache["refresh_ms"] += (time.perf_counter() - refresh_t0) * 1000.0
    overlap_cache["candidate_mode"] = "skipped"
    area2_all = overlap_cache["area2"]
    box_all = overlap_cache["box"]
    edge_normals_all = overlap_cache["normals"]
    scale_all = overlap_cache["scale"]
    area2 = area2_all
    biggest = float(np.abs(area2).max()) if len(area2) else 0.0
    area_threshold = 1e-12 * max(biggest, 1e-30)
    valid = np.abs(area2) > area_threshold
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size < 2:
        empty = np.zeros((0, 2), np.int64)
        return (0, 0, empty) if return_pairs else (0, 0)
    face_labels_all = np.zeros(len(tl), np.int64) if lab is None else np.asarray(lab)
    moved_all = None if moved is None else np.asarray(moved, dtype=bool)
    f = valid_ids
    n = len(f)
    box = box_all[f]
    edge_normals = edge_normals_all[f]
    scale = scale_all[f]
    face_labels = face_labels_all[f]
    mv = None if moved_all is None else moved_all[f]

    def sweep_candidates(indices):
        """Build the bounded x-sweep for a compact set of valid faces."""
        if len(indices) < 2:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        order = indices[np.argsort(box[indices, 0, 0])]
        left = box[order, 0, 0]
        ends = np.searchsorted(left, box[order, 1, 0], side="right")
        counts = np.clip(ends - np.arange(len(indices)) - 1, 0, window)
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        starts = np.cumsum(counts) - counts
        ii = np.repeat(np.arange(len(indices)), counts)
        jj = ii + 1 + (np.arange(total) - np.repeat(starts, counts))
        return order[ii], order[jj]

    candidate_t0 = time.perf_counter()
    # The all-face query is a single vectorized operation and wins once movement is dense.
    dense = mv is None or int(mv.sum()) >= max(2, n // 2)
    overlap_cache["candidate_mode"] = "dense" if dense else "sparse"
    if dense:
        a, b = sweep_candidates(np.arange(n, dtype=np.int64))
    else:
        moving = np.flatnonzero(mv)
        stationary = np.flatnonzero(~mv)
        aa, bb = sweep_candidates(moving)
        cross = []
        if len(stationary):
            static_order = stationary[np.argsort(box[stationary, 0, 0])]
            static_left = box[static_order, 0, 0]
            for face in moving.tolist():
                start = int(np.searchsorted(static_left, box[face, 0, 0], side="left"))
                end = int(np.searchsorted(static_left, box[face, 1, 0], side="right"))
                lo = max(0, start - window)
                cand = static_order[lo:min(end, lo + window)]
                if len(cand):
                    cand = cand[box[cand, 1, 0] > box[face, 0, 0]]
                if len(cand):
                    cross.append(np.stack([np.full(len(cand), face, np.int64), cand], 1))
        if cross:
            cross = np.concatenate(cross, axis=0)
            a = np.concatenate([aa, cross[:, 0]])
            b = np.concatenate([bb, cross[:, 1]])
        else:
            a, b = aa, bb
    if len(a) == 0:
        overlap_cache["candidate_ms"] += (time.perf_counter() - candidate_t0) * 1000.0
        empty = np.zeros((0, 2), np.int64)
        return (0, 0, empty) if return_pairs else (0, 0)
    sel = ((box[a, 1, 1] > box[b, 0, 1])
           & (box[b, 1, 1] > box[a, 0, 1]))
    a, b = a[sel], b[sel]
    same = face_labels[a] == face_labels[b]
    overlap_cache["candidate_ms"] += (time.perf_counter() - candidate_t0) * 1000.0
    overlap_cache["candidate_count"] += float(len(a))

    def narrow(ii):
        """Return the indices of overlapping candidates after the separating-axis test."""
        hit_blocks = []
        for s in range(0, len(ii), block):
            batch = ii[s:s + block]
            pa, pb = a[batch], b[batch]
            ta, tb = t[pa], t[pb]
            # Any separating axis rejects the pair. Check A's three axes first so rejected
            # candidates never allocate projections onto B's axes.
            ax = edge_normals[pa]
            ga = np.einsum("pnc,pkc->pnk", ta, ax)
            gb = np.einsum("pnc,pkc->pnk", tb, ax)
            gap = np.minimum(ga.max(1), gb.max(1)) - np.maximum(ga.min(1), gb.min(1))
            # the slack is the smaller face's own size, not the batch's: a block stamped down over
            # itself is made of faces a hundred times smaller than the ones around it, and a
            # tolerance measured on those would let the stamp pass for clean
            slack = 0.05 * np.minimum(scale[pa], scale[pb])[:, None]
            hit = (gap > slack).all(1)
            if not hit.any():
                continue
            batch = batch[hit]
            ax = edge_normals[pb[hit]]
            ga = np.einsum("pnc,pkc->pnk", ta[hit], ax)
            gb = np.einsum("pnc,pkc->pnk", tb[hit], ax)
            gap = np.minimum(ga.max(1), gb.max(1)) - np.maximum(ga.min(1), gb.min(1))
            hit_b = (gap > slack[hit]).all(1)
            if hit_b.any():
                hit_blocks.append(batch[hit_b])
        return (np.concatenate(hit_blocks) if hit_blocks
                else np.zeros(0, np.int64))

    # The SAT projection is independent of the island-label split. Process candidates together,
    # then split the surviving indices into within-island and across-island results.
    sat_candidates = (np.arange(len(a), dtype=np.int64) if mv is not None
                      else np.flatnonzero(same))
    sat_t0 = time.perf_counter()
    t = u[tl[f]]
    hit_idx = narrow(sat_candidates)
    overlap_cache["sat_ms"] += (time.perf_counter() - sat_t0) * 1000.0
    same_hit = same[hit_idx]
    same_idx = hit_idx[same_hit]
    across_idx = hit_idx[~same_hit] if mv is not None else np.zeros(0, np.int64)
    same_count = int(same_hit.sum())
    across_count = int((~same_hit).sum()) if mv is not None else 0
    if return_pairs:
        same_pairs = np.stack([f[a[same_idx]], f[b[same_idx]]], axis=1) \
            if len(same_idx) else np.zeros((0, 2), np.int64)
        across_pairs = np.stack([f[a[across_idx]], f[b[across_idx]]], axis=1) \
            if len(across_idx) else np.zeros((0, 2), np.int64)
        return same_count, across_count, np.concatenate([same_pairs, across_pairs])
    return same_count, across_count


def untangle_overlaps(tl, u, movable, lab=None, max_rounds=6,
                     moved_faces=None, initial=None, overlap_cache=None):
    """Apply small, flip-safe separating-axis corrections to detected face overlaps.

    Only vertices marked movable receive translation. A correction is retained only if the
    overlap count or penetration drops and no previously valid face flips, so this is a local
    cleanup stage, not a replacement for the main shape solve.
    """
    out = np.asarray(u, dtype=float).copy()
    movable = np.asarray(movable, dtype=bool)
    if moved_faces is None:
        moved_faces = np.any(movable[tl], axis=1)
    if initial is None:
        within, across, pairs = overlapping_pairs(
            tl, out, lab, moved_faces, return_pairs=True,
            overlap_cache=overlap_cache)
    else:
        within, across, pairs = initial
    initial_count = within + across
    if initial_count == 0:
        return out, {"before": 0, "after": 0, "rounds": 0,
                     "within": 0, "across": 0}
    if not movable.any():
        return out, {"before": initial_count, "after": initial_count, "rounds": 0,
                     "within": within, "across": across}
    start_area = signed_uv_area(out[tl])
    rounds = 0
    final_within, final_across = within, across

    for _ in range(max(0, int(max_rounds))):
        current = within + across
        if current == 0 or len(pairs) == 0:
            break

        tri_uv = out[tl]
        face_scale = np.sqrt(np.abs(signed_uv_area(tri_uv)))
        pair_faces = np.unique(pairs)
        max_step = max(float(np.median(face_scale[pair_faces])) * 0.2, 1e-8)
        edge = np.stack([tri_uv[:, 1] - tri_uv[:, 0],
                         tri_uv[:, 2] - tri_uv[:, 1],
                         tri_uv[:, 0] - tri_uv[:, 2]], axis=1)
        normals = np.stack([-edge[..., 1], edge[..., 0]], axis=-1)
        fa, fb = pairs[:, 0], pairs[:, 1]
        axes = np.concatenate([normals[fa], normals[fb]], axis=1)
        norm = np.linalg.norm(axes, axis=2)
        axes = axes / np.maximum(norm[..., None], 1e-30)
        pa = np.einsum("pnc,pkc->pnk", tri_uv[fa], axes)
        pb = np.einsum("pnc,pkc->pnk", tri_uv[fb], axes)
        move_a = pa.max(1) - pb.min(1)
        move_b = pb.max(1) - pa.min(1)
        direction_a = move_a <= move_b
        depth = np.minimum(move_a, move_b)
        axis_id = np.argmin(np.minimum(move_a, move_b), axis=1)
        pair_id = np.arange(len(pairs))
        current_cost = float(depth[pair_id, axis_id].sum())
        axis = axes[np.arange(len(pairs)), axis_id]
        axis[~direction_a[pair_id, axis_id]] *= -1.0

        delta = np.zeros_like(out)
        weight = np.zeros(len(out))
        # Each pair contributes to at most three corners on each side. Keep the original per-pair
        # sharing rule, but gather all free corner occurrences and scatter them in two vectorized
        # passes instead of running Python and np.add.at once per pair.
        verts_a, verts_b = tl[fa], tl[fb]
        free_mask_a = movable[verts_a]
        free_mask_b = movable[verts_b]
        count_a = free_mask_a.sum(1)
        count_b = free_mask_b.sum(1)
        share_a = np.where(count_b == 0, 1.0, np.where(count_a > 0, 0.5, 0.0))
        share_b = np.where(count_a == 0, 1.0, np.where(count_b > 0, 0.5, 0.0))
        push = np.minimum(depth[np.arange(len(pairs)), axis_id] + max_step * 0.1, max_step)
        contrib_a = -axis * (push * share_a)[:, None]
        contrib_b = axis * (push * share_b)[:, None]
        if free_mask_a.any():
            ids_a = verts_a[free_mask_a]
            np.add.at(delta, ids_a, np.broadcast_to(contrib_a[:, None, :],
                                                    free_mask_a.shape + (2,))[free_mask_a])
            np.add.at(weight, ids_a, 1.0)
        if free_mask_b.any():
            ids_b = verts_b[free_mask_b]
            np.add.at(delta, ids_b, np.broadcast_to(contrib_b[:, None, :],
                                                    free_mask_b.shape + (2,))[free_mask_b])
            np.add.at(weight, ids_b, 1.0)
        valid = weight > 0.0
        if not valid.any():
            break
        delta[valid] /= weight[valid, None]
        magnitude = np.linalg.norm(delta[valid], axis=1)
        delta[valid] *= np.minimum(1.0, max_step / np.maximum(magnitude, 1e-30))[:, None]

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = out.copy()
            candidate[valid] += delta[valid] * scale
            area = signed_uv_area(candidate[tl])
            live_area = np.abs(start_area) > 1e-14 * max(
                float(np.abs(start_area).max()), 1e-30)
            if np.any(area[live_area] * start_area[live_area] <= 0.0):
                continue
            next_within, next_across, next_pairs = overlapping_pairs(
                tl, candidate, lab, moved_faces, return_pairs=True,
                overlap_cache=overlap_cache)
            next_tri = candidate[tl]
            next_edge = np.stack([next_tri[:, 1] - next_tri[:, 0],
                                  next_tri[:, 2] - next_tri[:, 1],
                                  next_tri[:, 0] - next_tri[:, 2]], axis=1)
            next_normals = np.stack([-next_edge[..., 1], next_edge[..., 0]], axis=-1)
            next_axes = np.concatenate([next_normals[next_pairs[:, 0]],
                                        next_normals[next_pairs[:, 1]]], axis=1)
            next_axes /= np.maximum(np.linalg.norm(next_axes, axis=2)[..., None], 1e-30)
            next_pa = np.einsum("pnc,pkc->pnk", next_tri[next_pairs[:, 0]], next_axes)
            next_pb = np.einsum("pnc,pkc->pnk", next_tri[next_pairs[:, 1]], next_axes)
            next_depth = np.minimum(next_pa.max(1) - next_pb.min(1),
                                    next_pb.max(1) - next_pa.min(1))
            next_cost = float(np.maximum(next_depth, 0.0).min(1).sum()) if len(next_pairs) else 0.0
            if (next_within + next_across < current
                    or (next_within + next_across == current
                        and next_cost < current_cost - 1e-12 * max(current_cost, 1.0))):
                out = candidate
                within, across, pairs = next_within, next_across, next_pairs
                final_within, final_across = within, across
                rounds += 1
                accepted = True
                break
        if not accepted:
            # A rejected trial refreshed only the moved entries with its temporary coordinates;
            # restore the cache to the last accepted state before leaving this round.
            _refresh_overlap_cache(overlap_cache, tl, out, lab, moved_faces)
            break

    return out, {"before": initial_count, "after": final_within + final_across,
                 "rounds": rounds, "within": final_within, "across": final_across}
