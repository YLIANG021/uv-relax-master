"""Reading and writing a mesh that is in edit mode.

While an object is being edited, Mesh.uv_layers.active.data is empty and the UV selection
is not on the datablock at all: the live data only exists in the edit mesh, so everything
here goes through bmesh.  bmesh has no foreach access, hence the Python loops.
"""

import numpy as np


def read(bm):
    """(co, tri, lt, uv, loops, uvl, poly_of_tri) of a live edit mesh.

    co is indexed by vertex index, uv by loop position; tri holds vertex indices and
    lt the matching loop positions, n-gons fan-triangulated.  loops is the BMLoop list
    uv rows belong to, so write() can put results back.
    """
    uvl = bm.loops.layers.uv.active
    if uvl is None:
        return None
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    co = np.empty((len(bm.verts), 3))
    for v in bm.verts:
        co[v.index] = v.co

    loops = []
    tri = []
    lt = []
    counts = []
    for f in bm.faces:
        fl = f.loops[:]
        base = len(loops)
        loops.extend(fl)
        vs = [crn.vert.index for crn in fl]
        n = len(fl) - 2
        for i in range(1, n + 1):
            tri.append((vs[0], vs[i], vs[i + 1]))
            lt.append((base, base + i, base + i + 1))
        counts.append(n)
    if not tri:
        return None

    uv = np.empty((len(loops), 2))
    for i, crn in enumerate(loops):
        uv[i] = crn[uvl].uv
    poly_of_tri = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return (co, np.array(tri, np.int64), np.array(lt, np.int64), uv, loops, uvl,
            poly_of_tri)


def write(loops, uvl, uv_per_loop):
    for crn, uv in zip(loops, uv_per_loop.tolist()):
        crn[uvl].uv = uv


def selected_corners(bm, lt, sync, tool_settings=None):
    """Return the selected state of each UV corner in triangle order.

    Selection is kept at corner level so a selected point or edge does not turn every other
    vertex of the touching face into a movable point.
    """
    if sync:
        modes = tuple(getattr(tool_settings, "mesh_select_mode", (False, False, True))) \
            if tool_settings is not None else (False, False, True)
        mode = "VERT" if len(modes) >= 1 and modes[0] else (
            "EDGE" if len(modes) >= 2 and modes[1] else "FACE")
    else:
        uv_mode = str(getattr(tool_settings, "uv_select_mode", "VERTEX")) \
            if tool_settings is not None else "VERTEX"
        mode = ("VERT" if uv_mode == "VERTEX" else
                "EDGE" if uv_mode == "EDGE" else "FACE")
    sync_valid = bool(sync and getattr(bm, "uv_select_sync_valid", False))

    def flags_for_face(face, use_uv):
        loops = face.loops[:]
        if use_uv and mode == "VERT":
            return [bool(getattr(loop, "uv_select_vert", False)) for loop in loops]
        if use_uv and mode == "EDGE":
            # A selected UV edge selects both of its endpoint corners.  Include the edge
            # entering the corner as well as the edge leaving it.
            return [bool(getattr(loop, "uv_select_edge", False)
                         or getattr(loops[i - 1], "uv_select_edge", False))
                    for i, loop in enumerate(loops)]
        if use_uv:
            return [bool(getattr(face, "uv_select", False))] * len(loops)
        if mode == "VERT":
            return [bool(loop.vert.select) for loop in loops]
        if mode == "EDGE":
            return [bool(loop.edge.select or loops[i - 1].edge.select)
                    for i, loop in enumerate(loops)]
        return [bool(face.select)] * len(loops)

    use_uv = sync_valid or not sync
    loop_flags = [flag for face in bm.faces for flag in flags_for_face(face, use_uv)]
    # Blender can report a valid sync UV layer before it has populated any flags.  In that
    # state, retain the mesh-selection fallback instead of treating the operation as a no-op.
    if sync and use_uv and not any(loop_flags):
        loop_flags = [flag for face in bm.faces for flag in flags_for_face(face, False)]

    return np.asarray(loop_flags, dtype=bool)[lt]


def selection_counts(bm, sync):
    """Count the components visible to the user in the current selection mode."""
    if sync:
        return {
            "verts": sum(bool(v.select) for v in bm.verts),
            "edges": sum(bool(e.select) for e in bm.edges),
            "faces": sum(bool(f.select) for f in bm.faces),
        }

    uvl = bm.loops.layers.uv.active
    verts = set()
    edges = set()
    faces = 0
    for face in bm.faces:
        loops = face.loops[:]
        face_selected = bool(getattr(face, "uv_select", False))
        for i, loop in enumerate(loops):
            if getattr(loop, "uv_select_vert", False):
                if uvl is None:
                    verts.add(loop.vert.index)
                else:
                    uv = loop[uvl].uv
                    verts.add((loop.vert.index, round(float(uv.x), 6), round(float(uv.y), 6)))
            if getattr(loop, "uv_select_edge", False):
                nxt = loops[(i + 1) % len(loops)]
                if uvl is None:
                    a = (loop.vert.index,)
                    b = (nxt.vert.index,)
                else:
                    ua = loop[uvl].uv
                    ub = nxt[uvl].uv
                    a = (loop.vert.index, round(float(ua.x), 6), round(float(ua.y), 6))
                    b = (nxt.vert.index, round(float(ub.x), 6), round(float(ub.y), 6))
                edges.add(tuple(sorted((a, b))))
            face_selected = (face_selected or getattr(loop, "uv_select_vert", False)
                             or getattr(loop, "uv_select_edge", False))
        faces += int(face_selected)
    return {"verts": len(verts), "edges": len(edges), "faces": faces}
