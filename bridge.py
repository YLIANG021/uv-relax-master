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


def _sync_mode(tool_settings):
    """Return the mesh selection mode used by Sync Selection."""
    modes = tuple(getattr(tool_settings, "mesh_select_mode", (False, False, True)))
    if len(modes) >= 3:
        if modes[0]:
            return "VERT"
        if modes[1]:
            return "EDGE"
    return "FACE"


def selected_faces(bm, poly_of_tri, sync, tool_settings=None):
    """Faces represented by the current UV selection.

    Blender 5.x can retain a per-corner UV selection while Sync Selection is on.  That
    state is essential for distinguishing two UV copies that share a mesh vertex or edge.
    It is only valid after a selection operation that preserves it, so mesh selection is
    retained as the documented fallback when the state is unavailable or empty.
    """
    sync_valid = bool(sync and getattr(bm, "uv_select_sync_valid", False))
    if sync_valid and tool_settings is not None:
        mode = _sync_mode(tool_settings)
        if mode == "VERT":
            def sync_face_selected(face):
                return any(bool(getattr(loop, "uv_select_vert", False))
                           for loop in face.loops)
        elif mode == "EDGE":
            def sync_face_selected(face):
                return any(bool(getattr(loop, "uv_select_edge", False))
                           for loop in face.loops)
        else:
            def sync_face_selected(face):
                return bool(getattr(face, "uv_select", False))

        uv_flag = np.fromiter(
            (1 if sync_face_selected(face) else 0 for face in bm.faces),
            np.int8, count=len(bm.faces)).astype(bool)
        # A valid sync layer with no UV flags means the selection came from a mesh
        # operation or the layer has not been populated yet.  Keep the mesh fallback
        # rather than turning that into an unexpected no-op.
        if uv_flag.any():
            return uv_flag[poly_of_tri]

    if sync:
        # In sync mode Blender stores the active component on the mesh elements.  A UV edge-only
        # selection does not necessarily select either adjacent face, so include selected edges and
        # vertices here instead of letting the operator mistake the selection for "nothing".
        def mesh_face_selected(face):
            return (face.select
                    or any(edge.select for edge in face.edges)
                    or any(vert.select for vert in face.verts))

        flag = (1 if mesh_face_selected(f) else 0 for f in bm.faces)
    else:
        # UV vertex/edge selection is stored per loop in Blender 5.x.  Using
        # only f.uv_select loses partial island selections because that flag
        # describes whole-face selection in some selection modes.
        def uv_face_selected(face):
            for loop in face.loops:
                if getattr(loop, "uv_select_vert", False):
                    return True
                if getattr(loop, "uv_select_edge", False):
                    return True
            return bool(getattr(face, "uv_select", False))

        flag = (1 if uv_face_selected(f) else 0 for f in bm.faces)
    sel = np.fromiter(flag, np.int8, count=len(bm.faces)).astype(bool)
    if not sync and not sel.any():
        sel = np.fromiter((1 if f.select else 0 for f in bm.faces),
                          np.int8, count=len(bm.faces)).astype(bool)
    return sel[poly_of_tri]


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
