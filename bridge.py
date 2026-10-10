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


def selected_faces(bm, poly_of_tri, sync):
    """Faces the UV editor considers selected.  With select-sync on, that is the ordinary
    face selection; with it off, the UV selection is a separate flag."""
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
