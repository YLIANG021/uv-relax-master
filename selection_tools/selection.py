import bmesh
import numpy as np
import bpy

from ..Relax import bridge, core


def _set_uv_flag(loop, uv_layer, name, value):
    """Set a UV selection flag on the layer data and on Blender's loop shortcut when present."""
    try:
        setattr(loop[uv_layer], name, bool(value))
    except (AttributeError, TypeError):
        pass
    shortcut = "uv_" + name
    try:
        setattr(loop, shortcut, bool(value))
    except (AttributeError, TypeError):
        pass


def _set_selection(bm, uv_layer, loops, target_loop, sync, tool_settings):
    """Replace the current UV selection with target_loop and switch to point selection mode."""
    target_mesh_vert_indices = set()
    # Clear the face-level UV flag left by a previous whole-island selection.  Blender's UV
    # editor can use that flag for the yellow display even when the per-point flags are cleared.
    for face in bm.faces:
        try:
            face.uv_select = False
        except (AttributeError, TypeError):
            pass

    for index, loop in enumerate(loops):
        selected = bool(target_loop[index])
        _set_uv_flag(loop, uv_layer, "select_vert", selected)
        _set_uv_flag(loop, uv_layer, "select_edge", False)
        _set_uv_flag(loop, uv_layer, "select", selected)
        if selected:
            target_mesh_vert_indices.add(loop.vert.index)

    if sync:
        # Sync Selection has one mesh vertex for every UV copy at a seam.  Selecting the
        # corresponding mesh vertices is the closest representation Blender can store there.
        for vert in bm.verts:
            vert.select = vert.index in target_mesh_vert_indices
        for edge in bm.edges:
            edge.select = False
        for face in bm.faces:
            face.select = False
        tool_settings.mesh_select_mode = (True, False, False)
    else:
        tool_settings.uv_select_mode = 'VERTEX'


def _marked_from_current(point_islands, selected_uv):
    """Mark islands touched by the selection present immediately before the button click."""
    marked = np.zeros(len(point_islands), dtype=bool)
    if selected_uv.any():
        marked[point_islands[selected_uv]] = True
    return marked


def _boundary_points(inv, point_islands, marked):
    """Return UV point ids on the outer or hole boundary of marked islands."""
    boundary = np.zeros(len(point_islands), dtype=bool)
    face_islands = point_islands[inv[:, 0]]
    for island in np.flatnonzero(marked):
        faces = np.flatnonzero(face_islands == island)
        if not len(faces):
            continue
        ids = np.unique(inv[faces])
        remap = -np.ones(len(point_islands), dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        triangles = remap[inv[faces]]
        edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                        triangles[:, [2, 0]])), axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary[ids[np.unique(unique_edges[counts == 1])]] = True
    return boundary


class UVRELAX_OT_select_region(bpy.types.Operator):
    """Select the interior or boundary UV points of islands touched by the current selection."""

    bl_idname = "uv_relax.select_region"
    bl_label = "Select UV Region"
    bl_options = {"REGISTER", "UNDO"}

    region: bpy.props.EnumProperty(
        items=(
            ('INTERIOR', "Interior", "Select all non-boundary points"),
            ('BOUNDARY', "Boundary", "Select all boundary points"),
        ),
        default='INTERIOR')

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == "EDIT_MESH" and ob is not None
                and getattr(ob.data, "uv_layers", None) is not None
                and ob.data.uv_layers.active is not None)

    def execute(self, context):
        me = context.active_object.data
        bm = bmesh.from_edit_mesh(me)
        got = bridge.read(bm)
        if got is None:
            self.report({"ERROR"}, "This object has no UV map")
            return {"CANCELLED"}
        _co, tri, lt, uvs, loops, uv_layer, _poly_of_tri = got

        inv = core.uvvert_ids(tri, uvs[lt])
        point_islands = core.islands(inv, int(inv.max()) + 1)
        sync = bool(context.tool_settings.use_uv_select_sync)
        corner_selected = bridge.selected_corners(
            bm, lt, sync, context.tool_settings)
        selected_uv = np.zeros(len(point_islands), dtype=bool)
        np.logical_or.at(selected_uv, inv.ravel(), corner_selected.ravel())
        if not selected_uv.any():
            self.report({"WARNING"}, "Select at least one UV point to mark an island")
            return {"CANCELLED"}

        marked = _marked_from_current(point_islands, selected_uv)
        boundary = _boundary_points(inv, point_islands, marked)
        marked_points = marked[point_islands]
        target = (marked_points & ~boundary if self.region == 'INTERIOR'
                  else marked_points & boundary)

        target_loop = np.zeros(len(loops), dtype=bool)
        np.logical_or.at(target_loop, lt.ravel(), target[inv.ravel()])
        _set_selection(bm, uv_layer, loops, target_loop, sync, context.tool_settings)
        bmesh.update_edit_mesh(me)

        for area in context.screen.areas if context.screen else ():
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()

        label = "interior" if self.region == 'INTERIOR' else "boundary"
        count = int(target.sum())
        self.report({"INFO"}, "%d %s UV point(s) selected on %d island(s)" %
                    (count, label, int(marked.sum())))
        return {"FINISHED"}
