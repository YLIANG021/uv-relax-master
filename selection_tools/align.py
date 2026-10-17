import bmesh
import bpy
import numpy as np

from ..Relax import bridge, core


class UVRELAX_OT_align(bpy.types.Operator):
    """Align selected UV points to an edge of their combined bounding box."""

    bl_idname = "uv_relax.align"
    bl_label = "Align UV Points"
    bl_options = {"REGISTER", "UNDO"}

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=(
            ("LEFT", "Left", "Align selected UV points to the leftmost selected point"),
            ("RIGHT", "Right", "Align selected UV points to the rightmost selected point"),
            ("DOWN", "Down", "Align selected UV points to the lowest selected point"),
            ("UP", "Up", "Align selected UV points to the highest selected point"),
        ),
        default="UP",
    )

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == "EDIT_MESH" and ob is not None
                and getattr(ob.data, "uv_layers", None) is not None
                and ob.data.uv_layers.active is not None)

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        got = bridge.read(bm)
        if got is None:
            self.report({"ERROR"}, "This object has no UV map")
            return {"CANCELLED"}

        _co, tri, lt, uvs, loops, uv_layer, _poly_of_tri = got
        sync = bool(context.tool_settings.use_uv_select_sync)
        corner_selected = bridge.selected_corners(bm, lt, sync, context.tool_settings)
        if not corner_selected.any():
            self.report({"WARNING"}, "Select at least one UV point to align")
            return {"CANCELLED"}

        inv = core.uvvert_ids(tri, uvs[lt])
        selected_ids = np.unique(inv.ravel()[corner_selected.ravel()])
        if not len(selected_ids):
            self.report({"WARNING"}, "Select at least one UV point to align")
            return {"CANCELLED"}

        flat_uv = uvs[lt].reshape(-1, 2)
        selected_uv = flat_uv[np.isin(inv.ravel(), selected_ids)]
        axis = 0 if self.direction in {"LEFT", "RIGHT"} else 1
        target = (float(selected_uv[:, axis].min()) if self.direction in {"LEFT", "DOWN"}
                  else float(selected_uv[:, axis].max()))

        new_uvs = uvs.copy()
        new_uvs[lt[np.isin(inv, selected_ids)], axis] = target
        bridge.write(loops, uv_layer, new_uvs)
        bmesh.update_edit_mesh(context.active_object.data)

        for area in context.screen.areas if context.screen else ():
            if area.type == "IMAGE_EDITOR":
                area.tag_redraw()
        self.report({"INFO"}, "%d UV point(s) aligned %s" %
                    (len(selected_ids), self.direction.lower()))
        return {"FINISHED"}
