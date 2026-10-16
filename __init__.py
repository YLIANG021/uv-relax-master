if "bpy" in locals():
    import importlib
    importlib.reload(core)  # noqa: F821
    importlib.reload(bridge)  # noqa: F821
    importlib.reload(relax)  # noqa: F821
    importlib.reload(selection)  # noqa: F821
    from .display import overlay as _display_overlay
    importlib.reload(_display_overlay)

import bpy

from . import core, bridge, relax, selection
from .display import overlay


def _redraw_prop(self, _context):
    # Toggling the overlay only changes whether the cached paint is shown.
    if self.overlay_on:
        overlay.invalidate_colors()
    else:
        overlay.drop()


class UVRELAX_PT_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    overlay_on: bpy.props.BoolProperty(
        name="Density Overlay", default=False, description="Colour each UV face by how much "
        "texel space it got compared to the average of the island it is on",
        update=_redraw_prop)

    steps: bpy.props.IntProperty(
        name="Steps", default=200, min=1, max=5000,
        description="How many refinements one island may make inside a single solve.  The island's "
                    "shape is worked out rather than walked towards, so this only decides how exact "
                    "that answer is; raise it on a big island whose report says it ran out")
    density_weight: bpy.props.FloatProperty(
        name="Density Balance", default=0.30, min=0.0, max=1.0, soft_max=0.5,
        description="How much of the island's shape relaxation is allowed to spend in order to even "
                    "out texel density, weighed against the island's own area so one setting behaves "
                    "the same on a tightly packed island and a loosely packed one.  0 keeps pure shape "
                    "fidelity and leaves the packing as the unwrap left it.  Around this default the "
                    "packing evens out and the faces keep their own shape; past about 0.5 the evenness "
                    "is bought from the faces themselves, which start to look skewed, and under about "
                    "0.1 the packing does not even out -- some faces keep sitting far off the island's "
                    "average density")
    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "steps")
        col.prop(self, "density_weight")
        col.prop(self, "overlay_on")


class UVRELAX_PT_panel(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UV Relax"
    bl_label = "UV Relax"

    def draw(self, context):
        prefs = context.preferences.addons[__package__].preferences
        layout = self.layout
        row = layout.row(align=True)
        op = row.operator(relax.UVRELAX_OT_relax.bl_idname,
                          text="Relax Selection", icon='UV')
        op.steps = prefs.steps
        op.density_weight = prefs.density_weight
        # Operator properties can retain the last executed value in Blender's UI. Set the
        # main button explicitly so it always relaxes both UV axes after an X/Y-only action.
        op.axis = 'XY'
        x_op = row.operator(relax.UVRELAX_OT_relax.bl_idname,
                            text="", icon='EVENT_X')
        x_op.steps = prefs.steps
        x_op.density_weight = prefs.density_weight
        x_op.axis = 'X'
        y_op = row.operator(relax.UVRELAX_OT_relax.bl_idname,
                            text="", icon='EVENT_Y')
        y_op.steps = prefs.steps
        y_op.density_weight = prefs.density_weight
        y_op.axis = 'Y'
        layout.prop(prefs, "steps", text="Steps", slider=True)
        layout.prop(prefs, "density_weight", text="Density Balance", slider=True)
        box = layout.box()
        select_row = box.row(align=True)
        interior = select_row.operator(selection.UVRELAX_OT_select_region.bl_idname,
                                       text="Select Interior", icon='SELECT_INTERSECT')
        interior.region = 'INTERIOR'
        boundary = select_row.operator(selection.UVRELAX_OT_select_region.bl_idname,
                                       text="Select Boundary", icon='SELECT_SET')
        boundary.region = 'BOUNDARY'
        box.prop(prefs, "overlay_on", text="Density Overlay", icon='IMAGE_RGB')


_classes = (relax.UVRELAX_OT_relax,
            selection.UVRELAX_OT_select_region,
            UVRELAX_PT_preferences,
            UVRELAX_PT_panel)

_handle = None


def register():
    global _handle
    for cls in _classes:
        bpy.utils.register_class(cls)
    _handle = bpy.types.SpaceImageEditor.draw_handler_add(
        overlay.draw, (), 'WINDOW', 'POST_VIEW')
    # Reloading an extension creates new function objects.  Remove an older copy first so a
    # hot reload cannot leave two refresh handlers and two independent caches behind.
    for name, handler in (("depsgraph_update_post", overlay.on_depsgraph_update),
                          ("load_post", overlay.on_load_post)):
        handlers = getattr(bpy.app.handlers, name)
        for old in list(handlers):
            if (getattr(old, "__module__", None) == handler.__module__
                    and getattr(old, "__name__", None) == handler.__name__):
                handlers.remove(old)
        handlers.append(handler)


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for name, handler in (("depsgraph_update_post", overlay.on_depsgraph_update),
                          ("load_post", overlay.on_load_post)):
        lst = getattr(bpy.app.handlers, name)
        for old in list(lst):
            if (getattr(old, "__module__", None) == handler.__module__
                    and getattr(old, "__name__", None) == handler.__name__):
                lst.remove(old)
    # the draw handler is off by now, so there is no frame to ask for -- just let the cached
    # arrays go instead of leaving a mesh's worth of vertices behind
    overlay.drop()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
