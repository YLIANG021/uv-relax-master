if "bpy" in locals():
    import importlib
    importlib.reload(core)  # noqa: F821
    importlib.reload(bridge)  # noqa: F821
    importlib.reload(relax)  # noqa: F821
    importlib.reload(overlay)  # noqa: F821

import bpy

from . import core, bridge, relax, overlay


def _redraw_prop(self, _context):
    # every pref that updates through here only decides how the triangles are painted, so the
    # geometry that is already read stays good
    overlay.invalidate_colors()


class UVRELAX_PT_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    overlay_on: bpy.props.BoolProperty(
        name="Density Overlay", default=False, description="Colour each UV face by how much "
        "texel space it got compared to the average of the island it is on",
        update=_redraw_prop)
    span: bpy.props.FloatProperty(
        name="Range", default=1.0, min=0.2, max=4.0, soft_max=2.0,
        description="How far a face's density ratio has to sit from 1.0 to reach full red or "
                    "full blue.  1.0 is the reference bar: red at 0, grey at 1, blue at 2",
        update=_redraw_prop)
    alpha: bpy.props.FloatProperty(
        name="Opacity", default=0.55, min=0.05, max=1.0, update=_redraw_prop)
    fade_grey: bpy.props.BoolProperty(
        name="Fade Matched Faces", default=False, description="Leave faces whose density "
        "already matches unpainted, so the texture shows through", update=_redraw_prop)

    # the button's own settings live here, not on the operator: an operator property drawn in a
    # panel is rebuilt from its default at every redraw, so it can never show a state and a click
    # on it runs the operator instead of setting it
    hold_border: bpy.props.BoolProperty(
        name="Hold Border", default=False,
        description="On: a UV point only moves when every face around it is selected, so the "
                    "edge of your selection is held exactly where it is.  Off: the islands you "
                    "selected relax whole, including their outer contour, so sharp or badly "
                    "unwrapped borders can be repaired.  Selecting nothing relaxes every island "
                    "on the mesh")
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
    boundary_density: bpy.props.FloatProperty(
        name="Boundary Density Weight", default=0.20, min=0.0, max=1.0,
        description="Density Balance weight for faces touching the island boundary. Lower values "
                    "protect the contour from sharp points")

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "hold_border")
        col.prop(self, "steps")
        col.prop(self, "density_weight")
        col.prop(self, "overlay_on")
        col.prop(self, "span", slider=True)
        col.prop(self, "alpha", slider=True)
        col.prop(self, "fade_grey")


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
        op.pin_border = prefs.hold_border
        op.steps = prefs.steps
        op.density_weight = prefs.density_weight
        op.boundary_density = prefs.boundary_density
        row.prop(prefs, "hold_border", text="", icon='LOCKED')
        layout.prop(prefs, "steps", text="Steps", slider=True)
        layout.prop(prefs, "density_weight", text="Density Balance", slider=True)
        layout.prop(prefs, "boundary_density", text="Boundary Protection", slider=True)

        box = layout.box()
        box.prop(prefs, "overlay_on", text="Density Overlay", icon='IMAGE_RGB')
        sub = box.column(align=True)
        sub.active = prefs.overlay_on
        sub.prop(prefs, "span", text="Range", slider=True)
        sub.prop(prefs, "alpha", text="Opacity", slider=True)
        sub.prop(prefs, "fade_grey")
        if prefs.overlay_on:
            row = box.row(align=True)
            row.scale_y = 0.8
            row.label(text="Red = too little UV   Blue = too much", icon='INFO')


_classes = (relax.UVRELAX_OT_relax,
            UVRELAX_PT_preferences,
            UVRELAX_PT_panel)

_handle = None


def register():
    global _handle
    for cls in _classes:
        bpy.utils.register_class(cls)
    _handle = bpy.types.SpaceImageEditor.draw_handler_add(
        overlay.draw, (), 'WINDOW', 'POST_VIEW')
    if overlay.on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(overlay.on_depsgraph_update)
    if overlay.on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(overlay.on_load_post)


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for name, handler in (("depsgraph_update_post", overlay.on_depsgraph_update),
                          ("load_post", overlay.on_load_post)):
        lst = getattr(bpy.app.handlers, name)
        if handler in lst:
            lst.remove(handler)
    # the draw handler is off by now, so there is no frame to ask for -- just let the cached
    # arrays go instead of leaving a mesh's worth of vertices behind
    overlay.drop()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
