import os

import bpy
import bpy.utils.previews

from ..Relax import relax
from ..selection_tools import selection, align
from ..display import overlay

ADDON_ID = __package__.rpartition(".")[0]


_icon_previews = None
_ICON_FILES = {
    "relax_horizontal": "relax_horizontal.png",
    "relax_vertical": "relax_vertical.png",
}


def _load_icon_previews():
    global _icon_previews
    if _icon_previews is not None:
        return
    previews = bpy.utils.previews.new()
    icon_dir = os.path.join(os.path.dirname(__file__), "Icons")
    for name, filename in _ICON_FILES.items():
        path = os.path.join(icon_dir, filename)
        if os.path.isfile(path):
            previews.load(name, path, 'IMAGE')
    _icon_previews = previews


def _icon_value(name):
    return _icon_previews[name].icon_id if _icon_previews and name in _icon_previews else 0


def load_icons():
    _load_icon_previews()


def unload_icons():
    global _icon_previews
    if _icon_previews is not None:
        bpy.utils.previews.remove(_icon_previews)
        _icon_previews = None


def _redraw_prop(self, _context):
    # Apply the display preference once per toggle; user edits remain untouched until the next
    # time the overlay is enabled.
    if self.overlay_on:
        overlay.enable_handlers()
        overlay.apply_uv_display_settings()
        overlay.invalidate_colors()
    else:
        overlay.disable_handlers()
        overlay.restore_uv_display_settings()
        overlay.drop()


class UVRELAX_PT_preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    overlay_on: bpy.props.BoolProperty(
        name="Density Overlay", default=False, description="Colour each UV face by how much "
        "texel space it got compared to the average of the island it is on",
        update=_redraw_prop)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "overlay_on")


class UVRELAX_PT_panel(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UV Relax Master"
    bl_label = "UV Relax Master"

    def draw(self, context):
        prefs = context.preferences.addons[ADDON_ID].preferences
        layout = self.layout
        layout.label(text="Relax")
        action_box = layout.box()
        row = action_box.row(align=True)
        row.scale_y = 1.25
        op = row.operator(relax.UVRELAX_OT_relax.bl_idname,
                          text="Relax Selection")
        op.steps = relax.DEFAULT_STEPS
        op.density_weight = relax.DEFAULT_DENSITY_WEIGHT
        # Operator properties can retain the last executed value in Blender's UI. Set the
        # main button explicitly so it always relaxes both UV axes after an X/Y-only action.
        op.axis = 'XY'
        x_op = row.operator(relax.UVRELAX_OT_relax.bl_idname,
                            text="", icon_value=_icon_value("relax_horizontal"))
        x_op.steps = relax.DEFAULT_STEPS
        x_op.density_weight = relax.DEFAULT_DENSITY_WEIGHT
        x_op.axis = 'X'
        y_op = row.operator(relax.UVRELAX_OT_relax.bl_idname,
                            text="", icon_value=_icon_value("relax_vertical"))
        y_op.steps = relax.DEFAULT_STEPS
        y_op.density_weight = relax.DEFAULT_DENSITY_WEIGHT
        y_op.axis = 'Y'
        overlay_row = action_box.row(align=True)
        overlay_row.scale_y = 1.25
        overlay_row.prop(prefs, "overlay_on", text="Density Overlay", toggle=True,
                         icon='HIDE_OFF' if prefs.overlay_on else 'HIDE_ON')
        layout.label(text="Selection & Alignment")
        controls_box = layout.box()
        controls = controls_box.row()
        direction_col = controls.column(align=True)
        direction_col.scale_x = 0.9
        direction_col.scale_y = 0.9

        def draw_direction_row(row, left=None, middle=None, right=None):
            row.scale_x = 1.1
            row.scale_y = 1.1
            for item in (left, middle, right):
                if item is None:
                    # Keep the three-column geometry without adding inactive controls.
                    row.label(text="", icon='BLANK1')
                    continue
                direction, icon = item
                custom_icon = _icon_value(icon)
                if custom_icon:
                    op = row.operator(align.UVRELAX_OT_align.bl_idname,
                                      text="", icon_value=custom_icon)
                else:
                    op = row.operator(align.UVRELAX_OT_align.bl_idname,
                                      text="", icon=icon)
                op.direction = direction

        draw_direction_row(direction_col.row(align=True),
                           None, ('UP', 'TRIA_UP_BAR'), None)
        draw_direction_row(direction_col.row(align=True),
                           ('LEFT', 'TRIA_LEFT_BAR'), None, ('RIGHT', 'TRIA_RIGHT_BAR'))
        draw_direction_row(direction_col.row(align=True),
                           None, ('DOWN', 'TRIA_DOWN_BAR'), None)

        select_col = controls.column()
        select_col.scale_y = 1.3
        interior = select_col.operator(selection.UVRELAX_OT_select_region.bl_idname,
                                       text="Select Interior", icon='STICKY_UVS_LOC')
        interior.region = 'INTERIOR'
        boundary = select_col.operator(selection.UVRELAX_OT_select_region.bl_idname,
                                       text="Select Boundary", icon='MESH_PLANE')
        boundary.region = 'BOUNDARY'


CLASSES = (relax.UVRELAX_OT_relax,
           selection.UVRELAX_OT_select_region,
           align.UVRELAX_OT_align,
           UVRELAX_PT_preferences,
           UVRELAX_PT_panel)
