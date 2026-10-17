if "bpy" in locals():
    import importlib
    for _module in (core, bridge, relax, selection, align, overlay, panel):
        importlib.reload(_module)

import bpy

from .Relax import bridge, core, relax
from .selection_tools import selection, align
from .display import overlay
from .UI import panel
from . import i18n


def register():
    i18n.register()
    panel.load_icons()
    for cls in panel.CLASSES:
        bpy.utils.register_class(cls)
    addon = bpy.context.preferences.addons.get(__package__)
    if addon is not None and addon.preferences is not None and addon.preferences.overlay_on:
        overlay.enable_handlers()
        overlay.apply_uv_display_settings()
        overlay.invalidate()


def unregister():
    overlay.disable_handlers()
    overlay.restore_uv_display_settings()
    overlay.drop()
    for cls in reversed(panel.CLASSES):
        bpy.utils.unregister_class(cls)
    panel.unload_icons()
    i18n.unregister()
