if "bpy" in locals():
    import importlib
    for _module in (core, bridge, relax, selection, align, overlay, panel):
        importlib.reload(_module)

import bpy

from .Relax import bridge, core, relax
from .selection_tools import selection, align
from .display import overlay
from .UI import panel


_handle = None


def register():
    global _handle
    panel.load_icons()
    for cls in panel.CLASSES:
        bpy.utils.register_class(cls)
    _handle = bpy.types.SpaceImageEditor.draw_handler_add(
        overlay.draw, (), 'WINDOW', 'POST_VIEW')
    for name, handler in (("depsgraph_update_post", overlay.on_depsgraph_update),
                          ("load_post", overlay.on_load_post)):
        handlers = getattr(bpy.app.handlers, name)
        for old in list(handlers):
            if (getattr(old, "__module__", None) == handler.__module__
                    and getattr(old, "__name__", None) == handler.__name__):
                handlers.remove(old)
        handlers.append(handler)
    addon = bpy.context.preferences.addons.get(__package__)
    if addon is not None and addon.preferences is not None and addon.preferences.overlay_on:
        overlay.apply_uv_display_settings()


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for name, handler in (("depsgraph_update_post", overlay.on_depsgraph_update),
                          ("load_post", overlay.on_load_post)):
        handlers = getattr(bpy.app.handlers, name)
        for old in list(handlers):
            if (getattr(old, "__module__", None) == handler.__module__
                    and getattr(old, "__name__", None) == handler.__name__):
                handlers.remove(old)
    overlay.restore_uv_display_settings()
    overlay.drop()
    for cls in reversed(panel.CLASSES):
        bpy.utils.unregister_class(cls)
    panel.unload_icons()
