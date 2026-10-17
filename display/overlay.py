import numpy as np
import bpy
import bmesh
import gpu
import struct
from bpy.app.handlers import persistent

from ..Relax import core, bridge

_ADDON_PACKAGE = __package__.rpartition(".")[0]

# Sampled from the reference bar the colours are meant to match: grey at the neutral mark,
# deep red at the 0 end, deep blue at the 2 end.
GRAY = (0.470, 0.460, 0.455)
TOO_LITTLE = (0.760, 0.024, 0.028)   # red:  this face has too little UV for its 3D area
TOO_MUCH = (0.043, 0.357, 0.745)     # blue: this face has more UV than it needs
OVERLAY_SPAN = 1.40
OVERLAY_ALPHA = 0.65
OVERLAY_BRIGHTNESS = 1.20

MAX_TRIANGLES = 400000
# Quiet period after the last edit event. During a transform the existing overlay remains visible;
# one complete snapshot is read only after the transform has settled.
REFRESH_INTERVAL = 0.05

_shader = None
# The UVO-style cache has two layers: per-object geometry/density data, and the single GPU batch
# painted from it.
_geom = None
_batch = None
_vbo = None
_vbo_count = 0
_vbo_pos_id = None
_vbo_color_id = None
_col_key = None
_dirty = True
_edit_keys = ()
_refresh_fn = None
_uv_display_saved = {}
_draw_handle = globals().get("_draw_handle")
_registered_app_handlers = globals().get("_registered_app_handlers", [])


def _object_cache():
    cache = getattr(_object_cache, "state", None)
    if cache is None:
        cache = {}
        _object_cache.state = cache
    return cache


def enable_handlers():
    """Register overlay handlers when the user enables the overlay."""
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
            draw, (), 'WINDOW', 'POST_VIEW')

    for name, handler in (("depsgraph_update_post", on_depsgraph_update),
                          ("load_post", on_load_post)):
        handlers = getattr(bpy.app.handlers, name)
        if handler in handlers:
            registration = (name, handler)
            if registration not in _registered_app_handlers:
                _registered_app_handlers.append(registration)
            continue
        # Clean up this add-on's previous function object after a module reload.
        for old in list(handlers):
            if (getattr(old, "__module__", None) == handler.__module__
                    and getattr(old, "__name__", None) == handler.__name__):
                handlers.remove(old)
        handlers.append(handler)
        _registered_app_handlers.append((name, handler))


def disable_handlers():
    """Remove only handlers installed by this overlay module."""
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    for name, handler in _registered_app_handlers:
        handlers = getattr(bpy.app.handlers, name)
        if handler in handlers:
            handlers.remove(handler)
    _registered_app_handlers.clear()


def get_shader():
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    return _shader


def log_ratio(co, tri, lt, uvs, model=None):
    """UV corners of every triangle plus, per corner, log2 of that face's texel density
    divided by the average density of its own island.

    Each island is judged against itself, so the colour shows uneven packing inside the
    island and says nothing about one island being bigger or smaller than another -- an
    island that is far too tight reads as flat grey, which is the point: relaxing cannot
    fix a whole island's size, only its shape.

    The ratio is averaged into each UV point weighted by the surrounding UV areas, so a
    point between several faces gets one continuous value instead of flat triangles.  That
    is what makes the overlay read as a gradient rather than a checkerboard."""
    tuv = uvs[lt]
    if (model is None or model[0].shape != (len(tri), 3, 3)
            or model[1].shape != (len(tri),)):
        p, a3 = core.uv3d(co, tri)
        metric_prep = core.face_metric_prep(p)
        model = (p, a3, metric_prep)
    else:
        p, a3, metric_prep = model
    _, det = core.face_metric_uv(metric_prep, tuv)

    e1 = tuv[:, 1] - tuv[:, 0]
    e2 = tuv[:, 2] - tuv[:, 0]
    auv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    inv = core.uvvert_ids(tri, tuv)
    U = int(inv.max()) + 1
    lab = core.islands(inv, U)[inv[:, 0]]
    num = np.bincount(lab, weights=det * a3)
    den = np.maximum(np.bincount(lab, weights=a3), 1e-30)
    ref = (num / den)[lab]

    per_face = np.log2(np.maximum(det, 1e-12) / np.maximum(ref, 1e-30))
    corners = inv.ravel()
    num = np.bincount(corners, weights=np.repeat(per_face * auv, 3), minlength=U)
    den = np.bincount(corners, weights=np.repeat(auv, 3), minlength=U)
    per_point = np.where(den > 0.0, num / np.maximum(den, 1e-30), 0.0)
    return tuv, per_point[inv], model


def colors(lg):
    """The reference bar's mapping: the density ratio itself read on a linear scale, grey at
    1.0, full red at 1-OVERLAY_SPAN and full blue at 1+OVERLAY_SPAN, clamped past both ends.
    lg still arrives as log2 of the ratio because that is what comes out of the point averaging,
    so it is turned back into a ratio here."""
    t = np.clip((np.exp2(lg) - 1.0) / OVERLAY_SPAN, -1.0, 1.0)
    m = np.abs(t)
    target = np.where(t[:, None] < 0.0, np.array(TOO_LITTLE), np.array(TOO_MUCH))
    out = np.empty((len(t), 4), np.float32)
    out[:, :3] = m[:, None] * target + (1.0 - m[:, None]) * np.array(GRAY)
    out[:, :3] = np.clip(out[:, :3] * OVERLAY_BRIGHTNESS, 0.0, 1.0)
    out[:, 3] = OVERLAY_ALPHA
    return out


def _uv_hash(bm, uv_layer):
    """Order-sensitive UV hash copied from UVO's edit-mesh cache strategy."""
    h = 0
    pack = struct.pack
    for face in bm.faces:
        for loop in face.loops:
            uv = loop[uv_layer].uv
            h = (h * 1000003 ^ hash(pack("2f", uv.x, uv.y))) & 0xFFFFFFFFFFFFFFFF
    return h


def edited_objects(scene=None):
    """Return the edit meshes from a scene, independent of the current draw context."""
    scene = scene or getattr(bpy.context, "scene", None)
    if scene is None:
        return []
    return [ob for ob in scene.objects
            if ob.type == 'MESH' and ob.mode == 'EDIT'
            and getattr(ob.data, "uv_layers", None) is not None]


def _edit_key(objects):
    return tuple(sorted((ob.name, ob.data.as_pointer()) for ob in objects))


def _build_object_geometry(ob):
    """Read one live edit mesh, using a copied BMesh like UVO to keep the snapshot stable."""
    bm_copy = None
    try:
        bm_live = bmesh.from_edit_mesh(ob.data)
        bm_copy = bm_live.copy()
        bm_copy.faces.ensure_lookup_table()
        uv_layer = bm_copy.loops.layers.uv.verify()
        if uv_layer is None or not bm_copy.faces:
            return None, None
        current_hash = _uv_hash(bm_copy, uv_layer)
        cached = _object_cache().get(ob.name)
        if cached and cached["hash"] == current_hash:
            return current_hash, {"geom": cached["geom"], "model": cached.get("model")}

        got = bridge.read(bm_copy)
        if got is None:
            return current_hash, None
        co, tri, lt, uvs, _loops, _uvl, _poly = got
        if len(tri) > MAX_TRIANGLES:
            return current_hash, None
        cached_model = cached.get("model") if cached else None
        tuv, lg, model = log_ratio(co, tri, lt, uvs, cached_model)
        xyz = np.empty((len(tuv) * 3, 3), np.float32)
        xyz[:, :2] = tuv.reshape(-1, 2)
        xyz[:, 2] = 0.0
        return current_hash, {"geom": (xyz, lg.reshape(-1)), "model": model}
    finally:
        if bm_copy is not None:
            bm_copy.free()


def read_geometry(objects=None):
    """Refresh changed objects and concatenate their cached geometry for painting."""
    objects = edited_objects() if objects is None else objects
    obj_cache = _object_cache()
    active_names = set()
    for ob in objects:
        active_names.add(ob.name)
        current_hash, geom = _build_object_geometry(ob)
        if current_hash is not None:
            obj_cache[ob.name] = ({"hash": current_hash, **geom}
                                  if geom is not None
                                  else {"hash": current_hash, "geom": None, "model": None})

    for name in list(obj_cache):
        if name not in active_names:
            del obj_cache[name]

    out = [entry["geom"] for name, entry in obj_cache.items()
           if entry.get("geom") is not None]
    return out or None


def paint(geom):
    """The cheap half: one batch out of cached geometry.

    The arrays go to batch_for_shader as they are -- attr_fill takes numpy, so turning them
    into Python lists first only buys a chance to be slow about it: 65 ms against 0.6 ms on
    290k vertices."""
    global _batch, _vbo, _vbo_count, _vbo_pos_id, _vbo_color_id
    positions = np.concatenate([g[0] for g in geom])
    color_data = np.concatenate([colors(g[1]) for g in geom])
    if _vbo is None or _batch is None or _vbo_count != len(positions):
        from gpu.types import GPUBatch, GPUVertBuf, GPUVertFormat
        fmt = GPUVertFormat()
        _vbo_pos_id = fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
        _vbo_color_id = fmt.attr_add(id="color", comp_type="F32", len=4, fetch_mode="FLOAT")
        _vbo = GPUVertBuf(format=fmt, len=len(positions))
        _batch = GPUBatch(type="TRIS", buf=_vbo)
        _vbo_count = len(positions)
    _vbo.attr_fill(id=_vbo_pos_id, data=positions)
    _vbo.attr_fill(id=_vbo_color_id, data=color_data)
    return _batch


def rebuild():
    """Build from the last completed snapshot; the draw callback never reads Edit BMesh."""
    global _geom, _batch, _col_key, _dirty
    key = (OVERLAY_SPAN, OVERLAY_ALPHA, OVERLAY_BRIGHTNESS)
    if _batch is not None and not _dirty and key == _col_key:
        return _batch

    # The timer prepares _geom after the transform. During a drag, the previous complete frame is
    # intentionally kept on screen and no BMesh read occurs here.
    if _dirty:
        _dirty = False
        _batch = None
    if _geom is None:
        return None
    _batch = paint(_geom)
    _col_key = key
    return _batch


def draw():
    global _edit_keys
    ctx = bpy.context
    # The draw handler is owned by SpaceImageEditor.  Its ui_type changes between UV and
    # image display states in Blender 5.x, so checking for the literal "UV" can suppress the
    # handler even while the UV editor is visible.
    area = getattr(ctx, "area", None)
    if area is not None and getattr(area, "type", None) != "IMAGE_EDITOR":
        return
    objects = edited_objects(getattr(ctx, "scene", None))
    current_keys = _edit_key(objects)
    if current_keys != _edit_keys:
        drop()
        _edit_keys = current_keys
    if not objects:
        return
    addon = ctx.preferences.addons.get(_ADDON_PACKAGE)
    if addon is None or addon.preferences is None:
        return
    prefs = addon.preferences
    if not prefs.overlay_on:
        return
    if _geom is None:
        if _refresh_fn is None:
            _schedule_refresh()
        return
    batch = rebuild()
    if batch is None:
        return
    shader = get_shader()
    gpu.state.blend_set("ALPHA")
    try:
        shader.bind()
        batch.draw(shader)
    finally:
        gpu.state.blend_set("NONE")


def tag_uv_redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            # The image editor's ui_type is not stable across UV/image display states.  Tag every
            # image editor so enabling the overlay always schedules the draw handler.
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def _image_editor_spaces():
    """Yield the live Image Editor spaces in every open window."""
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'IMAGE_EDITOR':
                continue
            space = getattr(area, "spaces", None)
            space = getattr(space, "active", None) if space is not None else None
            uv_editor = getattr(space, "uv_editor", None) if space is not None else None
            if uv_editor is not None:
                yield area, uv_editor


def apply_uv_display_settings():
    """Apply the overlay's preferred UV display values once per Image Editor area."""
    for area, uv_editor in _image_editor_spaces() or ():
        key = area.as_pointer()
        if key not in _uv_display_saved:
            try:
                _uv_display_saved[key] = (area, float(uv_editor.uv_opacity),
                                          str(uv_editor.edge_display_type))
                uv_editor.uv_opacity = 1.0
                uv_editor.edge_display_type = 'BLACK'
            except (ReferenceError, RuntimeError, TypeError):
                _uv_display_saved.pop(key, None)
    tag_uv_redraw()


def restore_uv_display_settings():
    """Restore only values still owned by the overlay; preserve edits made while it was on."""
    saved = list(_uv_display_saved.values())
    _uv_display_saved.clear()
    for area, opacity, edge_display_type in saved:
        try:
            space = area.spaces.active
            uv_editor = space.uv_editor
            # Restore each property independently. If the user changed one while the overlay was
            # enabled, it no longer equals our temporary value and must be left untouched.
            if abs(float(uv_editor.uv_opacity) - 1.0) <= 1e-6:
                uv_editor.uv_opacity = opacity
            if str(uv_editor.edge_display_type) == 'BLACK':
                uv_editor.edge_display_type = edge_display_type
        except (ReferenceError, RuntimeError, TypeError, AttributeError):
            pass
    tag_uv_redraw()


def _cancel_refresh():
    global _refresh_fn
    if _refresh_fn is not None:
        try:
            bpy.app.timers.unregister(_refresh_fn)
        except Exception:
            pass
        _refresh_fn = None


def _schedule_refresh():
    """Refresh once after Blender stops reporting the transform, keeping the last frame visible."""
    global _refresh_fn
    _cancel_refresh()

    def _fire():
        global _refresh_fn, _dirty, _geom
        _refresh_fn = None
        addon = bpy.context.preferences.addons.get(_ADDON_PACKAGE)
        if addon is not None and addon.preferences is not None and addon.preferences.overlay_on:
            try:
                refreshed = read_geometry()
            except Exception:
                # Blender may still be committing the transform. Keep the old frame and retry
                # instead of reading a partially updated Edit BMesh.
                _refresh_fn = _fire
                return REFRESH_INTERVAL
            _geom = refreshed
            _dirty = True
        tag_uv_redraw()
        return None

    _refresh_fn = _fire
    bpy.app.timers.register(_fire, first_interval=REFRESH_INTERVAL)


def invalidate():
    """Mark the painted layer stale, and ask for a frame so the refresh actually lands: the
    UV editor only draws when something tags it, so a deferred rebuild could otherwise sit
    there showing the old positions until the user happens to move the mouse."""
    _schedule_refresh()
    tag_uv_redraw()


def invalidate_colors():
    """Mark the painted layer stale without changing the cached geometry."""
    global _col_key
    _col_key = None
    if _geom is None:
        _schedule_refresh()
    tag_uv_redraw()


def drop():
    """Forget every cached array, and mark the geometry stale so the next draw reads it back.
    Clearing the caches without that would leave rebuild() with nothing to paint and no reason
    to go and get it, which is a blank overlay that never comes back."""
    global _geom, _batch, _vbo, _vbo_count, _vbo_pos_id, _vbo_color_id
    global _col_key, _dirty, _edit_keys
    _cancel_refresh()
    cache = getattr(_object_cache, "state", None)
    if cache is not None:
        cache.clear()
        _object_cache.state = None
    _geom = None
    _batch = None
    _vbo = None
    _vbo_count = 0
    _vbo_pos_id = None
    _vbo_color_id = None
    _col_key = None
    _dirty = True
    _edit_keys = ()


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Invalidate current edit meshes and reset the cache across mode transitions."""
    global _edit_keys
    edited = edited_objects(scene)
    current_keys = _edit_key(edited)
    if current_keys != _edit_keys:
        drop()
        _edit_keys = current_keys
        tag_uv_redraw()
        return
    if not edited:
        return
    for update in depsgraph.updates:
        idt = update.id
        # Selection changes can also produce a Mesh depsgraph update, but they do not change UV
        # coordinates. Only a geometry-flagged Mesh update starts the hide/readback cycle; this
        # keeps selecting all UVs from making the density layer blink.
        if (isinstance(idt, bpy.types.Mesh) and update.is_updated_geometry):
            _schedule_refresh()
            return
        if (isinstance(idt, bpy.types.Object) and idt.type == 'MESH'
                and (update.is_updated_geometry or update.is_updated_transform)):
            _schedule_refresh()
            return


@persistent
def on_load_post(_dummy):
    drop()
    tag_uv_redraw()
