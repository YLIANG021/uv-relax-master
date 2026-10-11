import numpy as np
import bpy
import bmesh
import gpu
import struct
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

from .. import core, bridge

_ADDON_PACKAGE = __package__.rpartition(".")[0]

# Sampled from the reference bar the colours are meant to match: grey at the neutral mark,
# deep red at the 0 end, deep blue at the 2 end.
GRAY = (0.470, 0.460, 0.455)
TOO_LITTLE = (0.760, 0.024, 0.028)   # red:  this face has too little UV for its 3D area
TOO_MUCH = (0.043, 0.357, 0.745)     # blue: this face has more UV than it needs

MAX_TRIANGLES = 400000
# Maximum refresh rate while an edit transform is producing update events.  This is a throttle,
# rather than a trailing debounce: a continuous drag still gets refreshes every 20 ms.
REFRESH_INTERVAL = 0.02

_shader = None
# The UVO-style cache has two layers: per-object geometry/density data, and the single GPU batch
# painted from it.  A colour preference only rebuilds the latter.
_obj_cache = {}
_geom = None
_batch = None
_col_key = None
_dirty = True
_edit_keys = ()
_refresh_fn = None


def get_shader():
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    return _shader


def log_ratio(co, tri, lt, uvs):
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
    p, a3 = core.uv3d(co, tri)
    _, det = core.face_metric(p, tuv)

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
    return tuv, per_point[inv]


def colors(lg, span, alpha, fade_grey, brightness):
    """The reference bar's mapping: the density ratio itself read on a linear scale, grey at
    1.0, full red at 1-span (0 with the default span) and full blue at 1+span (2), clamped
    past both ends.  lg still arrives as log2 of the ratio because that is what comes out of
    the point averaging, so it is turned back into a ratio here."""
    t = np.clip((np.exp2(lg) - 1.0) / max(span, 1e-3), -1.0, 1.0)
    m = np.abs(t)
    target = np.where(t[:, None] < 0.0, np.array(TOO_LITTLE), np.array(TOO_MUCH))
    out = np.empty((len(t), 4), np.float32)
    out[:, :3] = m[:, None] * target + (1.0 - m[:, None]) * np.array(GRAY)
    out[:, :3] = np.clip(out[:, :3] * brightness, 0.0, 1.0)
    out[:, 3] = alpha
    if fade_grey:
        out[:, 3] = alpha * np.minimum(m * 8.0, 1.0)
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
        cached = _obj_cache.get(ob.name)
        if cached and cached["hash"] == current_hash:
            return current_hash, cached["geom"]

        got = bridge.read(bm_copy)
        if got is None:
            return current_hash, None
        co, tri, lt, uvs, _loops, _uvl, _poly = got
        if len(tri) > MAX_TRIANGLES:
            return current_hash, None
        tuv, lg = log_ratio(co, tri, lt, uvs)
        xyz = np.empty((len(tuv) * 3, 3), np.float32)
        xyz[:, :2] = tuv.reshape(-1, 2)
        xyz[:, 2] = 0.0
        return current_hash, (xyz, lg.reshape(-1))
    finally:
        if bm_copy is not None:
            bm_copy.free()


def read_geometry(objects=None):
    """Refresh changed objects and concatenate their cached geometry for painting."""
    objects = edited_objects() if objects is None else objects
    active_names = set()
    for ob in objects:
        active_names.add(ob.name)
        current_hash, geom = _build_object_geometry(ob)
        if current_hash is not None:
            _obj_cache[ob.name] = {"hash": current_hash, "geom": geom}

    for name in list(_obj_cache):
        if name not in active_names:
            del _obj_cache[name]

    out = [entry["geom"] for name, entry in _obj_cache.items()
           if entry.get("geom") is not None]
    return out or None


def paint(geom, prefs):
    """The cheap half: one batch out of cached geometry.

    The arrays go to batch_for_shader as they are -- attr_fill takes numpy, so turning them
    into Python lists first only buys a chance to be slow about it: 65 ms against 0.6 ms on
    290k vertices."""
    return batch_for_shader(get_shader(), "TRIS", {
        "pos": np.concatenate([g[0] for g in geom]),
        "color": np.concatenate([colors(g[1], prefs.span, prefs.alpha, prefs.fade_grey,
                                         prefs.brightness)
                                 for g in geom])})


def rebuild(prefs):
    """One batch holding every edited mesh, rebuilt only when something changed."""
    global _geom, _batch, _col_key, _dirty
    key = (round(prefs.span, 3), round(prefs.alpha, 3), prefs.fade_grey,
           round(prefs.brightness, 3))
    if _batch is not None and not _dirty and key == _col_key:
        return _batch

    # The depsgraph handler throttles the event burst; the actual GPU rebuild happens on the next
    # UV draw.  Idle frames only reuse the existing batch.
    if _dirty:
        _geom = read_geometry()
        _dirty = False
        _batch = None
    if _geom is None:
        return None
    _batch = paint(_geom, prefs)
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
    if addon is None:
        return
    prefs = addon.preferences
    if not prefs.overlay_on:
        return
    batch = rebuild(prefs)
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


def _cancel_refresh():
    global _refresh_fn
    if _refresh_fn is not None:
        try:
            bpy.app.timers.unregister(_refresh_fn)
        except Exception:
            pass
        _refresh_fn = None


def _schedule_refresh():
    """Throttle a burst of depsgraph events generated by a modal UV transform.

    The timer is active only after an edit event arrives.  Further events while it is pending are
    coalesced instead of resetting the timer, so a continuous drag receives periodic refreshes.
    """
    global _refresh_fn
    if _refresh_fn is not None:
        return

    def _fire():
        global _refresh_fn, _dirty
        _refresh_fn = None
        addon = bpy.context.preferences.addons.get(_ADDON_PACKAGE)
        if addon is not None and addon.preferences.overlay_on:
            _dirty = True
        tag_uv_redraw()
        return None

    _refresh_fn = _fire
    bpy.app.timers.register(_fire, first_interval=REFRESH_INTERVAL)


def invalidate():
    """Mark the painted layer stale, and ask for a frame so the refresh actually lands: the
    UV editor only draws when something tags it, so a deferred rebuild could otherwise sit
    there showing the old positions until the user happens to move the mouse."""
    global _dirty
    _dirty = True
    tag_uv_redraw()


def invalidate_colors():
    """Mark only the painting stale.  A colour pref cannot move a triangle, so the geometry
    already read is still good and the edit mesh does not get walked again -- dragging the
    Range or Opacity slider repaints instead of re-reading."""
    global _col_key
    _col_key = None
    tag_uv_redraw()


def drop():
    """Forget every cached array, and mark the geometry stale so the next draw reads it back.
    Clearing the caches without that would leave rebuild() with nothing to paint and no reason
    to go and get it, which is a blank overlay that never comes back."""
    global _geom, _batch, _col_key, _dirty, _edit_keys
    _cancel_refresh()
    _obj_cache.clear()
    _geom = None
    _batch = None
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
        # UV edits are reported as Mesh updates, but Blender does not guarantee that the
        # geometry flag or the evaluated datablock pointer is set consistently for every UV
        # transform.  UVO deliberately treats an edit-mode Mesh update as the invalidation
        # signal and lets the 0.02 s throttle collapse the event burst.
        if isinstance(idt, bpy.types.Mesh):
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
