import time

import numpy as np
import bpy
import bmesh
import gpu
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

from . import core, bridge

# Sampled from the reference bar the colours are meant to match: grey at the neutral mark,
# deep red at the 0 end, deep blue at the 2 end.
GRAY = (0.470, 0.460, 0.455)
TOO_LITTLE = (0.760, 0.024, 0.028)   # red:  this face has too little UV for its 3D area
TOO_MUCH = (0.043, 0.357, 0.745)     # blue: this face has more UV than it needs

MAX_TRIANGLES = 400000
REBUILD_MIN_SECONDS = 0.25

_shader = None
# the two halves of a rebuild, cached apart: _geom is every edited mesh's triangle corners and
# the density each corner is to be painted with, _batch is that geometry painted.  Reading the
# edit mesh is what costs; painting it is nearly free, so a colour pref must not invalidate _geom.
_geom = None
_batch = None
_col_key = None
_last_build = 0.0
_dirty = True


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


def colors(lg, span, alpha, fade_grey):
    """The reference bar's mapping: the density ratio itself read on a linear scale, grey at
    1.0, full red at 1-span (0 with the default span) and full blue at 1+span (2), clamped
    past both ends.  lg still arrives as log2 of the ratio because that is what comes out of
    the point averaging, so it is turned back into a ratio here."""
    t = np.clip((np.exp2(lg) - 1.0) / max(span, 1e-3), -1.0, 1.0)
    m = np.abs(t)
    target = np.where(t[:, None] < 0.0, np.array(TOO_LITTLE), np.array(TOO_MUCH))
    out = np.empty((len(t), 4), np.float32)
    out[:, :3] = m[:, None] * target + (1.0 - m[:, None]) * np.array(GRAY)
    out[:, 3] = alpha
    if fade_grey:
        out[:, 3] = alpha * np.minimum(m * 8.0, 1.0)
    return out


def modal_running():
    """True while any modal operator owns the mouse (a UV drag, G, R, a box select...)."""
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return False
    for window in wm.windows:
        if window.modal_operators:
            return True
    return False


def edited_objects():
    """The meshes the UV editor is currently showing.  A draw handler has a thinner context
    than an operator, so fall back to object selection if the edit-mode collection is not
    there."""
    ctx = bpy.context
    try:
        return list(ctx.objects_in_mode_unique_data)
    except Exception:
        return [ob for ob in ctx.selected_objects if ob.mode == 'EDIT']


def read_geometry():
    """The expensive half: every edited mesh's triangle corners in UV space, plus the density
    ratio each corner is to be painted with.  None of it depends on the colour prefs, so this
    is the only part that has to be thrown away when the mesh moves."""
    out = []
    for ob in edited_objects():
        if ob.mode != 'EDIT' or getattr(ob.data, "uv_layers", None) is None:
            continue
        got = bridge.read(bmesh.from_edit_mesh(ob.data))
        if got is None:
            continue
        co, tri, lt, uvs, _loops, _uvl, _poly = got
        if len(tri) > MAX_TRIANGLES:
            continue
        tuv, lg = log_ratio(co, tri, lt, uvs)
        # float32 is what the vertex buffer holds anyway, so filling it here rather than
        # letting attr_fill convert costs nothing and halves the array that gets cached
        xyz = np.empty((len(tuv) * 3, 3), np.float32)
        xyz[:, :2] = tuv.reshape(-1, 2)
        xyz[:, 2] = 0.0
        out.append((xyz, lg.reshape(-1)))
    return out or None


def paint(geom, prefs):
    """The cheap half: one batch out of cached geometry.

    The arrays go to batch_for_shader as they are -- attr_fill takes numpy, so turning them
    into Python lists first only buys a chance to be slow about it: 65 ms against 0.6 ms on
    290k vertices."""
    return batch_for_shader(get_shader(), "TRIS", {
        "pos": np.concatenate([g[0] for g in geom]),
        "color": np.concatenate([colors(g[1], prefs.span, prefs.alpha, prefs.fade_grey)
                                 for g in geom])})


def rebuild(prefs):
    """One batch holding every edited mesh, rebuilt only when something changed."""
    global _geom, _batch, _col_key, _last_build, _dirty
    key = (round(prefs.span, 3), round(prefs.alpha, 3), prefs.fade_grey)
    if _batch is not None and not _dirty and key == _col_key:
        return _batch

    # Only a stale mesh is worth the freeze and the rate limit; repainting from geometry that
    # is already read is cheap enough to do on every frame the slider asks for, and throttling
    # it would leave the overlay wearing the colours of wherever the slider last got let go.
    if _dirty:
        # Freezing during a drag is what keeps the painted layer from lagging behind the shells
        # and appearing to shove other, overlapping islands around while you move one point.
        if modal_running():
            return _batch
        now = time.perf_counter()
        if now - _last_build < REBUILD_MIN_SECONDS:
            return _batch
        _last_build = now
        _geom = read_geometry()
        _dirty = False
        _batch = None
    if _geom is None:
        return None
    _batch = paint(_geom, prefs)
    _col_key = key
    return _batch


def draw():
    ctx = bpy.context
    if getattr(getattr(ctx, "area", None), "ui_type", None) != "UV":
        return
    addon = ctx.preferences.addons.get(__package__)
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
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


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
    global _geom, _batch, _col_key, _dirty
    _geom = None
    _batch = None
    _col_key = None
    _dirty = True


@persistent
def on_depsgraph_update(_scene, depsgraph):
    """Any edited mesh counts as a reason to re-read the UVs."""
    for update in depsgraph.updates:
        idt = update.id
        if isinstance(idt, bpy.types.Mesh):
            invalidate()
            return
        if (isinstance(idt, bpy.types.Object) and idt.type == 'MESH'
                and (update.is_updated_geometry or update.is_updated_transform)):
            invalidate()
            return


@persistent
def on_load_post(_dummy):
    drop()
    tag_uv_redraw()
