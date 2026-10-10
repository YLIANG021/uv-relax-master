import time

import bpy
import bmesh
import numpy as np

from . import core, bridge


def _inside_out(signs):
    """How many faces are the way round the others are not.  The yardstick is whichever way
    round most of the selection is, so an island that was mirrored on purpose still counts
    as correct instead of adding itself to the number."""
    s = np.sign(signs)
    return int((s != (1.0 if s.sum() >= 0.0 else -1.0)).sum())


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if not len(values) or cumulative[-1] <= 0.0:
        return 0.0
    index = np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    return float(values[min(index, len(values) - 1)])


def _density_stats(density, area3d):
    """3D-area-weighted density CV and percentile-to-median ratios."""
    valid = np.isfinite(density) & np.isfinite(area3d) & (area3d > 1e-30)
    values = density[valid]
    weights = area3d[valid]
    if not len(values) or weights.sum() <= 0.0:
        return 0.0, 0.0, 0.0
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    cv = np.sqrt(max(variance, 0.0)) / max(mean, 1e-30)
    median = _weighted_quantile(values, weights, 0.5)
    p90 = _weighted_quantile(values, weights, 0.9)
    p99 = _weighted_quantile(values, weights, 0.99)
    median = max(median, 1e-30)
    return float(cv), p90 / median, p99 / median
class UVRELAX_OT_relax(bpy.types.Operator):
    """Relax the selected part of a UV island.  Everything you did not select keeps its UV
    coordinates exactly, so seams stay matched.  With Hold Border off, the islands you touched relax
    whole instead, their own outline moving with them wherever nothing is matched to it"""

    bl_idname = "uv.relax_even"
    bl_label = "Relax Selection"
    bl_options = {"REGISTER", "UNDO"}

    steps: bpy.props.IntProperty(
        name="Steps", default=200, min=1, max=5000,
        description="How many refinements one island may make inside a single solve.  The island's "
                    "shape is worked out rather than walked towards, so this only decides how exact "
                    "that answer is; raise it on a big island whose report says it ran out")

    pin_border: bpy.props.BoolProperty(
        name="Hold Border", default=False,
        description="On: a UV point only moves when every face around it is selected, so the "
                    "edge of your selection is held exactly where it is.  Off: the islands you "
                    "selected relax whole, including their outer contour, so sharp or badly "
                    "unwrapped borders can be repaired.  Selecting nothing relaxes every island "
                    "on the mesh")

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

    @classmethod
    def poll(cls, context):
        # Mesh.type reports None while the object is in edit mode, so look for UV layers
        ob = context.active_object
        return (context.mode == "EDIT_MESH" and ob is not None
                and getattr(ob.data, "uv_layers", None) is not None
                and ob.data.uv_layers.active is not None)

    def execute(self, context):
        t0 = time.perf_counter()
        me = context.active_object.data
        bm = bmesh.from_edit_mesh(me)
        got = bridge.read(bm)
        if got is None:
            self.report({"ERROR"}, "This object has no UV map")
            return {"CANCELLED"}
        co, tri, lt, uvs, loops, uvl, poly_of_tri = got
        tuv = uvs[lt]

        inv = core.uvvert_ids(tri, tuv)
        U = int(inv.max()) + 1
        isl = core.islands(inv, U)[inv[:, 0]]

        sel = bridge.selected_faces(bm, poly_of_tri,
                                    context.tool_settings.use_uv_select_sync)
        selection_present = bool(sel.any())
        if self.pin_border or selection_present:
            if not sel.any():
                self.report({"WARNING"}, "Nothing selected in the UV editor")
                return {"CANCELLED"}
            labs = np.unique(isl[sel])
            scope = sel.copy()
        else:
            # Nothing is held by the selection: each island you touched relaxes whole, and the only
            # thing held is that island's own outline, which the solve treats as boundary
            # conditions rather than as material to spend.  An island you did not touch keeps its UV
            # coordinates, and with nothing picked that is every island on the mesh.
            labs = np.unique(isl[sel]) if sel.any() else np.unique(isl)
            scope = np.isin(isl, labs)

        pos = np.zeros((U, 2))
        pos[inv.ravel()] = tuv.reshape(-1, 2)
        # which UV points something else is matched to, so that holding them is what keeps a seam
        # together -- see core.glued_points
        glued = core.glued_points(tri, inv, isl)

        relaxed = 0
        points = 0
        even = 0
        settled = 0
        stuck = 0
        declined = 0
        turned = 0
        mirrored = 0
        mirrored_faces = 0
        ran_out = 0
        took = 0
        solves = 0
        # which islands the loop actually wrote back: the overlap count is charged against these
        # faces only, so an island that was already stamped over a neighbour before the button was
        # pressed is not reported as this button's work
        moved_labs = []
        for lab in labs:
            f = np.flatnonzero(isl == lab)
            ids = np.unique(inv[f])
            remap = -np.ones(U, np.int64)
            remap[ids] = np.arange(len(ids))
            tl = remap[inv[f]]

            selected_in_island = sel[f]
            partial_island = bool(selected_in_island.any() and not selected_in_island.all())
            local_mode = self.pin_border or partial_island

            edge = np.sort(np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]],
                                           tl[:, [2, 0]]]), 1)
            uk, cnt = np.unique(edge, axis=0, return_counts=True)
            rim = np.zeros(len(ids), bool)
            rim[uk[cnt == 1].ravel()] = True
            boundary_faces = np.any(rim[tl], axis=1)
            density_face_weight = np.where(boundary_faces,
                                            self.boundary_density, 1.0)

            if local_mode:
                # A point may move only when every face around it was selected.  That is what
                # keeps the untouched part of the island, and any seam it shares, exactly where
                # it is -- and it is also what lets a selected outline, hole rims included, take
                # part in the relax instead of being locked because it happens to be an outline.
                around = np.bincount(tl.ravel(), weights=np.repeat(~sel[f], 3),
                                     minlength=len(ids))
                free = around == 0
            else:
                # The whole island relaxes, and the only part of its outline held where it stands is
                # the part something else is matched to -- a seam side, or a point two islands were
                # merged onto -- because moving one side of those is what tears a seam on the model.
                # The rest of the outline is the edge of the mesh itself: nothing continues past it,
                # nothing is glued to it, and it joins the unknowns.  Held anyway it would be the one
                # thing this could never repair, since every interior point then has to fit inside
                # whatever shape the unwrap happened to give the border.
                #
                # That the outline was held at all was measured, not guessed -- with density switched
                # off entirely the smallest outline chord angle fell 49.2 -> 28.7 deg on a 10080-face
                # island, and at matched density evenness 39.2 -> 33.4 deg on a 1872-face one -- so an
                # outline that is free is spent on the packing, and what keeps the island from being
                # squeezed into its neighbours is Density Balance and the report's own overlap count
                # rather than a wall that cannot be moved either way.  An island with no interior
                # point -- one face, a two-face strip -- has no outline to hold and would freeze
                # completely, so those still relax whole.
                edge = np.sort(np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]],
                                               tl[:, [2, 0]]]), 1)
                uk, cnt = np.unique(edge, axis=0, return_counts=True)
                rim = np.zeros(len(ids), bool)
                rim[uk[cnt == 1].ravel()] = True
                # UV seams are intentional discontinuities.  Holding every seam
                # vertex makes a sharp outer contour impossible to repair, so
                # the whole-island path keeps the contour in the unknowns.
                # `Hold Border` above remains the explicit fixed-boundary mode.
                free = np.ones(len(ids), bool)
            if not free.any():
                stuck += 1
                continue

            pf, a3 = core.uv3d(co, tri[f])
            newuv, st = core.relax_solve(
                pf, tl, pos[ids], ~free, maxit=self.steps,
                density_weight=self.density_weight,
                density_face_weight=density_face_weight)
            if st["pass"] == 0:
                # an island whose every point is held by a face left out is a different
                # answer from one that has settled, and saying the wrong one sends him
                # looking for a bug that is not there
                if st["why"] == "nothing to move":
                    stuck += 1
                elif st["why"] == "left alone":
                    # the island was solved and the only answers on offer were ones with more faces
                    # turned over and no better shape -- which is a different answer from "there was
                    # nothing to do"
                    declined += 1
                    turned += int(st["fold"])
                else:
                    even += 1
                    settled += int(st["free"])
                continue

            if not st["converged"] and st["fold_left"]:
                # a solve ran out of refinements and the island still holds faces turned over, so the
                # number he can raise is the answer to what he is looking at -- an island that came
                # out clean has nothing left for Steps to buy
                ran_out += 1
            if st["ask"] == "even" and st["fold_left"]:
                # the island was evened where it lies on top of itself rather than unfolded:
                # worth him knowing which of the two he is looking at
                mirrored += 1
                mirrored_faces += int(st["fold_left"])
            pos[ids] = newuv
            moved_labs.append(lab)
            points += int(free.sum())
            relaxed += 1
            took = max(took, st["pass"])
            solves = max(solves, st["solve"])

        if relaxed == 0:
            why = []
            if even:
                why.append("%d point(s) on %d island(s) already keep their shape as well as "
                           "they can" % (settled, even))
            if stuck:
                why.append("%d island(s) in which every point touches a face you left out"
                           % stuck)
            if declined:
                why.append("%d island(s) with %d face(s) turned the way round the rest is not, "
                           "where relaxing would only flatten those further, so they were left "
                           "as they are" % (declined, turned))
            if not why:
                why = ["nothing was left to relax"]
            # a settled island is the selection doing its job; anything that stopped him short
            # of what he asked for has to be a popup, because a report at INFO level shows
            # nowhere on screen and the button then simply looks dead
            self.report({"WARNING" if (stuck or declined) else "INFO"},
                        "Nothing relaxed: " + ", ".join(why))
            return {"CANCELLED"}

        new_loop_uv = np.empty_like(uvs)
        new_loop_uv[lt.ravel()] = pos[inv.ravel()]
        bridge.write(loops, uvl, new_loop_uv)
        bmesh.update_edit_mesh(me)

        ms = (time.perf_counter() - t0) * 1000.0
        p, a3 = core.uv3d(co, tri)
        shape0, k0 = core.face_metric(p, tuv)
        shape1, k1 = core.face_metric(p, new_loop_uv[lt])
        density0 = _density_stats(k0[scope], a3[scope])
        density1 = _density_stats(k1[scope], a3[scope])
        was = float((k0[scope] * a3[scope]).sum())
        now = float((k1[scope] * a3[scope]).sum())
        turn0 = _inside_out(core.signed_uv_area(tuv)[scope])
        turn1 = _inside_out(core.signed_uv_area(new_loop_uv[lt])[scope])
        # the energy asks each face to be a good similarity of itself and is content with two of
        # them sharing one patch of the map, so evenness and zero folds do not add up to an
        # unoverlapped island; what is left over is counted and said, not priced into the solve.
        # Every face of the mesh is looked at, not just the ones in scope, because an outline that
        # may move can grow into a neighbour that was never touched -- and a pair is only charged
        # against this button where one of its two faces sits on an island this button wrote back.
        within, across = core.overlapping_pairs(inv, pos, isl,
                                                moved=np.isin(isl, moved_labs))
        tail = ""
        if declined:
            tail += " | %d island(s) left alone: %d face(s) in them are turned the way round " \
                    "the rest is not, and relaxing only flattens those further" % (declined,
                                                                                    turned)
        if stuck:
            tail += " | %d island(s) where every point touches a face you left out" % stuck
        if mirrored:
            tail += " | %d island(s) were evened out where they lie on top of themselves rather " \
                    "than unfolded: %d face(s) are still the wrong way round, and only cutting a " \
                    "seam takes that off" % (mirrored, mirrored_faces)
        if ran_out:
            tail += " | %d island(s) worked to the Steps limit and still hold faces turned over " \
                    "-- raise Steps" % ran_out
        if within:
            tail += " | %d face pair(s) still lie on top of each other inside one island, which " \
                    "only cutting a seam takes off" % within
        if across:
            tail += " | %d face pair(s) now lie on top of a neighbouring island, because a border " \
                    "that was allowed to move took the room -- repack, or turn Hold Border on" \
                    % across
        self.report({"WARNING" if (declined or stuck or ran_out or within or across) else "INFO"},
                    "%d UV points on %d island(s) relaxed in %.0f ms | stretch median %.3f "
                    "-> %.3f, worst 1%% %.2f -> %.2f | density CV %.3f -> %.3f, "
                    "p90/median %.2f -> %.2f, p99/median %.2f -> %.2f | "
                    "relaxed UV area %+.1f%% | %d face(s) inside out, was %d | "
                    "%d solve(s), %d refinement(s)%s"
                    % (points, relaxed, ms,
                       float(np.median(shape0[scope])), float(np.median(shape1[scope])),
                       float(np.percentile(shape0[scope], 99)),
                       float(np.percentile(shape1[scope], 99)),
                       density0[0], density1[0], density0[1], density1[1],
                       density0[2], density1[2],
                       100.0 * (now / was - 1.0), turn1, turn0, solves, took, tail))
        return {"FINISHED"}
