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
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])

    def quantile(q):
        index = np.searchsorted(cumulative, q * cumulative[-1], side="left")
        return float(sorted_values[min(index, len(sorted_values) - 1)])

    median = quantile(0.5)
    p90 = quantile(0.9)
    p99 = quantile(0.99)
    median = max(median, 1e-30)
    return float(cv), p90 / median, p99 / median


def _console_log(message):
    """Keep timing output visible in Blender's system console without changing operator UI."""
    print("[UV Relax] " + message, flush=True)


class UVRELAX_OT_relax(bpy.types.Operator):
    """Relax the selected part of a UV island.  Everything you did not select keeps its UV
    coordinates exactly, so seams stay matched.  A partial island selection keeps its untouched
    UV points fixed; selecting a whole island relaxes its outer contour too."""

    bl_idname = "uv.relax_even"
    bl_label = "Relax Selection"
    bl_options = {"REGISTER", "UNDO"}

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

    axis: bpy.props.EnumProperty(
        name="Relax Axis",
        items=(('XY', "XY", "Relax both UV axes"),
               ('X', "X", "Relax only the horizontal UV coordinate"),
               ('Y', "Y", "Relax only the vertical UV coordinate")),
        default='XY',
        options={'HIDDEN'})

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
            _console_log("CANCEL reason=no_uv_map elapsed_ms=%.1f" %
                         ((time.perf_counter() - t0) * 1000.0))
            self.report({"ERROR"}, "This object has no UV map")
            return {"CANCELLED"}
        co, tri, lt, uvs, loops, uvl, poly_of_tri = got
        tuv = uvs[lt]
        read_ms = (time.perf_counter() - t0) * 1000.0

        topology_profile = {}
        inv, point_isl, topology = core.cached_uv_topology(
            tri, tuv, profile=topology_profile)
        U = int(inv.max()) + 1
        isl = topology["face_isl"]

        selection_t0 = time.perf_counter()
        sync = bool(context.tool_settings.use_uv_select_sync)
        sync_uv_valid = bool(sync and getattr(bm, "uv_select_sync_valid", False))
        selection_source = ("UV_CORNERS" if sync_uv_valid else "MESH") if sync else "UV"
        selection = bridge.selection_counts(bm, sync)
        corner_selected = bridge.selected_corners(bm, lt, sync, context.tool_settings)
        selected_point_ids = np.unique(inv.ravel()[corner_selected.ravel()])
        selection_present = bool(len(selected_point_ids))
        if not selection_present:
            _console_log(
                "CANCEL object=%s mode=%s selected_verts=%d selected_edges=%d "
                "selected_faces=%d reason=empty_selection elapsed_ms=%.1f" %
                (context.active_object.name, "SYNC" if sync else "UV",
                 selection["verts"], selection["edges"], selection["faces"],
                 (time.perf_counter() - t0) * 1000.0))
            self.report({"WARNING"}, "Nothing selected in the UV editor")
            return {"CANCELLED"}
        labs = np.unique(point_isl[selected_point_ids])
        scope_face_ids = core.incident_faces(topology, selected_point_ids)

        _console_log(
            "START object=%s mode=%s selected_verts=%d selected_edges=%d selected_faces=%d "
            "scope_faces=%d islands=%d selection_source=%s steps=%d density=%.3f" %
            (context.active_object.name, "SYNC" if sync else "UV",
             selection["verts"], selection["edges"], selection["faces"],
             len(scope_face_ids), len(labs),
             selection_source,
             int(self.steps), float(self.density_weight)))

        pos = np.zeros((U, 2))
        pos[inv.ravel()] = tuv.reshape(-1, 2)
        movable_uv = np.zeros(U, dtype=bool)
        selection_ms = (time.perf_counter() - selection_t0) * 1000.0

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
        solver_profile = {"model_ms": 0.0, "preconditioner_ms": 0.0,
                          "cg_ms": 0.0, "line_search_ms": 0.0}
        solver_algorithms = set()
        reduced_islands = 0
        reduced_faces = 0
        reduced_points = 0
        local_faces_total = 0
        local_points_total = 0
        selected_points_total = 0
        halo_points_total = 0
        local_rings = 0
        reinitialized_islands = 0
        capped_holes = 0
        geometry_profile = {}
        geometry_t0 = time.perf_counter()
        geom_p, geom_a3, geom_metric_prep = core.cached_uv_geometry(
            co, tri, profile=geometry_profile)
        geometry_ms = (time.perf_counter() - geometry_t0) * 1000.0
        island_reference_profile = {}
        island_reference_t0 = time.perf_counter()
        island_refs = core.cached_island_references(
            co, tri, inv, point_isl, tuv, pos, topology, geom_p, geom_a3,
            profile=island_reference_profile)
        island_reference_ms = (time.perf_counter() - island_reference_t0) * 1000.0
        setup_ms = (time.perf_counter() - t0) * 1000.0 - read_ms
        solve_t0 = time.perf_counter()
        for lab in labs:
            all_ids = core.grouped_indices(topology, int(lab), "point")
            selected_in_labs = selected_point_ids[point_isl[selected_point_ids] == lab]
            partial_island = len(selected_in_labs) < len(all_ids)
            local_mode = partial_island

            if local_mode:
                f = core.incident_faces(topology, selected_in_labs)
                ids = np.unique(inv[f])
                tl = np.searchsorted(ids, inv[f])
                selected_in_island = np.isin(ids, selected_in_labs, assume_unique=True)
            else:
                f = core.grouped_indices(topology, int(lab), "face")
                ids = all_ids
                tl = np.searchsorted(ids, inv[f])
                selected_in_island = np.ones(len(ids), dtype=bool)

            if local_mode:
                # Only UV points explicitly selected by the user may move.  All other points are
                # exact Dirichlet conditions, including vertices on faces touching the selection.
                free = selected_in_island.copy()
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
                # UV seams are intentional discontinuities.  Holding every seam
                # vertex makes a sharp outer contour impossible to repair, so
                # the whole-island path keeps the contour in the unknowns.
                free = np.ones(len(ids), bool)
            if not free.any():
                stuck += 1
                continue

            local_faces_total += len(f)
            local_points_total += len(ids)
            selected_points_total += len(selected_in_labs)
            halo_points_total += len(ids) - len(selected_in_labs)
            local_rings = max(local_rings, int(local_mode))

            pf = geom_p[f]
            a3 = geom_a3[f]
            solve_p = pf
            solve_tl = tl
            solve_uv = pos[ids]
            solve_pin = ~free
            solve_kwargs = {}
            reduced = local_mode
            if local_mode:
                boundary_edges_global, boundary_lengths, boundary_count = core.island_boundary(
                    topology, island_refs, int(lab))
                if len(boundary_edges_global):
                    loc = np.searchsorted(ids, boundary_edges_global)
                    inside = (loc < len(ids)) & (
                        ids[np.minimum(loc, len(ids) - 1)] == boundary_edges_global)
                    keep_boundary = np.all(inside, axis=1)
                    boundary_edges = loc[keep_boundary]
                    boundary_lengths = boundary_lengths[keep_boundary]
                else:
                    boundary_edges = np.zeros((0, 2), np.int64)
                    boundary_lengths = np.zeros(0, dtype=float)
                solve_kwargs = {
                    "total_a3_override": float(island_refs["total_a3"][lab]),
                    "density_target_override": float(island_refs["density_target"][lab]),
                    "area2_override": float(island_refs["area2"][lab]),
                    "spread_override": float(island_refs["spread"][lab]),
                    "scale_center_override": island_refs["center"][lab],
                    "majority_orientation": float(island_refs["orientation"][lab]),
                    "boundary_edges_override": boundary_edges,
                    "boundary_3d_length_override": boundary_lengths,
                    "boundary_count_override": boundary_count,
                }
                reduced_islands += 1
                reduced_faces += len(f)
                reduced_points += len(ids)
            newuv, st = core.relax_solve(
                solve_p, solve_tl, solve_uv, solve_pin, maxit=self.steps,
                density_weight=self.density_weight,
                boundary_weight=None,
                # A complete island can be safely re-charted from its 3D geometry.  A partial
                # selection must keep its current UV as the fixed-boundary problem's initial state.
                reinitialize=(not local_mode and self.axis == 'XY'),
                energy_mode="symmetric_dirichlet",
                # Temporary cap faces currently alter the full-island density normalization and
                # make repeated relax operations oscillate. Keep the stable real-face solve on the
                # button path until virtual-hole weighting is separated from the real chart.
                cap_holes=False,
                local_solver=local_mode,
                axis=self.axis,
                **solve_kwargs)
            for key in solver_profile:
                solver_profile[key] += st.get(key, 0.0)
            solver_algorithms.add(st.get("algorithm", "unknown"))
            reinitialized_islands += int(st.get("reinitialized", False))
            capped_holes += int(st.get("capped_holes", 0))
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
            if reduced:
                pos[ids] = newuv
                movable_uv[ids[free]] = True
            else:
                pos[ids] = newuv
                movable_uv[ids[free]] = True
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
            _console_log(
                "CANCEL object=%s reason=nothing_relaxed elapsed_ms=%.1f read_ms=%.1f "
            "setup_ms=%.1f topology_ms=%.1f topology_build_ms=%.1f topology_cache=%s "
            "selection_ms=%.1f geometry_ms=%.1f geometry_build_ms=%.1f geometry_cache=%s "
            "island_ref_ms=%.1f island_ref_build_ms=%.1f island_ref_cache=%s "
            "solve_ms=%.1f islands=%d reduced_islands=%d "
            "reduced_faces=%d reduced_points=%d solves=%d cg_passes=%d" %
                (context.active_object.name, (time.perf_counter() - t0) * 1000.0,
                 read_ms, setup_ms,
                 topology_profile["total_ms"], topology_profile["build_ms"],
                 topology_profile["cache"], selection_ms,
                 geometry_ms, geometry_profile["build_ms"], geometry_profile["cache"],
                 island_reference_ms, island_reference_profile["build_ms"],
                 island_reference_profile["cache"],
                 (time.perf_counter() - solve_t0) * 1000.0,
                 len(labs), reduced_islands, reduced_faces, reduced_points, solves, took))
            return {"CANCELLED"}

        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        solver_profile["misc_ms"] = max(
            0.0, solve_ms - sum(solver_profile.values()))
        overlap_fix_t0 = time.perf_counter()
        # Only faces touching UV points that actually moved enter the incremental overlap query.
        # The untouched faces remain in the cached stationary index, even when they belong to the
        # same island as a partial selection.
        moved_faces = np.any(movable_uv[inv], axis=1)
        overlap_cache = {}
        overlap_initial = core.overlapping_pairs(
            inv, pos, isl, moved=moved_faces, return_pairs=True,
            overlap_cache=overlap_cache)
        pos, overlap_fix = core.untangle_overlaps(
            inv, pos, movable_uv, isl, moved_faces=moved_faces,
            axis=self.axis,
            initial=overlap_initial, overlap_cache=overlap_cache)
        overlap_fix_ms = (time.perf_counter() - overlap_fix_t0) * 1000.0
        write_t0 = time.perf_counter()
        new_loop_uv = np.empty_like(uvs)
        new_loop_uv[lt.ravel()] = pos[inv.ravel()]
        bridge.write(loops, uvl, new_loop_uv)
        bmesh.update_edit_mesh(me)
        write_ms = (time.perf_counter() - write_t0) * 1000.0

        post_t0 = time.perf_counter()
        metric_t0 = time.perf_counter()
        metric_prep = geom_metric_prep[scope_face_ids]
        scope_tuv0 = tuv[scope_face_ids]
        scope_tuv1 = new_loop_uv[lt[scope_face_ids]]
        scope_a3 = geom_a3[scope_face_ids]
        shape0, k0 = core.face_metric_uv(metric_prep, scope_tuv0)
        shape1, k1 = core.face_metric_uv(metric_prep, scope_tuv1)
        density0 = _density_stats(k0, scope_a3)
        density1 = _density_stats(k1, scope_a3)
        was = float((k0 * scope_a3).sum())
        now = float((k1 * scope_a3).sum())
        turn0 = _inside_out(core.signed_uv_area(scope_tuv0))
        turn1 = _inside_out(core.signed_uv_area(scope_tuv1))
        metric_ms = (time.perf_counter() - metric_t0) * 1000.0
        # the energy asks each face to be a good similarity of itself and is content with two of
        # them sharing one patch of the map, so evenness and zero folds do not add up to an
        # unoverlapped island; what is left over is counted and said, not priced into the solve.
        # Every face of the mesh is looked at, not just the ones in scope, because an outline that
        # may move can grow into a neighbour that was never touched -- and a pair is only charged
        # against this button where one of its two faces sits on an island this button wrote back.
        within, across = overlap_fix["within"], overlap_fix["across"]
        overlap_ms = overlap_fix_ms
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
            tail += " | %d face pair(s) now lie on top of a neighbouring island because a border " \
                    "that moved took the room -- repack the UVs" % across
        post_ms = (time.perf_counter() - post_t0) * 1000.0
        ms = (time.perf_counter() - t0) * 1000.0
        _console_log(
            "DONE object=%s elapsed_ms=%.1f read_ms=%.1f setup_ms=%.1f "
            "topology_ms=%.1f topology_signature_ms=%.1f topology_build_ms=%.1f topology_cache=%s "
            "selection_ms=%.1f geometry_ms=%.1f geometry_signature_ms=%.1f "
            "geometry_build_ms=%.1f geometry_cache=%s "
            "island_ref_ms=%.1f island_ref_signature_ms=%.1f island_ref_build_ms=%.1f "
            "island_ref_cache=%s solve_ms=%.1f "
            "write_ms=%.1f post_ms=%.1f metric_ms=%.1f overlap_ms=%.1f "
            "overlap_refresh_ms=%.1f overlap_candidate_ms=%.1f overlap_sat_ms=%.1f "
            "overlap_candidates=%d overlap_mode=%s "
            "solver=%s "
            "solver_model_ms=%.1f precond_ms=%.1f cg_ms=%.1f density_line_ms=%.1f "
            "solver_misc_ms=%.1f "
            "selected_verts=%d selected_edges=%d "
            "selected_faces=%d islands=%d relaxed_islands=%d reduced_islands=%d "
            "reduced_faces=%d reduced_points=%d points=%d solves=%d "
            "cg_passes=%d overlap_fix=%d->%d/%d "
            "selection_source=%s steps=%d density=%.3f reinitialized=%d capped_holes=%d "
            "local_faces=%d local_points=%d selected_points=%d halo_points=%d halo_rings=%d" %
            (context.active_object.name, ms, read_ms, setup_ms,
             topology_profile["total_ms"], topology_profile["signature_ms"],
             topology_profile["build_ms"], topology_profile["cache"], selection_ms,
             geometry_ms, geometry_profile["signature_ms"], geometry_profile["build_ms"],
             geometry_profile["cache"], island_reference_ms,
             island_reference_profile["signature_ms"], island_reference_profile["build_ms"],
             island_reference_profile["cache"], solve_ms, write_ms, post_ms,
             metric_ms, overlap_ms, overlap_cache["refresh_ms"],
             overlap_cache["candidate_ms"], overlap_cache["sat_ms"],
             int(overlap_cache["candidate_count"]), overlap_cache["candidate_mode"],
             ",".join(sorted(solver_algorithms)),
             solver_profile["model_ms"], solver_profile["preconditioner_ms"],
             solver_profile["cg_ms"], solver_profile["line_search_ms"],
             solver_profile["misc_ms"],
             selection["verts"], selection["edges"], selection["faces"], len(labs), relaxed,
             reduced_islands, reduced_faces, reduced_points, points, solves, took,
             overlap_fix["before"], overlap_fix["after"], overlap_fix["rounds"],
             selection_source, int(self.steps), float(self.density_weight),
             reinitialized_islands, capped_holes,
             local_faces_total, local_points_total, selected_points_total,
             halo_points_total, local_rings))
        self.report({"WARNING" if (declined or stuck or ran_out or within or across) else "INFO"},
                    "%d UV points on %d island(s) relaxed in %.0f ms | stretch median %.3f "
                    "-> %.3f, worst 1%% %.2f -> %.2f | density CV %.3f -> %.3f, "
                    "p90/median %.2f -> %.2f, p99/median %.2f -> %.2f | "
                    "relaxed UV area %+.1f%% | %d face(s) inside out, was %d | "
                    "%d solve(s), %d refinement(s)%s"
                    % (points, relaxed, ms,
                       float(np.median(shape0)), float(np.median(shape1)),
                       float(np.percentile(shape0, 99)),
                       float(np.percentile(shape1, 99)),
                       density0[0], density1[0], density0[1], density1[1],
                       density0[2], density1[2],
                       100.0 * (now / was - 1.0), turn1, turn0, solves, took, tail))
        return {"FINISHED"}
