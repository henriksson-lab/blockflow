// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The acceptance suite for the globally consistent label volume**, and for the
// two ways of producing one that `ops::label` now offers:
//
// | | what it is |
// |---|---|
// | **materialise** | `LabelComponentsOp` then `RelabelComponentsOp` — two fragment phases, a second `u32` image |
// | **decorate** | `LabelComponentsOp`, then `GlobalLabels::merge` once, then `RelabelledEnvironment` applying the table as a read is served |
//
// The bar is the one this crate states everywhere: **byte-identical output
// across block sizes against a whole-volume reference**. Not "the same
// partition" — the same bytes, which is a stronger claim and the one a consumer
// that *stores* a label needs. `ops::components::label_members_into_with` over
// the whole mask in one call is the reference, and it is the same kernel each
// block runs, so a disagreement is a decomposition defect and not a modelling
// difference.
//
// The grids, and why they are checked before they are used
// ---------------------------------------------------------
// The standing lesson here is a decomposition-invariance test that had quietly
// shrunk to two grids, one of them a single block — a passing test that had
// stopped meaning anything, because a single block has no seams and two grids
// that differ only in one axis exercise one kind of seam. So
// [`the_grids_are_genuinely_distinct_and_genuinely_cut`] runs first and asserts,
// of the list every other test sweeps:
//
// * every grid is distinct from every other **as a lattice**, not merely as a
//   block edge — two block edges that produce the same `blocks_per_axis` are one
//   grid wearing two names;
// * the single-block grid is present exactly once, because the no-seam case is
//   worth having and is worth having *knowingly*;
// * at least three grids cut **all three axes**, so lattice edges and lattice
//   corners are crossed and not only faces;
// * the block edge divides the volume evenly on no grid but one, so a remainder
//   block is the ordinary case here rather than an afterthought.
//
// The liveness control
// --------------------
// Every assertion of agreement below has one beside it that fails if the thing
// under test is removed. They are named `..._is_not_vacuous` or are stated in
// the same test, and there are three distinct ones because there are three
// distinct ways this could pass while meaning nothing:
//
// 1. **the merge might not be load-bearing** — if no component crossed a seam,
//    the block-local labelling would already be the answer. Asserted by
//    comparing the local label count against the component count at every grid;
// 2. **the canonical numbering might not be load-bearing** — a union-find root
//    is a correct *partition* with a decomposition-dependent *name*, and a test
//    that compared partitions rather than bytes would not see the difference.
//    Asserted by building the root-numbered answer explicitly and requiring it
//    to disagree with the reference;
// 3. **the decoration might not be happening** — a decorator that matched no
//    read forwards local labels and answers plausibly. Asserted with
//    `RelabelledEnvironment::remapped_reads`, and by a run through the
//    *undecorated* environment being required to differ.

use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{s, Array3};

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::components::{label_members_into_with, Connectivity, LabelIndex, Union};
use blockflow::ops::label::{
    component_label_phases, component_labelling_phase, gather_component_faces, ComponentFaces,
    GlobalLabels, LabelComponentsOp, RelabelComponentsOp, RelabelledEnvironment,
};
use blockflow::ops::tabulate::{
    collect_tabulation, tabulate_phases, FixedPoint, MergeTabulationOp, TabulateValuesOp,
};
use blockflow::ops::voxelwise::CarryOp;
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [21, 19, 23];
const STREAM: &str = "components";

// ------------------------------------------------------------- the fixture --

/// A sparse pseudorandom mask, which is the fixture `tests/connectivity.rs`
/// reaches for and for the same reason: it produces many small components in no
/// arranged position, so **some** of them cross a seam at every grid below and
/// none of them was placed to.
///
/// The density is chosen so that the components are neither one blob nor all
/// singletons — both of those are fixtures that cannot fail. The counts are
/// asserted rather than described, in `the_fixture_has_the_shape_the_sweep_needs`.
fn mask() -> Array3<bool> {
    let mut out = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    let mut state = 0x9e3779b97f4a7c15u64;
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                out[[i, j, k]] = (state >> 33) % 100 < 28;
            }
        }
    }
    out
}

/// The grids every sweep below runs over. See the header for what
/// `the_grids_are_genuinely_distinct_and_genuinely_cut` asserts about them.
fn grids() -> Vec<[usize; 3]> {
    vec![
        VOLUME,       // one block: no seams at all
        [11, 19, 23], // one axis cut, evenly-ish
        [21, 10, 12], // two axes cut
        [8, 7, 9],    // all three, with remainders
        [5, 5, 5],    // all three, finer
        [4, 13, 6],   // all three, uneven edges
    ]
}

/// The whole-volume reference: the same kernel, called once.
fn reference(mask: &Array3<bool>, connectivity: Connectivity) -> Array3<u32> {
    let mut out = Array3::<u32>::zeros(mask.raw_dim());
    label_members_into_with(VOLUME, connectivity, |at| mask[at], out.view_mut())
        .expect("the reference labels");
    out
}

// -------------------------------------------------------------- the plans --

fn labelling_only(
    block: [usize; 3],
    connectivity: Connectivity,
) -> (Decomposition, LabelComponentsOp) {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label =
        LabelComponentsOp::new("label", STREAM, Lifecycle::Persistent).connecting(connectivity);
    let plan = component_labelling_phase(grid, Dtype::Bool, &label).expect("a plan");
    (plan, label)
}

/// Run the labelling phase and stop. Returns the environment (holding the local
/// labels on image 1 and the fragments in its sidecars) and the plan.
fn run_labelling(
    mask: &Array3<bool>,
    block: [usize; 3],
    connectivity: Connectivity,
) -> (ArrayEnvironment, Decomposition, [usize; 3]) {
    let (plan, label) = labelling_only(block, connectivity);
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    execute_phases(
        "label",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label)],
    )
    .expect("a run");
    (env, plan, block)
}

/// The **materialised** answer: two fragment phases, the second writing the
/// global label volume as an ordinary image.
fn materialised(mask: &Array3<bool>, block: [usize; 3], connectivity: Connectivity) -> Array3<u32> {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label =
        LabelComponentsOp::new("label", STREAM, Lifecycle::DeleteOnExit).connecting(connectivity);
    let relabel = RelabelComponentsOp::new("relabel", STREAM, 0, &grid).connecting(connectivity);
    let plan = component_label_phases(grid, Dtype::Bool, &label, &relabel).expect("a plan");
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    execute_phases(
        "relabel",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&relabel)],
    )
    .expect("a run");
    env.output()
        .view::<u32>()
        .expect("a label volume")
        .to_owned()
}

/// The table, built once, from a finished labelling run.
fn table(env: &ArrayEnvironment, block: [usize; 3], connectivity: Connectivity) -> GlobalLabels {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let reports: BTreeMap<[usize; 3], ComponentFaces> =
        gather_component_faces(env, STREAM, 0, &grid).expect("every block wrote one");
    GlobalLabels::merge(&reports, &grid, connectivity).expect("the merge")
}

/// The **decorated** answer: read the local-label image through a
/// `RelabelledEnvironment` and look at what comes back.
///
/// Returns the volume and how many reads the decorator rewrote, so that the
/// liveness control has something to assert on.
fn decorated(
    mask: &Array3<bool>,
    block: [usize; 3],
    connectivity: Connectivity,
) -> (Array3<u32>, u64, usize) {
    let (env, _, _) = run_labelling(mask, block, connectivity);
    let map = Arc::new(table(&env, block, connectivity));
    let components = map.components() as usize;
    let view = RelabelledEnvironment::new(&env, 1usize, map);
    let whole = Region::whole(&VOLUME);
    let buf = view.read(1, &whole).expect("a decorated read");
    let blockflow::env::BlockBuf::Array(voxels) = buf else {
        unreachable!("an array environment answers with an array");
    };
    let out = voxels.view::<u32>().expect("labels").to_owned();
    (out, view.remapped_reads(), components)
}

/// The user's own construction: **a trivial materialiser over the decorator.**
/// One `CarryOp` phase, run against the decorated environment, writing an
/// ordinary image. If the two designs are one mechanism, this is the other one.
fn materialised_over_decorator(
    mask: &Array3<bool>,
    block: [usize; 3],
    connectivity: Connectivity,
) -> Array3<u32> {
    let (labelling, _, _) = run_labelling(mask, block, connectivity);
    let map = Arc::new(table(&labelling, block, connectivity));
    let local = labelling.output();

    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U32, grid);
    builder
        .pixels(Chain::op(CarryOp::new("materialise")))
        .expect("a pixel phase");
    let assembly = builder.finish().expect("a plan");

    let env = ArrayEnvironment::for_decomposition(local, &assembly.decomposition, [4, 4, 4])
        .expect("environment");
    let view = RelabelledEnvironment::new(&env, 0usize, map);
    execute_phases(
        "materialise",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &view,
        &[],
        &assembly.work(),
    )
    .expect("a run");
    assert!(
        view.remapped_reads() > 0,
        "the identity materialiser read nothing through the decorator"
    );
    env.output().view::<u32>().expect("labels").to_owned()
}

// ------------------------------------------------- the fixture and the grids --

/// **The grids are genuinely distinct and genuinely cut.** Run before anything
/// uses them; see the header for why this test exists at all.
#[test]
fn the_grids_are_genuinely_distinct_and_genuinely_cut() {
    let mut lattices: Vec<[usize; 3]> = Vec::new();
    let mut single = 0usize;
    let mut all_three = 0usize;
    let mut even = 0usize;
    for block in grids() {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let counts = grid.blocks_per_axis();
        assert!(
            !lattices.contains(&counts),
            "block {block:?} gives lattice {counts:?}, which another grid already gives — two \
             names for one grid is one grid"
        );
        lattices.push(counts);
        if counts == [1, 1, 1] {
            single += 1;
        }
        if counts.iter().all(|&n| n > 1) {
            all_three += 1;
        }
        if (0..3).all(|axis| VOLUME[axis] % block[axis] == 0) {
            even += 1;
        }
    }
    assert_eq!(
        single, 1,
        "the no-seam case belongs in the sweep exactly once"
    );
    assert!(
        all_three >= 3,
        "only {all_three} grids cut all three axes; lattice edges and corners need at least three"
    );
    assert!(
        even <= 1,
        "{even} grids divide the volume evenly, so a remainder block is not the ordinary case here"
    );
}

/// **The fixture has the shape the sweep needs.** Many components, none of them
/// the whole volume, and enough of them that a merge has work at every grid.
#[test]
fn the_fixture_has_the_shape_the_sweep_needs() {
    let mask = mask();
    let set = mask.iter().filter(|&&v| v).count();
    assert!(
        set > VOLUME.iter().product::<usize>() / 5,
        "the mask is too sparse to make components that cross seams"
    );
    for connectivity in [
        Connectivity::Faces,
        Connectivity::FacesAndEdges,
        Connectivity::FacesEdgesAndCorners,
    ] {
        let labels = reference(&mask, connectivity);
        let components = labels.iter().copied().max().unwrap_or(0);
        assert!(
            components > 1,
            "{connectivity:?} finds {components} component(s): a fixture that is one blob \
             cannot discriminate a merge"
        );
    }
    // and the three connectivities do not agree, so the parameter is reachable
    let six = reference(&mask, Connectivity::Faces);
    let twenty_six = reference(&mask, Connectivity::FacesEdgesAndCorners);
    assert_ne!(
        six.iter().copied().max(),
        twenty_six.iter().copied().max(),
        "the fixture answers the same at 6 and 26, so connectivity is not reached"
    );
}

// ------------------------------------------------------- the acceptance bar --

/// **The property this file exists for, for the materialising design.**
/// Byte-identical to the whole-volume reference at every grid and every
/// connectivity.
#[test]
fn the_materialised_label_volume_is_byte_identical_to_the_whole_volume_reference() {
    let mask = mask();
    for connectivity in [
        Connectivity::Faces,
        Connectivity::FacesAndEdges,
        Connectivity::FacesEdgesAndCorners,
    ] {
        let want = reference(&mask, connectivity);
        for block in grids() {
            assert_eq!(
                materialised(&mask, block, connectivity),
                want,
                "{connectivity:?} at blocking {block:?} disagreed with the whole volume"
            );
        }
    }
}

/// **The same property for the decorated design**, which is the one the
/// "one mechanism" argument favours and therefore the one that has to be at
/// least as correct.
#[test]
fn the_decorated_label_volume_is_byte_identical_to_the_whole_volume_reference() {
    let mask = mask();
    for connectivity in [
        Connectivity::Faces,
        Connectivity::FacesAndEdges,
        Connectivity::FacesEdgesAndCorners,
    ] {
        let want = reference(&mask, connectivity);
        let expect = want.iter().copied().max().unwrap_or(0) as usize;
        for block in grids() {
            let (got, remapped, components) = decorated(&mask, block, connectivity);
            assert_eq!(
                got, want,
                "{connectivity:?} at blocking {block:?} disagreed with the whole volume"
            );
            assert!(
                remapped > 0,
                "the decorator rewrote no read at blocking {block:?}, so the agreement above is \
                 an accident of the local labels"
            );
            assert_eq!(
                components, expect,
                "the table says {components} components and the reference has {expect}"
            );
        }
    }
}

/// **The two designs are one mechanism.** A trivial identity materialiser run
/// over the decorated environment produces the same bytes as the purpose-built
/// materialising phase — which is the user's own construction and the reason
/// there does not have to be two of these.
#[test]
fn an_identity_materialiser_over_the_decorator_writes_what_the_materialising_phase_writes() {
    let mask = mask();
    for connectivity in [Connectivity::Faces, Connectivity::FacesEdgesAndCorners] {
        let want = reference(&mask, connectivity);
        for block in grids() {
            assert_eq!(
                materialised_over_decorator(&mask, block, connectivity),
                want,
                "the identity materialiser disagreed at {block:?} under {connectivity:?}"
            );
        }
    }
}

// ----------------------------------------------------------- the liveness --

/// **The merge is load-bearing at every grid** — control 1 of the three the
/// header names.
///
/// If no component crossed a seam, the block-local labelling would already be
/// the answer and every agreement above would be vacuous. So: at every grid that
/// cuts anything, there are strictly more block-local labels than there are
/// components.
#[test]
fn the_merge_is_load_bearing_at_every_grid_that_cuts() {
    let mask = mask();
    let connectivity = Connectivity::Faces;
    let components = reference(&mask, connectivity)
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as usize;
    for block in grids() {
        let (env, _, _) = run_labelling(&mask, block, connectivity);
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let reports = gather_component_faces(&env, STREAM, 0, &grid).expect("every block");
        let local: usize = reports.values().map(|report| report.labels as usize).sum();
        if grid.blocks_per_axis() == [1, 1, 1] {
            assert_eq!(
                local, components,
                "the single-block grid has no seams, so its local count is the answer"
            );
        } else {
            assert!(
                local > components,
                "blocking {block:?} produced {local} local labels for {components} components, \
                 so nothing crossed a seam and this grid proves nothing"
            );
        }
    }
}

/// **The canonical numbering is load-bearing** — control 2.
///
/// A union-find root is a perfectly correct name for a component and it is a
/// **decomposition-dependent** one. This builds exactly that answer — the same
/// merge, numbered by root rather than by the component's least voxel — and
/// requires it to disagree with the reference. If it agreed, the sort in
/// `GlobalLabels::merge` would be doing nothing and the acceptance tests above
/// would pass without it.
#[test]
fn numbering_by_union_find_root_instead_disagrees_with_the_reference() {
    let mask = mask();
    let connectivity = Connectivity::Faces;
    let want = reference(&mask, connectivity);
    let mut disagreed = 0usize;
    for block in grids() {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        if grid.blocks_per_axis() == [1, 1, 1] {
            continue;
        }
        let (env, _, _) = run_labelling(&mask, block, connectivity);
        let reports = gather_component_faces(&env, STREAM, 0, &grid).expect("every block");
        let counts = grid.blocks_per_axis();
        let index = LabelIndex::build(&reports, counts, |report| report.labels).expect("an index");
        let mut sets = Union::new(index.total());
        blockflow::ops::components::walk_seams_with(
            &reports,
            counts,
            &index,
            connectivity,
            |report| &report.faces,
            |a, b| sets.union(a, b),
        )
        .expect("one lattice");

        let local = env.output();
        let local = local.view::<u32>().expect("labels");
        let edge = grid.block();
        let mut rooted = Array3::<u32>::zeros(local.raw_dim());
        for i in 0..VOLUME[0] {
            for j in 0..VOLUME[1] {
                for k in 0..VOLUME[2] {
                    let label = local[[i, j, k]];
                    if label == 0 {
                        continue;
                    }
                    let at = [i / edge[0], j / edge[1], k / edge[2]];
                    rooted[[i, j, k]] = sets.find(index.node(at, label)) as u32 + 1;
                }
            }
        }
        if rooted != want {
            disagreed += 1;
        }
        // and it is the *names* that differ, not the partition: the two agree
        // everywhere on which voxels are labelled at all
        for (left, right) in rooted.iter().zip(want.iter()) {
            assert_eq!(
                *left == 0,
                *right == 0,
                "the root numbering disagreed about which voxels are in a component at all, \
                 which would be a merge defect rather than a numbering one"
            );
        }
    }
    assert!(
        disagreed > 0,
        "numbering by union-find root agreed with the reference at every grid, so the canonical \
         sort in `GlobalLabels::merge` is not load-bearing and this suite would pass without it"
    );
}

/// **The decoration is load-bearing** — control 3.
///
/// The same read, through the same environment, *without* the decorator: it must
/// differ, or the local labels happened to be the global ones and every
/// agreement above is about nothing.
#[test]
fn reading_the_same_image_undecorated_gives_the_local_labels_and_not_the_answer() {
    let mask = mask();
    let connectivity = Connectivity::Faces;
    let want = reference(&mask, connectivity);
    for block in grids() {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        if grid.blocks_per_axis() == [1, 1, 1] {
            continue;
        }
        let (env, _, _) = run_labelling(&mask, block, connectivity);
        let plain = env.output().view::<u32>().expect("labels").to_owned();
        assert_ne!(
            plain, want,
            "at blocking {block:?} the undecorated local labels already equal the reference"
        );
    }
}

// ------------------------------------------------- the decorator's own bar --

/// **A decorated read is right at any extent, not only at a block.** The
/// decorator's whole claim is that the consumer's lattice need not be the
/// labelling's, so this reads the same volume back in windows that agree with no
/// block boundary — including windows that straddle three blocks on an axis —
/// and requires every one of them to be the corresponding sub-box of the answer.
#[test]
fn a_decorated_read_of_any_window_is_the_answer_restricted_to_that_window() {
    let mask = mask();
    let connectivity = Connectivity::FacesAndEdges;
    let want = reference(&mask, connectivity);
    let block = [5usize, 5, 5];
    let (env, _, _) = run_labelling(&mask, block, connectivity);
    let map = Arc::new(table(&env, block, connectivity));
    let view = RelabelledEnvironment::new(&env, 1usize, map);

    // deliberately not multiples of the block edge, and one of them longer than
    // two blocks on every axis
    let windows = [
        ([0usize, 0, 0], [3usize, 4, 7]),
        ([2, 3, 1], [13, 11, 17]),
        ([7, 9, 11], [8, 4, 12]),
        ([VOLUME[0] - 4, VOLUME[1] - 6, VOLUME[2] - 3], [4, 6, 3]),
    ];
    for (start, shape) in windows {
        let region = Region::new(&start, &shape);
        let buf = view.read(1, &region).expect("a decorated read");
        let blockflow::env::BlockBuf::Array(voxels) = buf else {
            unreachable!("an array environment answers with an array");
        };
        let got = voxels.view::<u32>().expect("labels");
        let expect = want.slice(s![
            start[0]..start[0] + shape[0],
            start[1]..start[1] + shape[1],
            start[2]..start[2] + shape[2],
        ]);
        assert_eq!(
            got, expect,
            "the window at {start:?}+{shape:?} disagreed with the reference"
        );
    }
}

/// **The table is small, and how small is the number the design turns on.** Not
/// an optimisation claim: the decorator's whole premise is that the
/// reconciliation between a decomposed labelling and a global one is a table
/// rather than a volume, and if the table were volume-sized the argument would
/// collapse. Asserted as a ratio so it does not rot into a recorded count.
#[test]
fn the_reconciliation_table_is_far_smaller_than_the_volume_it_reconciles() {
    let mask = mask();
    let connectivity = Connectivity::Faces;
    let voxel_bytes = VOLUME.iter().product::<usize>() * std::mem::size_of::<u32>();
    for block in grids() {
        let (env, _, _) = run_labelling(&mask, block, connectivity);
        let map = table(&env, block, connectivity);
        assert!(
            map.table_bytes() * 4 < voxel_bytes,
            "at blocking {block:?} the table is {} bytes against a {voxel_bytes}-byte label \
             volume, which is not the ratio the decorated design rests on",
            map.table_bytes()
        );
    }
}

/// **A table built on one lattice applied to labels written on another is
/// refused or wrong, and the refusal is what there is.**
///
/// This is the invariant the decorator adds and cannot check, written down as a
/// test so that it is a known property rather than a surprise: the table carries
/// the lattice it was built on, so a *shape* mismatch is caught — but a table
/// built on a different lattice of the same volume produces a well-formed wrong
/// answer, and here is that wrong answer, asserted to be wrong.
#[test]
fn a_table_from_another_lattice_is_a_well_formed_wrong_answer() {
    let mask = mask();
    let connectivity = Connectivity::Faces;
    let want = reference(&mask, connectivity);
    let (env, _, _) = run_labelling(&mask, [5, 5, 5], connectivity);
    // the table from a *different* cut of the same volume
    let (other, _, _) = run_labelling(&mask, [8, 7, 9], connectivity);
    let wrong = Arc::new(table(&other, [8, 7, 9], connectivity));
    let view = RelabelledEnvironment::new(&env, 1usize, wrong);
    let got = view.read(1, &Region::whole(&VOLUME));
    match got {
        Err(_) => {}
        Ok(blockflow::env::BlockBuf::Array(voxels)) => {
            let got = voxels.view::<u32>().expect("labels").to_owned();
            assert_ne!(
                got, want,
                "a table from another lattice produced the right answer, which would mean the \
                 lattice is not part of what the table means"
            );
        }
        Ok(_) => unreachable!("an array environment answers with an array"),
    }
}

// ------------------------------------------------- a real downstream consumer --

/// **The crate's own segmentation drives the crate's own per-object
/// measurement**, which is the thing the ops-survey index recorded as
/// unreachable: *"no op under `src/ops/` produces a label volume, while
/// `ops::tabulate`'s header opens 'One row per region of a label volume', so the
/// crate's most complete per-object measurement cannot be driven by the crate's
/// own segmentation."*
///
/// It can now, both ways round, and the two agree with each other and with a
/// tabulation of the whole-volume reference — which is the assertion that says
/// the decorated label volume is a label volume to a consumer that was written
/// before either design existed.
#[test]
fn tabulate_over_the_decorated_and_the_materialised_label_volume_agree_with_the_reference() {
    let mask = mask();
    let connectivity = Connectivity::Faces;
    let block = [8usize, 7, 9];

    let from_reference = tabulate_over(&reference(&mask, connectivity), block, None);
    let from_materialised = tabulate_over(&materialised(&mask, block, connectivity), block, None);

    // and the decorated one, which never has a global label volume anywhere: the
    // image on disk holds block-local labels and the consumer is handed the
    // global ones as it reads.
    let (env, _, _) = run_labelling(&mask, block, connectivity);
    let map = Arc::new(table(&env, block, connectivity));
    let local = env.output().view::<u32>().expect("labels").to_owned();
    let from_decorated = tabulate_over(&local, block, Some(map));

    assert!(
        from_reference.len() > 1,
        "a tabulation of one row cannot tell two label volumes apart"
    );
    assert_eq!(from_materialised, from_reference);
    assert_eq!(from_decorated, from_reference);

    // liveness: the same consumer over the same image *without* the decorator
    // does not agree, so the row above is the decoration's doing
    assert_ne!(tabulate_over(&local, block, None), from_reference);
}

/// `ops::tabulate` over a label volume, reduced to the `(label, count)` pairs —
/// which is all this comparison needs and is the part that is exactly an
/// integer.
///
/// **The label volume is a `supplied` array**, which is the arrangement a real
/// consumer is in: the labelling ran earlier, in its own plan, and this plan is
/// handed the result beside its own input. That was not expressible until
/// `TabulateValuesOp::holding` existed — a supplied array is produced by no
/// phase, so nothing in the plan could say what it held and `fragment_phase`
/// refused the pair by name.
///
/// `decorate` is the whole of the difference between the two arms: with a table,
/// the run happens over a `RelabelledEnvironment` decorating the supplied array
/// and the consumer sees global labels; without one, it sees whatever is in it.
fn tabulate_over(
    labels: &Array3<u32>,
    block: [usize; 3],
    decorate: Option<Arc<GlobalLabels>>,
) -> Vec<(u64, u64)> {
    const ROWS: &str = "rows";
    const PARTIALS: &str = "partials";
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let fixed = FixedPoint::default();
    let held = ImageId::supplied(0);

    let tabulate = TabulateValuesOp::new(
        "tabulate",
        held,
        0usize,
        fixed,
        PARTIALS,
        Lifecycle::DeleteOnExit,
    )
    .expect("two different images")
    .holding(Dtype::U32, Dtype::F64);
    let merge = MergeTabulationOp::new(
        "merge",
        PARTIALS,
        0,
        lattice,
        fixed,
        ROWS,
        Lifecycle::Persistent,
    );
    let plan = tabulate_phases(grid, Dtype::F64, &tabulate, &merge).expect("a plan");

    // The values are the mask the labels came from, widened. Any array would do
    // for a `(label, count)` comparison; this one is the one a caller has.
    let values: Voxels = labels
        .mapv(|label| if label == 0 { 0.0f64 } else { 1.0 })
        .into();
    let env = ArrayEnvironment::with_inputs(values, vec![labels.clone().into()], &plan, [4, 4, 4])
        .expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64);
    let work = [
        PhaseWork::Fragments(&tabulate),
        PhaseWork::Fragments(&merge),
    ];

    let rows = match decorate {
        None => {
            run_tabulation(&workflow, &plan, &work, &env);
            collect_tabulation(&env, ROWS, 1, VOLUME, fixed).expect("the rows")
        }
        Some(map) => {
            let view = RelabelledEnvironment::new(&env, held, map);
            run_tabulation(&workflow, &plan, &work, &view);
            assert!(
                view.remapped_reads() > 0,
                "the consumer read nothing through the decorator"
            );
            collect_tabulation(&view, ROWS, 1, VOLUME, fixed).expect("the rows")
        }
    };
    let mut out: Vec<(u64, u64)> = rows
        .into_iter()
        .map(|region| (region.label, region.count))
        .collect();
    out.sort_unstable();
    out
}

fn run_tabulation(
    workflow: &Workflow,
    plan: &Decomposition,
    work: &[PhaseWork<'_>],
    env: &dyn Environment,
) {
    execute_phases(
        "tabulate",
        workflow,
        plan,
        &Hints::default(),
        env,
        &[],
        work,
    )
    .expect("a run");
}

// ------------------------------------------------------------- the fragment --

/// A fragment survives a round trip, and a corrupted one is refused rather than
/// decoded into something plausible.
#[test]
fn a_component_fragment_survives_a_round_trip_and_a_wrong_one_is_refused() {
    let mask = mask();
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_members_into_with(
        VOLUME,
        Connectivity::Faces,
        |at| mask[at],
        labels.view_mut(),
    )
    .expect("a labelling");
    let first = blockflow::ops::label::first_voxels(labels.view(), count, [0, 0, 0], VOLUME);
    let faces = ComponentFaces::of(labels.view(), count, first).expect("a fragment");

    let bytes = faces.encode();
    assert_eq!(ComponentFaces::decode(&bytes).expect("a round trip"), faces);

    assert!(ComponentFaces::decode(&bytes[..bytes.len() - 4]).is_err());
    let mut wrong = bytes.clone();
    wrong[0] ^= 0xff;
    assert!(ComponentFaces::decode(&wrong).is_err());
    assert!(ComponentFaces::decode(&bytes[..3]).is_err());

    // and the empty report is a real report
    let empty = ComponentFaces::empty();
    assert_eq!(
        ComponentFaces::decode(&empty.encode()).expect("a round trip"),
        empty
    );
}

/// The two phases must agree about connectivity, and a plan whose halves
/// disagree is refused at planning time rather than discovered as a
/// decomposition-dependent answer.
#[test]
fn a_plan_whose_two_phases_disagree_about_connectivity_is_refused() {
    let grid = BlockGrid::new(VOLUME, [5, 5, 5]).expect("a lattice");
    let label = LabelComponentsOp::new("label", STREAM, Lifecycle::DeleteOnExit)
        .connecting(Connectivity::FacesEdgesAndCorners);
    let relabel =
        RelabelComponentsOp::new("relabel", STREAM, 0, &grid).connecting(Connectivity::Faces);
    assert!(component_label_phases(grid, Dtype::Bool, &label, &relabel).is_err());
}

/// **A supplied label volume needs its element type declared, and the plan
/// refuses it when it is not.**
///
/// The liveness control for `TabulateValuesOp::holding`: the same plan with the
/// one thing changed. Without it the consumer above could not be built at all —
/// a supplied array is produced by no phase, so no fold of the plan can say what
/// it holds and `fragment_phase` refuses the pair by name rather than fetching
/// some width and reading another.
///
/// The second half is the part that makes this a control rather than a
/// restatement: an operand that is an **image of the plan** must *not* need one,
/// because its element type is in the fold of the chain that wrote it and a
/// second copy would be a second number to drift.
#[test]
fn a_supplied_label_volume_is_refused_until_its_element_type_is_declared() {
    const PARTIALS: &str = "partials";
    let grid = BlockGrid::new(VOLUME, [5, 5, 5]).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let fixed = FixedPoint::default();
    let merge = MergeTabulationOp::new(
        "merge",
        PARTIALS,
        0,
        lattice,
        fixed,
        "rows",
        Lifecycle::Persistent,
    );
    let undeclared = |labels: ImageId| {
        TabulateValuesOp::new(
            "tabulate",
            labels,
            0usize,
            fixed,
            PARTIALS,
            Lifecycle::DeleteOnExit,
        )
        .expect("two different images")
    };

    // supplied and silent: refused, and the message names the image
    let refusal = tabulate_phases(
        grid.clone(),
        Dtype::F64,
        &undeclared(ImageId::supplied(0)),
        &merge,
    )
    .expect_err("a supplied operand with no declared width is not plannable");
    let text = refusal.to_string();
    assert!(
        text.contains("supplied input 0"),
        "the refusal does not name the image: {text}"
    );

    // supplied and declared: plannable
    tabulate_phases(
        grid.clone(),
        Dtype::F64,
        &undeclared(ImageId::supplied(0)).holding(Dtype::U32, Dtype::F64),
        &merge,
    )
    .expect("a declared supplied operand is plannable");

    // an image of the plan, silent: plannable, and that is the correct default
    tabulate_phases(grid, Dtype::F64, &undeclared(ImageId::from(1usize)), &merge)
        .expect("an image the run writes has its width in the fold");
}
