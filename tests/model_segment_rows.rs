//! **The row fold is associative and commutative, in the type it folds in.**
//!
//! Every column of a row is a `u64` merged by `+`, `min` or `max`, and that is
//! not a stylistic preference — it is what makes a consumer's answer a function
//! of the objects rather than of the order it happened to visit them in. The
//! same discipline, for the same reason, as `blockflow::ops::tabulate`.
//!
//! What would break it, concretely, is one `f64` column. Floating-point
//! addition does not associate, so folding three partial readings of one object
//! would give a different last bit depending on which two were folded first —
//! and the run that produced them is a run over blocks, so which two were folded
//! first is a fact about the schedule. The tests here would start failing
//! intermittently, which is why they are written as *exact* equality over
//! permutations rather than as an approximate comparison.
//!
//! No image, no backend and no execution: this is arithmetic over the schema.

use blockflow::model_segment::{centroid, fold, measured_column, schema, FIXED_COLUMNS};

const MEASURED: usize = 3;

/// A row with distinguishable values in every column, so that a fold that
/// crossed two columns would be visible rather than cancelling out.
fn row(id: u64, seed: u64) -> Vec<u64> {
    let mut values = vec![
        id,
        seed,      // count
        seed * 11, // sum_0
        seed * 13, // sum_1
        seed * 17, // sum_2
    ];
    for image in 0..MEASURED {
        let step = (image as u64 + 1) * 19;
        values.push(seed * step); // sum
        values.push(seed % step); // min
        values.push(seed * step + image as u64); // max
    }
    values
}

fn fold_all(rows: &[Vec<u64>]) -> Vec<u64> {
    let mut folded = rows[0].clone();
    for next in &rows[1..] {
        folded = fold(MEASURED, &folded, next).expect("a fold");
    }
    folded
}

#[test]
fn the_schema_and_the_fold_agree_on_how_wide_a_row_is() {
    let schema = schema(MEASURED).expect("a schema");
    assert_eq!(schema.len(), FIXED_COLUMNS + MEASURED * 3);
    assert_eq!(row(1, 5).len(), schema.len());

    // The column names are what a consumer looks a value up by, so the offset
    // helper and the schema must not be able to drift apart.
    for image in 0..MEASURED {
        assert_eq!(
            schema.index_of(&format!("sum_c{image}")),
            Some(measured_column(image))
        );
        assert_eq!(
            schema.index_of(&format!("min_c{image}")),
            Some(measured_column(image) + 1)
        );
        assert_eq!(
            schema.index_of(&format!("max_c{image}")),
            Some(measured_column(image) + 2)
        );
    }
}

/// `fold(a, b) == fold(b, a)`, exactly.
#[test]
fn the_fold_is_commutative() {
    let a = row(7, 5);
    let b = row(7, 12);
    assert_eq!(
        fold(MEASURED, &a, &b).expect("a fold"),
        fold(MEASURED, &b, &a).expect("a fold")
    );
}

/// `fold(fold(a, b), c) == fold(a, fold(b, c))`, exactly — and the same for
/// every order the three could arrive in.
#[test]
fn the_fold_is_associative_and_order_independent() {
    let rows = [row(9, 3), row(9, 40), row(9, 17), row(9, 8)];

    let left_to_right = fold_all(&rows);

    // Every permutation of four rows, folded left to right, and one explicit
    // re-association — which is the half a permutation test does not cover.
    let mut orders: Vec<Vec<usize>> = Vec::new();
    permute(&mut vec![0, 1, 2, 3], 0, &mut orders);
    assert_eq!(orders.len(), 24);
    for order in orders {
        let permuted: Vec<Vec<u64>> = order.iter().map(|&index| rows[index].clone()).collect();
        assert_eq!(
            fold_all(&permuted),
            left_to_right,
            "folding in order {order:?} gave a different answer"
        );
    }

    let paired = fold(
        MEASURED,
        &fold(MEASURED, &rows[0], &rows[1]).expect("a fold"),
        &fold(MEASURED, &rows[2], &rows[3]).expect("a fold"),
    )
    .expect("a fold");
    assert_eq!(paired, left_to_right, "re-association changed the answer");
}

/// What the fold actually computes, column by column, checked against numbers
/// rather than against a second implementation of the same loop.
#[test]
fn the_fold_adds_the_sums_and_takes_the_extremes() {
    let a = row(4, 6);
    let b = row(4, 25);
    let folded = fold(MEASURED, &a, &b).expect("a fold");

    assert_eq!(folded[0], 4, "the id is the key and is carried");
    for column in 1..FIXED_COLUMNS {
        assert_eq!(folded[column], a[column] + b[column], "column {column}");
    }
    for image in 0..MEASURED {
        let base = measured_column(image);
        assert_eq!(folded[base], a[base] + b[base], "sum_c{image}");
        assert_eq!(
            folded[base + 1],
            a[base + 1].min(b[base + 1]),
            "min_c{image}"
        );
        assert_eq!(
            folded[base + 2],
            a[base + 2].max(b[base + 2]),
            "max_c{image}"
        );
    }
}

/// Two rows for two different objects are not two readings of one, and folding
/// them would add together things that are genuinely different.
#[test]
fn folding_across_ids_is_refused_by_name() {
    let error = fold(MEASURED, &row(1, 5), &row(2, 5))
        .expect_err("two objects are not one")
        .to_string();
    assert!(
        error.contains("not two readings of one object"),
        "got: {error}"
    );
}

/// A row of the wrong width is refused rather than read past its end.
#[test]
fn a_row_of_the_wrong_width_is_refused() {
    let short = row(1, 5)[..4].to_vec();
    let error = fold(MEASURED, &short, &row(1, 5))
        .expect_err("a short row")
        .to_string();
    assert!(error.contains("columns"), "got: {error}");
}

/// The centroid comes back as the two integers the row holds, so the division
/// is the caller's and is made once.
#[test]
fn the_centroid_is_the_sums_over_the_count() {
    let values = row(3, 9);
    let (sums, count) = centroid(&values);
    assert_eq!(count, 9);
    assert_eq!(sums, [9 * 11, 9 * 13, 9 * 17]);

    // And it folds: two halves of one object give the whole object's centre.
    let folded = fold(MEASURED, &row(3, 4), &row(3, 5)).expect("a fold");
    let (folded_sums, folded_count) = centroid(&folded);
    assert_eq!(folded_count, 9);
    assert_eq!(
        folded_sums, sums,
        "the sums add, so the centre is the whole's"
    );
}

/// Every permutation of `items`, appended to `out`.
fn permute(items: &mut Vec<usize>, at: usize, out: &mut Vec<Vec<usize>>) {
    if at == items.len() {
        out.push(items.clone());
        return;
    }
    for index in at..items.len() {
        items.swap(at, index);
        permute(items, at + 1, out);
        items.swap(at, index);
    }
}
