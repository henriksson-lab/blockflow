use blockflow::ops::fft::{correlate_direct, Correlation2, Padding, ShiftWindow};
use ndarray::Array2;
#[test]
fn prepared_operands_match_direct_lags_and_are_not_consumed_by_inverse() {
    let a = Array2::from_shape_fn((7, 11), |(y, x)| {
        ((x * 13 + y * 7 + x * y) % 23) as f64 - 8.
    });
    let b = Array2::from_shape_fn((4, 5), |(y, x)| {
        ((x * 3 + y * 11) % 17) as f64 + 0.2 * y as f64
    });
    let window = ShiftWindow::new([-3, -4], [6, 10]).unwrap();
    let mut plan = Correlation2::new([7, 11], [4, 5], window, Padding::Smooth).unwrap();
    let pa = plan.prepare_operand(a.view()).unwrap();
    let pb = plan.prepare_operand(b.view()).unwrap();
    assert!(pa.bytes() > 0);
    let expected = correlate_direct(a.view(), b.view(), window);
    let legacy = plan.correlate(a.view(), b.view()).unwrap();
    for _ in 0..3 {
        let result = plan.correlate_prepared(&pa, &pb).unwrap();
        for ((x, y), z) in result.iter().zip(&expected).zip(&legacy) {
            assert!((x - y).abs() < 1e-9);
            assert!((x - z).abs() < 1e-9);
        }
    }
    // Preparing a different reference does not mutate the cached template.
    let changed = a.mapv(|x| x * 0.7 + 2.);
    let pc = plan.prepare_operand(changed.view()).unwrap();
    let result = plan.correlate_prepared(&pc, &pb).unwrap();
    let expected = correlate_direct(changed.view(), b.view(), window);
    assert!(result
        .iter()
        .zip(expected)
        .all(|(x, y)| (x - y).abs() < 1e-9));
}
#[test]
fn prepared_spectra_reject_wrong_padding_shapes_and_roles() {
    let a = Array2::ones((7, 11));
    let b = Array2::ones((4, 5));
    let window = ShiftWindow::new([-3, -4], [6, 10]).unwrap();
    let mut plan = Correlation2::new([7, 11], [4, 5], window, Padding::Smooth).unwrap();
    let pa = plan.prepare_operand(a.view()).unwrap();
    let pb = plan.prepare_operand(b.view()).unwrap();
    assert!(plan.correlate_prepared(&pb, &pa).is_err());
    let mut other = Correlation2::new([7, 11], [4, 5], window, Padding::Exact([32, 32])).unwrap();
    let po = other.prepare_operand(b.view()).unwrap();
    assert!(plan.correlate_prepared(&pa, &po).is_err());
    assert!(plan.prepare_operand(Array2::ones((3, 3)).view()).is_err());
    assert!(plan
        .correlate_prepared_into(&pa, &pb, &mut Array2::zeros((1, 1)))
        .is_err());
}
