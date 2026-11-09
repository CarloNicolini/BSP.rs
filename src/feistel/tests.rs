
use super::*;
#[test]
fn exhaustive_permutations() {
    for m in 1..=260 {
        for seed in [0, 1, 42, u64::MAX] {
            let f = FeistelTopology::new(3, m, 100, seed).unwrap();
            for q in 0..3 {
                let mut seen = vec![false; m as usize];
                let mut steps = 0;
                for a in 0..m {
                    let (t, count) = f.walk(a, q, false);
                    assert!(!seen[t as usize]);
                    seen[t as usize] = true;
                    assert_eq!(f.walk(t, q, true).0, a);
                    assert_eq!(f.walk(f.walk(a, q, true).0, q, false).0, a);
                    steps += count;
                }
                assert!(steps <= 1u64 << (2 * f.matching.bits));
            }
        }
    }
}
#[test]
fn round_tables_preserve_both_directions_and_walk_lengths() {
    for limit in [1, 17, 103, 1024, 6000, 65536] {
        let permutation = Permutation::new(limit);
        let keys = Permutation::round_keys(91);
        let table = permutation.round_table(&keys);
        assert!(table.len() <= 2048);
        for index in 0..limit {
            for inverse in [false, true] {
                assert_eq!(
                    permutation.walk_with_keys(index, &keys, inverse),
                    permutation.walk_with_table(index, &table, inverse),
                );
            }
        }
    }
    assert!(Permutation::new(65537).round_table(&[0; 8]).is_empty());
}

#[test]
fn invalid_dimensions() {
    for (k, m, n) in [
        (0, 1, 1),
        (2, 1, 1),
        (65, 1, 65),
        (1, 0, 1),
        (3, u64::MAX, 100),
    ] {
        assert!(FeistelTopology::new(k, m, n, 0).is_err());
    }
    let f = FeistelTopology::new(1, u64::MAX, 1, 0).unwrap();
    for a in [0, 1, u64::MAX - 1] {
        assert_eq!(f.walk(f.walk(a, 0, false).0, 0, true).0, a);
    }
}
