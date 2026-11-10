/// SIMD-group records for `cavities_packed`: variables in CSR order, each on a
/// power-of-two lane block that covers its degree, aligned to that size.
/// Variables of higher degree take one start lane and close their record.
pub(crate) fn warp_plan(var_offsets: &[u32]) -> Vec<[u32; 4]> {
    let n = var_offsets.len() / 2;
    let mut records: Vec<[u32; 4]> = Vec::new();
    let mut offset = 32u32;
    for v in 0..n {
        let degree = var_offsets[2 * v + 2] - var_offsets[2 * v];
        let positives = var_offsets[2 * v + 1] - var_offsets[2 * v];
        let small = degree <= 32;
        let width = if small {
            degree.max(1).next_power_of_two()
        } else {
            1
        };
        offset = offset.div_ceil(width) * width;
        if offset + width > 32 {
            records.push([v as u32, 0, 0, 0]);
            offset = 0;
        }
        let record = records.last_mut().unwrap();
        record[1] |= 1 << offset;
        if small {
            record[2] |= mask(degree, offset);
            record[3] |= mask(positives, offset);
            offset += width;
        } else {
            record[3] |= 1 << offset;
            offset = 32;
        }
    }
    records
}

fn mask(bits: u32, offset: u32) -> u32 {
    if bits == 0 {
        0
    } else {
        (u32::MAX >> (32 - bits)) << offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_align_blocks_and_close_after_wide_variables() {
        // Degrees 3 (2 positive), 5, 40, 1: blocks of 4 and 8, then a larger
        // start lane that closes the record, then a new record.
        let offsets = [0, 2, 3, 6, 8, 30, 48, 48, 49];
        assert_eq!(
            warp_plan(&offsets),
            vec![
                [
                    0,
                    1 | 1 << 8 | 1 << 16,
                    0b111 | 0b11111 << 8,
                    0b11 | 0b111 << 8 | 1 << 16
                ],
                [3, 1, 0b1, 0],
            ]
        );
    }
}
