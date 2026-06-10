// NEON-shaped sparse index extraction: 32 bytes per iteration become an 8-bit
// group mask that picks a precomputed SparseEntry, replacing the previous
// scalar group scan. Stores past `count` write scratch lanes that later
// iterations overwrite; the final store ends exactly at L1_SIZE / 4 entries.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub unsafe fn find_nnz(
    ft_out: &Aligned<[u8; L1_SIZE]>, nnz_table: &[SparseEntry],
) -> (Aligned<[u16; L1_SIZE / 4]>, usize) {
    use std::arch::wasm32::*;

    let mut indexes = Aligned::new([0; L1_SIZE / 4]);
    let mut count = 0;

    let increment = i16x8_splat(8);
    let mut base = i16x8_splat(0);

    for i in (0..L1_SIZE).step_by(32) {
        let v0 = *ft_out.as_ptr().add(i).cast();
        let v1 = *ft_out.as_ptr().add(i + 16).cast();

        let mask = (simd::nnz_bitmask(v0) | (simd::nnz_bitmask(v1) << 4)) as usize;
        let entry = nnz_table.get_unchecked(mask);

        let indexed = i16x8_add(base, v128_load(entry.indexes.as_ptr().cast()));
        v128_store(indexes.as_mut_ptr().add(count).cast(), indexed);

        count += entry.count;
        base = i16x8_add(base, increment);
    }

    (indexes, count)
}
