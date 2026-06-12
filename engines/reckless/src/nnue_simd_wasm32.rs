use std::{arch::wasm32::*, mem::size_of};

pub const F32_LANES: usize = size_of::<v128>() / size_of::<f32>();
pub const I32_LANES: usize = size_of::<v128>() / size_of::<i32>();
pub const I16_LANES: usize = size_of::<v128>() / size_of::<i16>();
pub const MUL_HI_SHIFT: i32 = 0;

pub fn add_i16(a: v128, b: v128) -> v128 { i16x8_add(a, b) }
pub fn sub_i16(a: v128, b: v128) -> v128 { i16x8_sub(a, b) }

pub unsafe fn zeroed() -> v128 { i32x4_splat(0) }
pub unsafe fn splat_i16(a: i16) -> v128 { i16x8_splat(a) }
pub unsafe fn clamp_i16(x: v128, min: v128, max: v128) -> v128 { i16x8_max(i16x8_min(x, max), min) }
pub unsafe fn min_i16(a: v128, b: v128) -> v128 { i16x8_min(a, b) }
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: v128) -> v128 { i16x8_shl(a, SHIFT as u32) }

pub unsafe fn mul_high_i16(a: v128, b: v128) -> v128 {
    let lo = i32x4_shr(i32x4_extmul_low_i16x8(a, b), 16);
    let hi = i32x4_shr(i32x4_extmul_high_i16x8(a, b), 16);
    i16x8_narrow_i32x4(lo, hi)
}

pub unsafe fn convert_i8_i16(a: i64) -> v128 { i16x8_extend_low_i8x16(i64x2(a, 0)) }
pub unsafe fn packus(a: v128, b: v128) -> v128 { u8x16_narrow_i16x8(a, b) }
pub unsafe fn permute(a: v128) -> v128 { a }
pub unsafe fn splat_i32(a: i32) -> v128 { i32x4_splat(a) }
pub unsafe fn zero_f32() -> v128 { f32x4_splat(0.0) }
pub unsafe fn splat_f32(a: f32) -> v128 { f32x4_splat(a) }
// Relaxed f32 madd/min/max were tried and removed: parity held but they gave
// no win on x86_64 and regressed on Apple Silicon Chromium. The integer dot
// product below is the only op where relaxed SIMD has real upside here.
pub unsafe fn mul_add_f32(a: v128, b: v128, c: v128) -> v128 { f32x4_add(f32x4_mul(a, b), c) }
pub unsafe fn convert_to_f32(a: v128) -> v128 { f32x4_convert_i32x4(a) }
pub unsafe fn clamp_f32(x: v128, min: v128, max: v128) -> v128 { f32x4_max(f32x4_min(x, max), min) }

// activate_ft output is provably in [0, 127]: lhs is clamped to [0, FT_QUANT],
// products are lhs * rhs >> FT_SHIFT = 255 * 255 >> 9 = 127 max, and packus
// saturates negatives to zero. That satisfies the i7x16 operand precondition,
// so the relaxed dot is exact (not implementation-defined) on every lowering,
// and intermediate i16 pair sums (max 2 * 127 * 127 = 32258) cannot saturate
// pmaddubsw-style x86 lowerings. On ARM this lowers to a single SDOT.
#[cfg(target_feature = "relaxed-simd")]
unsafe fn dpbusd_once(i32s: v128, u8s: v128, i8s: v128) -> v128 {
    i32x4_relaxed_dot_i8x16_i7x16_add(i8s, u8s, i32s)
}

#[cfg(not(target_feature = "relaxed-simd"))]
unsafe fn dpbusd_once(i32s: v128, u8s: v128, i8s: v128) -> v128 {
    let prod_lo = i16x8_mul(u16x8_extend_low_u8x16(u8s), i16x8_extend_low_i8x16(i8s));
    let prod_hi = i16x8_mul(u16x8_extend_high_u8x16(u8s), i16x8_extend_high_i8x16(i8s));
    let pair_lo = i32x4_extadd_pairwise_i16x8(prod_lo);
    let pair_hi = i32x4_extadd_pairwise_i16x8(prod_hi);
    // Group sums of four byte products via two shuffles + add, staying in
    // SIMD registers; lane extraction here scalarized the hottest NNUE loop.
    let sums = i32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(pair_lo, pair_hi),
        i32x4_shuffle::<1, 3, 5, 7>(pair_lo, pair_hi),
    );
    i32x4_add(i32s, sums)
}

pub unsafe fn dpbusd(i32s: v128, u8s: v128, i8s: v128) -> v128 { dpbusd_once(i32s, u8s, i8s) }
pub unsafe fn double_dpbusd(i32s: v128, u8s1: v128, i8s1: v128, u8s2: v128, i8s2: v128) -> v128 {
    dpbusd_once(dpbusd_once(i32s, u8s1, i8s1), u8s2, i8s2)
}

pub unsafe fn horizontal_sum(x: [v128; 4]) -> f32 {
    let mut sum = 0.0;
    for vector in x {
        sum += f32x4_extract_lane::<0>(vector);
        sum += f32x4_extract_lane::<1>(vector);
        sum += f32x4_extract_lane::<2>(vector);
        sum += f32x4_extract_lane::<3>(vector);
    }
    sum
}

// One bit per i32 group, matching the avx2/neon contract that callers combine
// as mask0 | mask1 << 4 to index the 256-entry nnz_table. ne-zero (not gt)
// keeps it correct even if a packed group's high byte ever sets the sign bit.
pub unsafe fn nnz_bitmask(x: v128) -> u16 { i32x4_bitmask(i32x4_ne(x, i32x4_splat(0))) as u16 }
