// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! RaBitQ 1-bit FastScan for residual sign codes.
//!
//! Each MRQ level is a packed `±1` code in the same 32-row nibble layout as
//! IVF_RQ. One query builds the distance table; every level reuses it.
//! `⟨±1, q⟩ = 2 * FastScan(q) - Σ qᵢ`.
//!
//! The LUT is the u16 table used by RaBitQ accurate mode. An 8-bit table is
//! enough for a prune bound, but MRQ ranks by this estimate, and that error
//! moves the top-k.

use arrow::array::AsArray;
use arrow_array::types::{Float32Type, UInt8Type};
use arrow_array::{FixedSizeListArray, UInt8Array};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::Result;
use lance_linalg::simd::dist_table::{
    BATCH_SIZE, sum_4bit_hacc_dist_table, transfer_4bit_dist_table_u16,
};

use super::super::bq::dist_table_quant::{DistTableDequant, quantize_dist_table_u16_into};
use super::super::bq::storage::{build_dist_table_direct, pack_codes, rabit_packed_code_ip};

pub(super) struct QueryLut {
    dist_table: Vec<f32>,
    hacc_table: Vec<u8>,
    qmin: f32,
    qmax: f32,
    quantize: bool,
    pub(super) sum_q: f32,
    pub(super) query_sq: f32,
}

impl QueryLut {
    pub(super) fn build(rotated_query: &[f32]) -> Self {
        let dist_table = build_dist_table_direct::<Float32Type>(rotated_query);
        let mut quantized = Vec::new();
        let dequant = quantize_dist_table_u16_into(&dist_table, &mut quantized);
        let (qmin, qmax, quantize) = match dequant {
            DistTableDequant::Affine { qmin, qmax } => (qmin, qmax, true),
            DistTableDequant::Exact => (0.0, 0.0, false),
        };
        let mut hacc_table = Vec::new();
        if quantize {
            transfer_4bit_dist_table_u16(&quantized, &mut hacc_table);
        }
        let sum_q = rotated_query.iter().copied().sum();
        let query_sq = rotated_query.iter().map(|value| value * value).sum();
        Self {
            dist_table,
            hacc_table,
            qmin,
            qmax,
            quantize,
            sum_q,
            query_sq,
        }
    }

    /// Worst-case `|⟨±1, q⟩_hat - ⟨±1, q⟩|` of the u16 table.
    pub(super) fn pm1_error(&self) -> f32 {
        if self.quantize && self.qmax > self.qmin {
            let num_tables = (self.dist_table.len() / 16) as f32;
            let ip_error = num_tables * 0.5 * (self.qmax - self.qmin) / 65535.0;
            2.0 * ip_error
        } else {
            0.0
        }
    }

    fn reconstruct(&self, code_sum: f32) -> f32 {
        let range = if self.qmax > self.qmin {
            (self.qmax - self.qmin) / 65535.0
        } else {
            0.0
        };
        let num_tables = (self.dist_table.len() / 16) as f32;
        code_sum * range + num_tables * self.qmin
    }
}

pub(super) fn supports_fastscan(dim: usize) -> bool {
    dim.is_multiple_of(8) && dim >= 8
}

/// Pack each level into the RaBitQ FastScan layout. `codes` is row-major and
/// level-major inside the row.
pub(super) fn pack_level_codes(
    codes: &[u8],
    rows: usize,
    levels: usize,
    code_bytes: usize,
) -> Result<Vec<u8>> {
    let mut packed = vec![0u8; codes.len()];
    for level in 0..levels {
        let mut level_bytes = Vec::with_capacity(rows * code_bytes);
        for row in 0..rows {
            let start = (row * levels + level) * code_bytes;
            level_bytes.extend_from_slice(&codes[start..start + code_bytes]);
        }
        let list = FixedSizeListArray::try_new_from_values(
            UInt8Array::from(level_bytes),
            code_bytes as i32,
        )?;
        let packed_level = pack_codes(&list);
        let values = packed_level.values().as_primitive::<UInt8Type>().values();
        let dst = level * rows * code_bytes;
        packed[dst..dst + values.len()].copy_from_slice(values);
    }
    Ok(packed)
}

pub(super) fn scan_packed_ips(
    packed_level: &[u8],
    rows: usize,
    code_bytes: usize,
    lut: &QueryLut,
    ips: &mut [f32],
) {
    let simd_len = if lut.quantize {
        rows - rows % BATCH_SIZE
    } else {
        0
    };
    if simd_len > 0 {
        let codes = &packed_level[..simd_len * code_bytes];
        let mut acc = vec![0u32; simd_len];
        sum_4bit_hacc_dist_table(simd_len, code_bytes, codes, &lut.hacc_table, &mut acc);
        for (ip, code_sum) in ips.iter_mut().zip(acc) {
            *ip = lut.reconstruct(code_sum as f32);
        }
    }
    for (id, ip) in ips.iter_mut().enumerate().take(rows).skip(simd_len) {
        *ip = rabit_packed_code_ip(packed_level, id, rows, code_bytes, &lut.dist_table);
    }
}

pub(super) fn pm1(ip: f32, sum_q: f32) -> f32 {
    2.0 * ip - sum_q
}
