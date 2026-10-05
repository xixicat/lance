// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Multi-level residual 1-bit quantization (`IVF_MRQ`).
//!
//! One Fast rotation is shared by the index. Each vector is then encoded as
//! 1 to 8 residual sign codes in that rotated space. Joint least squares
//! picks the scales, then one multiplier makes `⟨r, r̂⟩ = ||r||²`. Search
//! scores those signs with the RaBitQ 1-bit FastScan kernel.

use std::collections::BinaryHeap;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, UInt8Type, UInt64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array};
use arrow_schema::{DataType, Field, SchemaRef};
use async_trait::async_trait;
use half::f16;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ID, Result};
use lance_file::versions::v1::reader::FileReader as V1FileReader;
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use super::bq::residual_levels::{ResidualEncoder, dot_packed_pm1};
use super::bq::rotation::random_fast_rotation_signs;
use super::graph::OrderedNode;
use super::quantizer::{
    Quantization, QuantizationMetadata, QuantizationType, Quantizer, QuantizerBuildParams,
    QuantizerMetadata, QuantizerStorage,
};
use super::storage::{
    DistCalculator, DistanceCalculatorOptions, QueryResidual, VectorStore,
    accumulate_distances_into_heap,
};
use super::transform::Transformer;
use crate::frag_reuse::FragReuseIndex;
use crate::pb::vector_index_details::MultiResidualQuantization;
use crate::scalar::RowIdRemapper;

mod scan;

pub const MRQ_METADATA_KEY: &str = "lance:mrq";
pub const MRQ_CODE_COLUMN: &str = "__mrq_codes";
pub const MRQ_ALPHA_COLUMN: &str = "__mrq_alpha";
pub const MRQ_NORM_SQ_COLUMN: &str = "__mrq_norm_sq";

/// Uncompressed payload bytes for one row, including the 8-byte row id.
///
/// Codes, one `f32` scale per level, and one `f32` norm. Null bitmaps and
/// Lance page compression are not included.
pub fn logical_row_bytes(dim: usize, levels: usize) -> usize {
    std::mem::size_of::<u64>()
        + levels * dim.div_ceil(8)
        + std::mem::size_of::<f32>() * levels
        + std::mem::size_of::<f32>()
}

/// Build parameters for [`MrqQuantizer`].
#[derive(Debug, Clone)]
pub struct MrqBuildParams {
    /// Number of residual 1-bit levels. Valid range is `1..=8`.
    pub levels: u8,
    /// Shared Fast-rotation signs. Generated at build time when absent.
    pub signs: Option<Vec<u8>>,
}

impl MrqBuildParams {
    pub fn new(levels: u8) -> Result<Self> {
        if !(1..=8).contains(&levels) {
            return Err(Error::invalid_input(format!(
                "IVF_MRQ levels must be in 1..=8, got {levels}"
            )));
        }
        Ok(Self {
            levels,
            signs: None,
        })
    }
}

impl QuantizerBuildParams for MrqBuildParams {
    fn sample_size(&self) -> usize {
        0
    }
}

impl From<&MrqBuildParams> for MultiResidualQuantization {
    fn from(params: &MrqBuildParams) -> Self {
        Self {
            levels: u32::from(params.levels),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MrqQuantizationMetadata {
    pub dim: u32,
    pub levels: u8,
    pub signs: Vec<u8>,
}

impl DeepSizeOf for MrqQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.signs.capacity()
    }
}

#[async_trait]
impl QuantizerMetadata for MrqQuantizationMetadata {
    async fn load(reader: &V1FileReader) -> Result<Self> {
        let metadata_str = reader
            .schema()
            .metadata
            .get(MRQ_METADATA_KEY)
            .ok_or(Error::index(format!(
                "Reading MRQ metadata: metadata key {MRQ_METADATA_KEY} not found"
            )))?;
        serde_json::from_str(metadata_str)
            .map_err(|_| Error::index(format!("Failed to parse MRQ metadata: {metadata_str}")))
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct MrqQuantizer {
    metadata: MrqQuantizationMetadata,
}

impl MrqQuantizer {
    pub fn metadata_ref(&self) -> &MrqQuantizationMetadata {
        &self.metadata
    }

    pub fn levels(&self) -> u8 {
        self.metadata.levels
    }

    pub fn dim(&self) -> usize {
        self.metadata.dim as usize
    }

    fn code_bytes(&self) -> usize {
        self.dim().div_ceil(8)
    }

    fn code_width(&self) -> usize {
        self.levels() as usize * self.code_bytes()
    }

    fn encoder(&self) -> Result<ResidualEncoder> {
        ResidualEncoder::with_signs(self.dim(), self.metadata.signs.clone())
    }

    fn encode_rows(&self, values: &[f32], rows: usize) -> Result<EncodedColumns> {
        let dim = self.dim();
        let levels = self.levels() as usize;
        let code_bytes = self.code_bytes();
        let width = self.code_width();
        if values.len() != rows * dim {
            return Err(Error::invalid_input(format!(
                "IVF_MRQ expected {} values for {rows} x {dim}, got {}",
                rows * dim,
                values.len()
            )));
        }
        let encoder = self.encoder()?;
        let mut codes = vec![0u8; rows * width];
        let mut alpha = Vec::with_capacity(rows * levels);
        let mut norm_sq = Vec::with_capacity(rows);
        for row in 0..rows {
            let vector = &values[row * dim..(row + 1) * dim];
            let encoded = encoder.encode(vector, levels, true)?;
            if encoded.levels.len() != levels {
                return Err(Error::internal(format!(
                    "IVF_MRQ encoded {} levels, expected {levels}",
                    encoded.levels.len()
                )));
            }
            for (level_idx, level) in encoded.levels.iter().enumerate() {
                let start = row * width + level_idx * code_bytes;
                if level.packed.len() != code_bytes {
                    return Err(Error::internal(format!(
                        "IVF_MRQ packed len {}, expected {code_bytes}",
                        level.packed.len()
                    )));
                }
                codes[start..start + code_bytes].copy_from_slice(&level.packed);
                alpha.push(level.alpha);
            }
            norm_sq.push(encoded.norm_sq);
        }
        Ok(EncodedColumns {
            codes,
            alpha,
            norm_sq,
        })
    }

    fn quantize_values(&self, values: &[f32], rows: usize) -> Result<ArrayRef> {
        let encoded = self.encode_rows(values, rows)?;
        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(encoded.codes),
            self.code_width() as i32,
        )?))
    }
}

struct EncodedColumns {
    codes: Vec<u8>,
    alpha: Vec<f32>,
    norm_sq: Vec<f32>,
}

fn f32_values(array: &dyn Array) -> Result<Vec<f32>> {
    match array.data_type() {
        DataType::Float16 => Ok(array
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .map(|value| f16::to_f32(*value))
            .collect()),
        DataType::Float32 => Ok(array.as_primitive::<Float32Type>().values().to_vec()),
        DataType::Float64 => Ok(array
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .map(|value| *value as f32)
            .collect()),
        DataType::FixedSizeList(_, _) => {
            let list = array.as_fixed_size_list();
            f32_values(list.values().as_ref())
        }
        other => Err(Error::invalid_input(format!(
            "IVF_MRQ expected a float vector, got {other:?}"
        ))),
    }
}

fn fixed_size_list_field(name: &str, item: DataType, width: i32) -> Field {
    Field::new(
        name,
        DataType::FixedSizeList(Arc::new(Field::new("item", item, true)), width),
        true,
    )
}

impl Quantization for MrqQuantizer {
    type BuildParams = MrqBuildParams;
    type Metadata = MrqQuantizationMetadata;
    type Storage = MrqStorage;

    fn build(
        data: &dyn Array,
        _distance_type: DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        let list = data.as_fixed_size_list_opt().ok_or_else(|| {
            Error::invalid_input(format!(
                "IVF_MRQ build expected a fixed size list, got {:?}",
                data.data_type()
            ))
        })?;
        let dim = list.value_length() as usize;
        if dim == 0 {
            return Err(Error::invalid_input(
                "IVF_MRQ requires a non-zero vector dimension".to_string(),
            ));
        }
        let signs = match &params.signs {
            Some(signs) => signs.clone(),
            None => random_fast_rotation_signs(dim),
        };
        // Validate the sign width before the quantizer is stored.
        ResidualEncoder::with_signs(dim, signs.clone())?;
        Ok(Self {
            metadata: MrqQuantizationMetadata {
                dim: dim as u32,
                levels: params.levels,
                signs,
            },
        })
    }

    fn retrain(&mut self, _data: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.dim()
    }

    fn column(&self) -> &'static str {
        MRQ_CODE_COLUMN
    }

    fn use_residual(_: DistanceType) -> bool {
        true
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        let list = vectors.as_fixed_size_list_opt().ok_or_else(|| {
            Error::invalid_input(format!(
                "IVF_MRQ quantize expected a fixed size list, got {:?}",
                vectors.data_type()
            ))
        })?;
        let values = f32_values(list.values().as_ref())?;
        self.quantize_values(&values, list.len())
    }

    fn metadata_key() -> &'static str {
        MRQ_METADATA_KEY
    }

    fn quantization_type() -> QuantizationType {
        QuantizationType::Mrq
    }

    fn metadata(&self, _: Option<QuantizationMetadata>) -> Self::Metadata {
        self.metadata.clone()
    }

    fn from_metadata(metadata: &Self::Metadata, _: DistanceType) -> Result<Quantizer> {
        if !(1..=8).contains(&metadata.levels) {
            return Err(Error::invalid_input(format!(
                "IVF_MRQ levels must be in 1..=8, got {}",
                metadata.levels
            )));
        }
        ResidualEncoder::with_signs(metadata.dim as usize, metadata.signs.clone())?;
        Ok(Quantizer::Mrq(Self {
            metadata: metadata.clone(),
        }))
    }

    fn field(&self) -> Field {
        fixed_size_list_field(MRQ_CODE_COLUMN, DataType::UInt8, self.code_width() as i32)
    }

    fn extra_fields(&self) -> Vec<Field> {
        let levels = i32::from(self.levels());
        vec![
            fixed_size_list_field(MRQ_ALPHA_COLUMN, DataType::Float32, levels),
            Field::new(MRQ_NORM_SQ_COLUMN, DataType::Float32, true),
        ]
    }
}

impl TryFrom<Quantizer> for MrqQuantizer {
    type Error = Error;

    fn try_from(quantizer: Quantizer) -> Result<Self> {
        match quantizer {
            Quantizer::Mrq(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input(
                "Cannot convert non-MrqQuantizer to MrqQuantizer",
            )),
        }
    }
}

impl From<MrqQuantizer> for Quantizer {
    fn from(quantizer: MrqQuantizer) -> Self {
        Self::Mrq(quantizer)
    }
}

/// Writes residual 1-bit columns for vectors that are already IVF residuals.
#[derive(Debug)]
pub struct MrqTransformer {
    quantizer: MrqQuantizer,
    vector_column: String,
}

impl MrqTransformer {
    pub fn new(quantizer: MrqQuantizer, vector_column: impl Into<String>) -> Self {
        Self {
            quantizer,
            vector_column: vector_column.into(),
        }
    }

    fn columns_batch(&self, vectors: &FixedSizeListArray) -> Result<EncodedColumns> {
        let values = f32_values(vectors.values().as_ref())?;
        self.quantizer.encode_rows(&values, vectors.len())
    }
}

impl Transformer for MrqTransformer {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if batch.column_by_name(MRQ_CODE_COLUMN).is_some()
            && batch.column_by_name(MRQ_ALPHA_COLUMN).is_some()
            && batch.column_by_name(MRQ_NORM_SQ_COLUMN).is_some()
        {
            return Ok(batch.clone());
        }
        let vectors = batch
            .column_by_name(&self.vector_column)
            .ok_or_else(|| {
                Error::index(format!(
                    "MRQ transform: column {} not found",
                    self.vector_column
                ))
            })?
            .as_fixed_size_list_opt()
            .ok_or_else(|| {
                Error::index(format!(
                    "MRQ transform: column {} is not a fixed size list",
                    self.vector_column
                ))
            })?;
        let encoded = self.columns_batch(vectors)?;
        let levels = i32::from(self.quantizer.levels());
        let mut batch = batch.clone();
        batch = batch.try_with_column(
            self.quantizer.field(),
            Arc::new(FixedSizeListArray::try_new_from_values(
                UInt8Array::from(encoded.codes),
                self.quantizer.code_width() as i32,
            )?),
        )?;
        batch = batch.try_with_column(
            fixed_size_list_field(MRQ_ALPHA_COLUMN, DataType::Float32, levels),
            Arc::new(FixedSizeListArray::try_new_from_values(
                Float32Array::from(encoded.alpha),
                levels,
            )?),
        )?;
        batch = batch.try_with_column(
            Field::new(MRQ_NORM_SQ_COLUMN, DataType::Float32, true),
            Arc::new(Float32Array::from(encoded.norm_sq)),
        )?;
        Ok(batch)
    }
}

#[derive(Debug, Clone)]
pub struct MrqStorage {
    batch: RecordBatch,
    metadata: MrqQuantizationMetadata,
    row_ids: Vec<u64>,
    /// Row-major, level-major sign bytes.
    codes: Vec<u8>,
    /// FastScan layout, one packed block per level. Empty when the dimension
    /// is not a multiple of 8.
    fast_codes: Vec<u8>,
    alpha: Vec<f32>,
    norm_sq: Vec<f32>,
    dim: usize,
    levels: usize,
    code_bytes: usize,
    distance_type: DistanceType,
}

impl DeepSizeOf for MrqStorage {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.batch.deep_size_of_children(context)
            + self.row_ids.capacity() * std::mem::size_of::<u64>()
            + self.codes.capacity()
            + self.fast_codes.capacity()
            + (self.alpha.capacity() + self.norm_sq.capacity()) * std::mem::size_of::<f32>()
            + self.metadata.signs.capacity()
    }
}

impl MrqStorage {
    fn from_batch(
        batch: RecordBatch,
        metadata: &MrqQuantizationMetadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        let mut batch = batch;
        if let Some(remapper) = frag_reuse_index.as_ref() {
            batch = remapper.remap_row_ids_record_batch(batch, 0)?;
        }
        let dim = metadata.dim as usize;
        let levels = metadata.levels as usize;
        let code_bytes = dim.div_ceil(8);
        let width = levels * code_bytes;
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::index("IVF_MRQ batch is missing _rowid".to_string()))?
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let rows = row_ids.len();
        let codes = fsl_bytes(&batch, MRQ_CODE_COLUMN, width, rows)?;
        let alpha = fsl_f32(&batch, MRQ_ALPHA_COLUMN, levels, rows)?;
        let fast_codes = if scan::supports_fastscan(dim) && rows > 0 {
            scan::pack_level_codes(&codes, rows, levels, code_bytes)?
        } else {
            Vec::new()
        };
        let norm_sq = batch
            .column_by_name(MRQ_NORM_SQ_COLUMN)
            .ok_or_else(|| Error::index("IVF_MRQ batch is missing __mrq_norm_sq".to_string()))?
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        if norm_sq.len() != rows {
            return Err(Error::index(format!(
                "IVF_MRQ __mrq_norm_sq len {} does not match {} rows",
                norm_sq.len(),
                rows
            )));
        }
        Ok(Self {
            batch,
            metadata: metadata.clone(),
            row_ids,
            codes,
            fast_codes,
            alpha,
            norm_sq,
            dim,
            levels,
            code_bytes,
            distance_type,
        })
    }

    fn rotated_query(&self, query: &dyn Array) -> Result<Vec<f32>> {
        let values = f32_values(query)?;
        if values.len() != self.dim {
            return Err(Error::invalid_input(format!(
                "IVF_MRQ query len {}, expected dim {}",
                values.len(),
                self.dim
            )));
        }
        ResidualEncoder::with_signs(self.dim, self.metadata.signs.clone())?.rotate(&values)
    }

    fn row_distance(&self, rotated_query: &[f32], row: usize) -> f32 {
        let query_sq = rotated_query.iter().map(|value| value * value).sum::<f32>();
        let mut score = 0.0f32;
        for level in 0..self.levels {
            let code_start = row * self.levels * self.code_bytes + level * self.code_bytes;
            let packed = &self.codes[code_start..code_start + self.code_bytes];
            let alpha = self.alpha[row * self.levels + level];
            score += alpha * dot_packed_pm1(packed, rotated_query);
        }
        self.norm_sq[row] + query_sq - 2.0 * score
    }

    fn fastscan_distances(&self, lut: &scan::QueryLut, dists: &mut [f32]) {
        let rows = self.row_ids.len();
        let mut ips = vec![0.0f32; rows];
        let mut score = vec![0.0f32; rows];
        for level in 0..self.levels {
            let start = level * rows * self.code_bytes;
            let end = start + rows * self.code_bytes;
            scan::scan_packed_ips(
                &self.fast_codes[start..end],
                rows,
                self.code_bytes,
                lut,
                &mut ips,
            );
            for row in 0..rows {
                let inner = scan::pm1(ips[row], lut.sum_q);
                score[row] += self.alpha[row * self.levels + level] * inner;
            }
        }
        for row in 0..rows {
            dists[row] = self.norm_sq[row] + lut.query_sq - 2.0 * score[row];
        }
    }
}

fn fsl_bytes(batch: &RecordBatch, name: &str, width: usize, rows: usize) -> Result<Vec<u8>> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| Error::index(format!("IVF_MRQ batch is missing column {name}")))?;
    let list = array
        .as_fixed_size_list_opt()
        .ok_or_else(|| Error::index(format!("IVF_MRQ column {name} is not a fixed size list")))?;
    if list.value_length() as usize != width {
        return Err(Error::index(format!(
            "IVF_MRQ column {name} width {}, expected {width}",
            list.value_length()
        )));
    }
    let values = list
        .values()
        .as_primitive_opt::<UInt8Type>()
        .ok_or_else(|| Error::index(format!("IVF_MRQ column {name} values are not uint8")))?;
    if values.len() != rows * width {
        return Err(Error::index(format!(
            "IVF_MRQ column {name} has {} values, expected {}",
            values.len(),
            rows * width
        )));
    }
    Ok(values.values().to_vec())
}

fn fsl_f32(batch: &RecordBatch, name: &str, width: usize, rows: usize) -> Result<Vec<f32>> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| Error::index(format!("IVF_MRQ batch is missing column {name}")))?;
    let list = array
        .as_fixed_size_list_opt()
        .ok_or_else(|| Error::index(format!("IVF_MRQ column {name} is not a fixed size list")))?;
    if list.value_length() as usize != width {
        return Err(Error::index(format!(
            "IVF_MRQ column {name} width {}, expected {width}",
            list.value_length()
        )));
    }
    let values = list
        .values()
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| Error::index(format!("IVF_MRQ column {name} values are not float32")))?;
    if values.len() != rows * width {
        return Err(Error::index(format!(
            "IVF_MRQ column {name} has {} values, expected {}",
            values.len(),
            rows * width
        )));
    }
    Ok(values.values().to_vec())
}

#[async_trait]
impl QuantizerStorage for MrqStorage {
    type Metadata = MrqQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let remapper = frag_reuse_index.map(|index| {
            Arc::new(crate::frag_reuse::FragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>
        });
        Self::from_batch(batch, metadata, distance_type, remapper)
    }

    fn try_from_batch_with_remapper(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        Self::from_batch(batch, metadata, distance_type, frag_reuse_index)
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        reader: &V1FileReader,
        range: Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let schema = reader.schema();
        let batch = reader.read_range(range, schema).await?;
        Self::try_from_batch(batch, metadata, distance_type, frag_reuse_index)
    }
}

pub struct MrqDistCalculator<'a> {
    rotated_query: Vec<f32>,
    lut: scan::QueryLut,
    storage: &'a MrqStorage,
}

impl DistCalculator for MrqDistCalculator<'_> {
    fn distance(&self, id: u32) -> f32 {
        self.storage.row_distance(&self.rotated_query, id as usize)
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        let mut dists = Vec::new();
        self.distance_all_with_scratch(
            _k_hint,
            &mut dists,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        dists
    }

    fn distance_all_with_scratch(
        &self,
        _k_hint: usize,
        dists: &mut Vec<f32>,
        _u16_scratch: &mut Vec<u16>,
        _u8_scratch: &mut Vec<u8>,
        _u32_scratch: &mut Vec<u32>,
    ) {
        let rows = self.storage.row_ids.len();
        dists.clear();
        dists.resize(rows, 0.0);
        if self.storage.fast_codes.is_empty() {
            for (row, dist) in dists.iter_mut().enumerate().take(rows) {
                *dist = self.storage.row_distance(&self.rotated_query, row);
            }
            return;
        }
        self.storage.fastscan_distances(&self.lut, dists);
    }

    fn accumulate_topk_with_scratch(
        &self,
        k: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        row_id: impl Fn(u32) -> u64,
        res: &mut BinaryHeap<OrderedNode<u64>>,
        dists: &mut Vec<f32>,
        u16_scratch: &mut Vec<u16>,
        u8_scratch: &mut Vec<u8>,
        u32_scratch: &mut Vec<u32>,
    ) {
        if k == 0 {
            return;
        }
        // A per-batch L1 early exit was slower than one FastScan pass per level:
        // the ||q||_1 cap rarely rejects a whole 32-row block, and the extra
        // kernel calls dominated. `search_partition` still uses that cap for
        // the scalar scan, where skipping a row skips real work.
        self.distance_all_with_scratch(k, dists, u16_scratch, u8_scratch, u32_scratch);
        // u16 FastScan is a bound, not the rank. Rows that can still enter the
        // top-k are replaced by the scalar estimate; the rest cannot.
        if lower_bound.is_some() || upper_bound.is_some() || k >= dists.len() {
            for (row, dist) in dists.iter_mut().enumerate() {
                *dist = self.storage.row_distance(&self.rotated_query, row);
            }
        } else {
            self.refine_topk_band(k, dists);
        }
        accumulate_distances_into_heap(k, lower_bound, upper_bound, row_id, res, dists);
    }
}

impl MrqDistCalculator<'_> {
    fn refine_topk_band(&self, k: usize, dists: &mut [f32]) {
        let err = self.lut.pm1_error();
        if err == 0.0 || dists.is_empty() || k == 0 {
            return;
        }
        let rows = dists.len();
        let levels = self.storage.levels;
        let mut bounds = vec![0.0f32; rows];
        for (row, bound) in bounds.iter_mut().enumerate() {
            let base = row * levels;
            let mut scale = 0.0f32;
            for level in 0..levels {
                scale += self.storage.alpha[base + level].abs();
            }
            *bound = 2.0 * scale * err;
        }
        let mut sample = dists.to_vec();
        sample.select_nth_unstable_by(k - 1, |left, right| left.total_cmp(right));
        let kth = sample[k - 1];
        // The current top-k are witnesses: each exact distance is at most
        // `approx + bound`. An outlier bound outside that set must not widen it.
        let mut witness_bound = 0.0f32;
        for (dist, bound) in dists.iter().zip(bounds.iter()) {
            if *dist <= kth {
                witness_bound = witness_bound.max(*bound);
            }
        }
        for (row, dist) in dists.iter_mut().enumerate() {
            if *dist <= kth + witness_bound + bounds[row] {
                *dist = self.storage.row_distance(&self.rotated_query, row);
            } else {
                // `accumulate_distances_into_heap` drops distances at the
                // default upper bound, so these rows stay out of the heap.
                *dist = f32::MAX;
            }
        }
    }
}

impl VectorStore for MrqStorage {
    type DistanceCalculator<'a> = MrqDistCalculator<'a>;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send> {
        let names = [
            ROW_ID,
            MRQ_CODE_COLUMN,
            MRQ_ALPHA_COLUMN,
            MRQ_NORM_SQ_COLUMN,
        ];
        let indices = names
            .iter()
            .map(|name| {
                self.batch.schema().index_of(name).map_err(|_| {
                    Error::index(format!("IVF_MRQ storage batch is missing column {name}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let projected = self.batch.project(&indices)?;
        Ok(std::iter::once(projected))
    }

    fn len(&self) -> usize {
        self.row_ids.len()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids[id as usize]
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.iter()
    }

    fn append_batch(&self, batch: RecordBatch, vector_column: &str) -> Result<Self> {
        let quantizer = MrqQuantizer {
            metadata: self.metadata.clone(),
        };
        let transformed = if batch.column_by_name(MRQ_CODE_COLUMN).is_some() {
            batch
        } else {
            MrqTransformer::new(quantizer, vector_column).transform(&batch)?
        };
        let schema = if self.batch.num_columns() == 0 {
            transformed.schema()
        } else {
            self.batch.schema()
        };
        let combined = concat_batches(&schema, [&self.batch, &transformed])?;
        Self::from_batch(combined, &self.metadata, self.distance_type, None)
    }

    fn dist_calculator(&self, query: ArrayRef, _dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        let rotated_query = self
            .rotated_query(query.as_ref())
            .expect("IVF_MRQ query does not match the index dimension");
        MrqDistCalculator {
            lut: scan::QueryLut::build(&rotated_query),
            rotated_query,
            storage: self,
        }
    }

    fn dist_calculator_with_scratch<'a>(
        &'a self,
        query: ArrayRef,
        dist_q_c: f32,
        _residual: Option<QueryResidual<'a>>,
        _f32_scratch: &'a mut Vec<f32>,
        _options: DistanceCalculatorOptions,
    ) -> Self::DistanceCalculator<'a> {
        self.dist_calculator(query, dist_q_c)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        let row = id as usize;
        let mut rotated = vec![0.0f32; self.dim];
        for level in 0..self.levels {
            let code_start = row * self.levels * self.code_bytes + level * self.code_bytes;
            let packed = &self.codes[code_start..code_start + self.code_bytes];
            let alpha = self.alpha[row * self.levels + level];
            for dim in 0..self.dim {
                let bit = packed[dim / 8] >> (dim % 8) & 1;
                let sign = if bit == 1 { 1.0 } else { -1.0 };
                rotated[dim] += alpha * sign;
            }
        }
        MrqDistCalculator {
            lut: scan::QueryLut::build(&rotated),
            rotated_query: rotated,
            storage: self,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Instant;

    use arrow_array::UInt64Array;
    use arrow_schema::{Field, Schema};
    use lance_linalg::distance::DistanceType;

    use super::*;

    fn storage_for(dim: usize, rows: usize, levels: u8) -> MrqStorage {
        let values: Vec<f32> = (0..rows * dim)
            .map(|idx| ((idx * 17) % 100) as f32 / 10.0 - 5.0)
            .collect();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim as i32)
                .unwrap();
        let params = MrqBuildParams::new(levels).unwrap();
        let quantizer =
            <MrqQuantizer as Quantization>::build(&vectors, DistanceType::L2, &params).unwrap();
        let metadata = quantizer.metadata_ref().clone();
        let row_ids = Arc::new(UInt64Array::from_iter_values(0..rows as u64)) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(schema, vec![row_ids, Arc::new(vectors)]).unwrap();
        let transformed = MrqTransformer::new(quantizer, "vector")
            .transform(&batch)
            .unwrap();
        MrqStorage::from_batch(transformed, &metadata, DistanceType::L2, None).unwrap()
    }

    fn query_list(dim: usize, seed: usize) -> ArrayRef {
        let values: Vec<f32> = (0..dim)
            .map(|idx| ((idx + seed) * 13 % 50) as f32 / 8.0 - 2.0)
            .collect();
        Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim as i32)
                .unwrap(),
        )
    }

    #[test]
    fn stored_row_drops_bound_columns() {
        assert_eq!(logical_row_bytes(128, 4), 124 - 32);
        assert_eq!(logical_row_bytes(128, 8), 236 - 64);
        assert_eq!(logical_row_bytes(768, 8), 876 - 64);
        let storage = storage_for(32, 4, 4);
        let names: Vec<&str> = storage
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert!(names.contains(&MRQ_CODE_COLUMN));
        assert!(names.contains(&MRQ_ALPHA_COLUMN));
        assert!(names.contains(&MRQ_NORM_SQ_COLUMN));
        assert!(!names.iter().any(|name| name.contains("radius")));
        assert!(!names.iter().any(|name| name.contains("bias")));
        assert_eq!(
            storage
                .schema()
                .field_with_name(MRQ_CODE_COLUMN)
                .unwrap()
                .name(),
            MRQ_CODE_COLUMN
        );
    }

    #[test]
    fn fastscan_distances_stay_within_quantization_error() {
        let storage = storage_for(32, 100, 4);
        let calc = storage.dist_calculator(query_list(32, 3), 0.0);
        let fast = calc.distance_all(10);
        assert!(!storage.fast_codes.is_empty());
        for (row, &fast_dist) in fast.iter().enumerate() {
            let exact = storage.row_distance(&calc.rotated_query, row);
            let scale = (0..storage.levels)
                .map(|level| storage.alpha[row * storage.levels + level].abs())
                .sum::<f32>();
            let limit = 2.0 * scale * calc.lut.pm1_error() + 1.0e-3;
            let gap = (fast_dist - exact).abs();
            assert!(
                gap <= limit,
                "row {row} gap {gap} limit {limit} fast {fast_dist} exact {exact}"
            );
        }
    }

    #[test]
    fn fastscan_topk_matches_exact_estimator() {
        let storage = storage_for(32, 256, 8);
        let calc = storage.dist_calculator(query_list(32, 9), 0.0);
        let k = 10usize;
        let exact: Vec<f32> = (0..storage.len())
            .map(|row| storage.row_distance(&calc.rotated_query, row))
            .collect();
        let mut order: Vec<usize> = (0..exact.len()).collect();
        order.sort_by(|&left, &right| {
            exact[left]
                .total_cmp(&exact[right])
                .then_with(|| left.cmp(&right))
        });
        let kth = exact[order[k - 1]];
        let mut heap = BinaryHeap::new();
        calc.accumulate_topk_with_scratch(
            k,
            None,
            None,
            |id| id as u64,
            &mut heap,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        let hits: HashSet<u64> = heap.iter().map(|node| node.id).collect();
        assert_eq!(hits.len(), k);
        for id in order.into_iter().take(k) {
            if exact[id] < kth - 1.0e-4 {
                assert!(hits.contains(&(id as u64)), "dropped row {id}");
            }
        }
    }

    #[test]
    fn fastscan_eval_records_space_and_speed() {
        let dim = 128usize;
        let rows = 4096usize;
        let levels = 8u8;
        let storage = storage_for(dim, rows, levels);
        let bytes = logical_row_bytes(dim, levels as usize);
        let old_bytes = 8 + levels as usize * dim.div_ceil(8) + 12 * levels as usize + 4;
        let queries = 8usize;
        let mut scalar_ns = 0u128;
        let mut fast_ns = 0u128;
        let mut topk_ns = 0u128;
        for seed in 0..queries {
            let query = query_list(dim, seed);
            let calc = storage.dist_calculator(query, 0.0);
            let started = Instant::now();
            let mut checksum = 0.0f32;
            for row in 0..rows {
                checksum += storage.row_distance(&calc.rotated_query, row);
            }
            scalar_ns += started.elapsed().as_nanos();
            assert!(checksum.is_finite());
            let started = Instant::now();
            let dists = calc.distance_all(10);
            fast_ns += started.elapsed().as_nanos();
            assert_eq!(dists.len(), rows);
            let started = Instant::now();
            let mut heap = BinaryHeap::new();
            calc.accumulate_topk_with_scratch(
                10,
                None,
                None,
                |id| storage.row_id(id),
                &mut heap,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
            );
            topk_ns += started.elapsed().as_nanos();
            assert_eq!(heap.len(), 10);
        }
        let scalar_ms = scalar_ns as f64 / 1.0e6;
        let fast_ms = fast_ns as f64 / 1.0e6;
        let topk_ms = topk_ns as f64 / 1.0e6;
        let report = format!(
            "dim={dim} rows={rows} levels={levels} queries={queries}\n\
             logical_bytes_per_row={bytes} previous_with_radius_bias={old_bytes} saved={}\n\
             scalar_ms={scalar_ms:.3} fastscan_ms={fast_ms:.3} topk_ms={topk_ms:.3}\n",
            old_bytes - bytes
        );
        std::fs::create_dir_all("/opt/cursor/artifacts").ok();
        std::fs::write("/opt/cursor/artifacts/mrq_eval.log", &report).ok();
        assert!(bytes < old_bytes);
        assert!(
            fast_ms < scalar_ms,
            "fastscan {fast_ms} was not faster than scalar {scalar_ms}"
        );
    }
}
