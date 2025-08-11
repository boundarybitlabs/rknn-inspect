use {
    rknpu2::{
        query::{Io, TensorAttrView},
        tensor::{DataTypeKind, QuantTypeKind, TensorFormatKind},
    },
    stanza::table::Table,
    std::fmt::Write,
};

pub fn push_attr_row<T>(tbl: &mut Table, a: &T, full: bool)
where
    T: TensorAttrView,
{
    let io = format!("{:?}", a.io());
    let name = a.name();
    let dtype = a.dtype();
    let fmt = a.format();

    let dtype_str = format!("{dtype:?}");
    let (bit, signed) = bit_and_signed(dtype);
    let shape_str = fmt_dims(a.dims());
    let fmt_str = format!("{fmt:?}");

    let qnt = a.qnt_type();
    let quant_str = format!("{qnt:?}");
    let (scale_use, zp_use, fl_use) = match qnt {
        QuantTypeKind::AffineAsymmetric(_) => (Some(a.scale()), Some(a.zero_point()), None),
        QuantTypeKind::Dfp(_) => (None, None, Some(a.fl())),
        _ => (None, None, None),
    };

    let (qmin, qmax) = qmin_qmax(bit, signed);
    let float_range = match (qnt, scale_use, zp_use, fl_use) {
        (QuantTypeKind::AffineAsymmetric(_), Some(s), Some(zp), _) => {
            Some(float_range_str_affine(s, zp, qmin, qmax))
        }
        (QuantTypeKind::Dfp(_), _, _, Some(fl)) => Some(float_range_str_dfp(fl, qmin, qmax)),
        _ => None,
    };

    let size = a.size() as u64;
    let size_ws = a.size_with_stride() as u64;
    let contig = size == size_ws;

    let es = elem_size_bytes(dtype);
    let strides = per_dim_strides(
        fmt,
        a.dims(),
        es,
        a.size_with_stride(),
        a.w_stride(),
        a.h_stride(),
    );

    let qrange = format!("[{},{}]", qmin, qmax);
    let scale_str = scale_use
        .map(|x| format!("{:.9}", x))
        .unwrap_or_else(|| "-".into());
    let zp_str = zp_use.map(|x| x.to_string()).unwrap_or_else(|| "-".into());
    let fl_str = fl_use.map(|x| x.to_string()).unwrap_or_else(|| "-".into());
    let notes = infer_notes(fmt, a.io(), contig);

    if full {
        tbl.push_row(vec![
            io,
            name,
            dtype_str,
            bit.to_string(),
            if signed { "Y".into() } else { "N".into() },
            shape_str,
            fmt_str,
            quant_str,
            scale_str,
            zp_str,
            fl_str,
            qmin.to_string(),
            qmax.to_string(),
            float_range.unwrap_or_else(|| "-".into()),
            size.to_string(),
            size_ws.to_string(),
            if contig { "Y".into() } else { "N".into() },
            strides,
            notes,
        ]);
    } else {
        tbl.push_row(vec![
            io,
            name,
            dtype_str,
            shape_str,
            fmt_str,
            quant_str,
            scale_str,
            zp_str,
            qrange,
            size.to_string(),
            strides,
            notes,
        ]);
    }
}

fn fmt_dims(dims: &[u32]) -> String {
    let mut s = String::with_capacity(32);
    s.push('[');
    for (i, d) in dims.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{d}");
    }
    s.push(']');
    s
}

fn bit_and_signed(dtype: DataTypeKind) -> (u32, bool) {
    match dtype {
        DataTypeKind::Int4(_) => (4, true),
        DataTypeKind::Int8(_) => (8, true),
        DataTypeKind::UInt8(_) => (8, false),
        DataTypeKind::Int16(_) => (16, true),
        DataTypeKind::UInt16(_) => (16, false),
        DataTypeKind::Float16(_) => (16, true),
        DataTypeKind::BFloat16(_) => (16, true),
        DataTypeKind::Int32(_) => (32, true),
        DataTypeKind::UInt32(_) => (32, false),
        DataTypeKind::Float32(_) => (32, true),
        DataTypeKind::Int64(_) => (64, true),
        DataTypeKind::Bool(_) => (1, false),
        _ => (8, true),
    }
}

fn elem_size_bytes(dtype: DataTypeKind) -> u32 {
    match dtype {
        DataTypeKind::Int4(_) => 0, // packed; handle specially if needed
        DataTypeKind::Int8(_) | DataTypeKind::UInt8(_) | DataTypeKind::Bool(_) => 1,
        DataTypeKind::Int16(_)
        | DataTypeKind::UInt16(_)
        | DataTypeKind::Float16(_)
        | DataTypeKind::BFloat16(_) => 2,
        DataTypeKind::Int32(_) | DataTypeKind::UInt32(_) | DataTypeKind::Float32(_) => 4,
        DataTypeKind::Int64(_) => 8,
        _ => 1,
    }
}

fn qmin_qmax(bit: u32, signed: bool) -> (i32, i32) {
    match (bit, signed) {
        (1, false) => (0, 1),
        (4, true) => (-8, 7),
        (8, true) => (-128, 127),
        (8, false) => (0, 255),
        (16, true) => (-32768, 32767),
        (16, false) => (0, 65535),
        (32, true) => (i32::MIN, i32::MAX),
        (32, false) => (0, i32::MAX),
        (64, true) => (i32::MIN, i32::MAX),
        _ => (-128, 127),
    }
}

fn float_range_str_affine(scale: f32, zp: i32, qmin: i32, qmax: i32) -> String {
    let fmin = (qmin - zp) as f32 * scale;
    let fmax = (qmax - zp) as f32 * scale;
    format!("[{:.3},{:.3}]", fmin, fmax)
}

fn float_range_str_dfp(fl: i8, qmin: i32, qmax: i32) -> String {
    let s = 2f32.powi(-(fl as i32));
    let fmin = (qmin as f32) * s;
    let fmax = (qmax as f32) * s;
    format!("[{:.3},{:.3}]", fmin, fmax)
}

/// Compute labeled byte strides from RKNN pixel strides, with derivations when fields are 0.
/// - fmt: TensorFormatKind
/// - dims: logical dims as reported by RKNN
/// - elem_size: bytes per element (from your DataTypeKind)
/// - size_with_stride: total bytes spanned per batch (safe alloc span)
/// - w_stride_px: pixels per row incl. padding (0 => derive from W)
/// - h_stride_px: rows per image incl. padding (0 => derive from H)
pub fn per_dim_strides(
    fmt: TensorFormatKind,
    dims: &[u32],
    elem_size: u32,
    size_with_stride: u32,
    w_stride_px: u32,
    h_stride_px: u32,
) -> String {
    let es = elem_size as u64;
    let ns = size_with_stride as u64;

    // 0/1-D/2-D tensors: just show the batch stride (buffer span).
    if dims.len() <= 2 {
        return format!("N={ns}");
    }

    match fmt {
        // ---------------- NHWC ----------------
        // dims: [N,H,W,C]
        TensorFormatKind::NHWC(_) => {
            let h = *dims.get(1).unwrap_or(&1) as u64;
            let w = *dims.get(2).unwrap_or(&1) as u64;
            let c = *dims.get(3).unwrap_or(&1) as u64;

            let row_px = if w_stride_px == 0 {
                w
            } else {
                w_stride_px as u64
            };
            let rows_px = if h_stride_px == 0 {
                h
            } else {
                h_stride_px as u64
            };

            let stride_c = es; // next channel in same pixel
            let stride_w = c * es; // next pixel in same row
            let stride_h = row_px * c * es; // next row
            let stride_n = rows_px * row_px * c * es; // next batch

            return format!("N={stride_n}, H={stride_h}, W={stride_w}, C={stride_c}");
        }

        // ---------------- NCHW ----------------
        // dims: [N,C,H,W]
        TensorFormatKind::NCHW(_) => {
            let c = *dims.get(1).unwrap_or(&1) as u64;
            let h = *dims.get(2).unwrap_or(&1) as u64;
            let w = *dims.get(3).unwrap_or(&1) as u64;

            // In NCHW, a "row" is W elements in one channel plane.
            let row_px = if w_stride_px == 0 {
                w
            } else {
                w_stride_px as u64
            };
            let rows_px = if h_stride_px == 0 {
                h
            } else {
                h_stride_px as u64
            };

            let row_bytes = row_px * es; // bytes to jump by 1 in H
            let plane_bytes = rows_px * row_bytes; // bytes to jump by 1 in C

            let stride_w = es; // next element in row
            let stride_h = row_bytes; // next row
            let stride_c = plane_bytes; // next channel plane
            let stride_n = c * plane_bytes; // next batch

            return format!("N={stride_n}, H={stride_h}, W={stride_w}, C={stride_c}");
        }

        // ---------------- NC1HWC2 ----------------
        // Native packed order: [N, C1, H, W, C2]
        // RKNN often reports dims with 5 axes (C2 present) for native views.
        // If not, we infer C2 (common defaults: int8->2, fp16/bf16->8).
        TensorFormatKind::NC1HWC2(_) => {
            // Try to read C2 directly from dims[4] when present:
            let c2_from_shape = dims.get(4).copied().unwrap_or(0) as u64;
            let c2 = if c2_from_shape > 0 {
                c2_from_shape
            } else {
                // Heuristic fallback by dtype width:
                match elem_size {
                    1 => 2, // int8 native lanes commonly 2
                    2 => 8, // fp16/bf16 lanes commonly 8
                    _ => 2,
                }
            };

            // Logical H/W (for deriving when pixel strides are 0)
            // Many NC1HWC2 views still expose logical H/W at dims[2]/[3]
            let h = *dims.get(2).unwrap_or(&1) as u64;
            let w = *dims.get(3).unwrap_or(&1) as u64;

            // Pixel strides → bytes. For packed layout, one "pixel" on a row carries C2 lanes.
            let row_px = if w_stride_px == 0 {
                w
            } else {
                w_stride_px as u64
            };
            let rows_px = if h_stride_px == 0 {
                h
            } else {
                h_stride_px as u64
            };

            let stride_c2 = es; // next lane inside the vector
            let stride_w = c2 * es; // next logical pixel in row
            let row_bytes = row_px * c2 * es; // next row within same C1
            let stride_h = row_bytes; // next row
            let stride_c1 = rows_px * row_bytes; // next C1 slice (block of H*W rows)
            let stride_n = ns; // next batch (RKNN gives us the full span)

            return format!(
                "N={stride_n}, C1={stride_c1}, H={stride_h}, W={stride_w}, C2={stride_c2}"
            );
        }

        // ---------------- Other / Undefined ----------------
        _ => {
            // Best-effort: show raw pixel info and the total batch span.
            let row_px = if w_stride_px == 0 {
                // try to derive a sensible default row length if W is known
                (if dims.len() >= 3 {
                    dims[dims_index::w(fmt, dims)]
                } else {
                    0
                }) as u64
            } else {
                w_stride_px as u64
            };
            let rows_px = if h_stride_px == 0 {
                (if dims.len() >= 3 {
                    dims[dims_index::h(fmt, dims)]
                } else {
                    0
                }) as u64
            } else {
                h_stride_px as u64
            };
            let row_bytes = row_px * es;
            return format!("N={ns}, row_px={row_px} (≈{row_bytes}B), rows_px={rows_px}");
        }
    }
}

/// Tiny helper for indexing H/W when format is unknown.
/// Returns an index into `dims` for H/W based on a best-effort guess.
mod dims_index {
    use super::TensorFormatKind;
    pub fn h(fmt: TensorFormatKind, dims: &[u32]) -> usize {
        match fmt {
            TensorFormatKind::NHWC(_) => 1,
            TensorFormatKind::NCHW(_) => 2,
            TensorFormatKind::NC1HWC2(_) => 2,
            _ => {
                if dims.len() > 2 {
                    2
                } else {
                    0
                }
            }
        }
    }
    pub fn w(fmt: TensorFormatKind, dims: &[u32]) -> usize {
        match fmt {
            TensorFormatKind::NHWC(_) => 2,
            TensorFormatKind::NCHW(_) => 3,
            TensorFormatKind::NC1HWC2(_) => 3,
            _ => {
                if dims.len() > 3 {
                    3
                } else {
                    0
                }
            }
        }
    }
}

fn infer_notes(fmt: TensorFormatKind, io: Io, contig: bool) -> String {
    let mut n = String::new();
    if !contig {
        n.push_str("padded");
    }
    if matches!(fmt, TensorFormatKind::NC1HWC2(_)) {
        if !n.is_empty() {
            n.push_str(" | ");
        }
        n.push_str("NC1HWC2");
    }
    if matches!(io, Io::Output) {
        if !n.is_empty() {
            n.push_str(" | ");
        }
        n.push_str("postproc: dequant");
    }
    n
}
