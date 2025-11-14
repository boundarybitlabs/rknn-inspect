use {
    crate::perf::parsing::parse_perf_data,
    rknpu2::{
        RKNN,
        api::runtime::RuntimeAPI,
        bf16, f16,
        io::{buffer::BufView, input::Input},
        query::{InputOutputNum, PerfDetail, QueryWithInput, TensorAttrView},
        tensor::{DataTypeKind, TensorFormatKind},
    },
    stanza::{renderer::Renderer, table::Table},
    std::io::{BufReader, Cursor},
};

mod parsing;

pub fn do_perf<Q: QueryWithInput<Input = u32> + TensorAttrView>(
    rknn_model: &RKNN<RuntimeAPI>,
    core_mask: u32,
    console: &dyn Renderer<Output = String>,
    full_name: bool,
    passthrough: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    rknn_model.set_core_mask(core_mask)?;

    run_tensors::<Q>(rknn_model, passthrough)?;

    let perf_info = rknn_model.query::<PerfDetail>()?;
    let details = Cursor::new(perf_info.details().as_bytes());
    let reader = BufReader::with_capacity(1024, details);

    let (v, summary) = parse_perf_data(reader);

    let mut table = Table::default();
    if full_name {
        table.push_row(vec![
            "ID".to_string(),
            "Op Type".to_string(),
            "Target".to_string(),
            "Data Type".to_string(),
            "Input Shape".to_string(),
            "Output Shape".to_string(),
            "Cycles(DDR/NPU/Total)".to_string(),
            "Time(us)".to_string(),
            "WorkLoad(0/1/2)".to_string(),
            "RW(KB)".to_string(),
            "MacUsage(%)".to_string(),
            "FullName".to_string(),
        ]);
    } else {
        table.push_row(vec![
            "ID".to_string(),
            "Op Type".to_string(),
            "Target".to_string(),
            "Data Type".to_string(),
            "Input Shape".to_string(),
            "Output Shape".to_string(),
            "Cycles(DDR/NPU/Total)".to_string(),
            "Time(us)".to_string(),
            "WorkLoad(0/1/2)".to_string(),
            "RW(KB)".to_string(),
            "MacUsage(%)".to_string(),
        ]);
    }

    for item in v {
        if full_name {
            table.push_row(vec![
                item.id.to_string(),
                item.op_type,
                item.target,
                item.data_type,
                item.input_shape,
                item.output_shape,
                item.cycles,
                item.time.to_string(),
                item.work_load,
                item.rw,
                item.mac_usage,
                item.full_name,
            ])
        } else {
            table.push_row(vec![
                item.id.to_string(),
                item.op_type,
                item.target,
                item.data_type,
                item.input_shape,
                item.output_shape,
                item.cycles,
                item.time.to_string(),
                item.work_load,
                item.rw,
                item.mac_usage,
            ])
        }
    }

    println!("{}", console.render(&table));

    if let Some(summary) = summary {
        let mut table = Table::default();

        table.push_row(vec![
            "OpType".to_string(),
            "Calls".to_string(),
            "CPUTime(us)".to_string(),
            "GPUTime(us)".to_string(),
            "NPUTime(us)".to_string(),
            "TotalTime(us)".to_string(),
            "TimeRatio(%)".to_string(),
        ]);

        for item in &summary.op_time_ranking {
            table.push_row(vec![
                item.op_type.clone(),
                item.call_count.to_string(),
                item.cpu_time_us.to_string(),
                item.gpu_time_us.to_string(),
                item.npu_time_us.to_string(),
                item.total_time_us.to_string(),
                format!("{:.2}", item.time_ratio_pct),
            ]);
        }

        // Add totals row
        table.push_row(vec![
            "Total".to_string(),
            "".to_string(),
            summary.total_cpu_time_us.to_string(),
            summary.total_gpu_time_us.to_string(),
            summary.total_npu_time_us.to_string(),
            summary.total_all_time_us.to_string(),
            "".to_string(),
        ]);

        println!("{}", console.render(&table));

        // Optional: print the two top-level stats outside the table
        println!(
            "\nTotal Operator Elapsed Per Frame Time (us): {}",
            summary.total_operator_time_us
        );
        println!(
            "Total Memory Read/Write Per Frame Size (KB): {:.2}",
            summary.total_memory_rw_kb
        );
    }

    Ok(())
}

#[derive(Debug)]
enum OwnedBuf {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
}

fn run_tensors<Q: QueryWithInput<Input = u32> + TensorAttrView>(
    rknn_model: &RKNN<RuntimeAPI>,
    is_passthrough: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let io_num = rknn_model.query::<InputOutputNum>()?;

    // Hold buffers so they live long enough
    let mut bufs: Vec<(OwnedBuf, TensorFormatKind)> =
        Vec::with_capacity(io_num.input_num() as usize);
    let mut inputs: Vec<Input> = Vec::with_capacity(io_num.input_num() as usize);

    // 1) Allocate typed buffers based on DataTypeKind
    for i in 0..io_num.input_num() {
        let attr = rknn_model.query_with_input::<Q>(i)?;

        let elem_count = attr.num_elements() as usize;
        let format = attr.format();
        let dtype = attr.dtype();

        let owned = match dtype {
            DataTypeKind::Float32(_) => OwnedBuf::F32(vec![0.0f32; elem_count]),
            DataTypeKind::Float16(_) => OwnedBuf::F16(vec![f16::ZERO; elem_count]),
            DataTypeKind::BFloat16(_) => OwnedBuf::BF16(vec![bf16::ZERO; elem_count]),

            DataTypeKind::Int8(_) => OwnedBuf::I8(vec![0i8; elem_count]),
            DataTypeKind::UInt8(_) => OwnedBuf::U8(vec![0u8; elem_count]),
            DataTypeKind::UInt16(_) => OwnedBuf::U16(vec![0u16; elem_count]),
            DataTypeKind::Int32(_) => OwnedBuf::I32(vec![0i32; elem_count]),
            DataTypeKind::UInt32(_) => OwnedBuf::U32(vec![0u32; elem_count]),
            DataTypeKind::Int64(_) => OwnedBuf::I64(vec![0i64; elem_count]),

            _ => {
                return Err(format!("Unsupported data type in run_tensors: {:?}", dtype))?;
            }
        };

        bufs.push((owned, format));
    }

    // 2) Build Input objects with BufView over the owned buffers
    for (i, (buf, format)) in bufs.iter().enumerate() {
        let input = match buf {
            OwnedBuf::F32(v) => Input::new(
                i as u32,
                BufView::F32(v.as_slice()),
                is_passthrough,
                *format,
            ),
            OwnedBuf::F16(v) => Input::new(
                i as u32,
                BufView::F16(v.as_slice()),
                is_passthrough,
                *format,
            ),
            OwnedBuf::BF16(v) => Input::new(
                i as u32,
                BufView::BF16(v.as_slice()),
                is_passthrough,
                *format,
            ),
            OwnedBuf::I8(v) => {
                Input::new(i as u32, BufView::I8(v.as_slice()), is_passthrough, *format)
            }
            OwnedBuf::U8(v) => {
                Input::new(i as u32, BufView::U8(v.as_slice()), is_passthrough, *format)
            }
            OwnedBuf::U16(v) => Input::new(
                i as u32,
                BufView::U16(v.as_slice()),
                is_passthrough,
                *format,
            ),
            OwnedBuf::I32(v) => Input::new(
                i as u32,
                BufView::I32(v.as_slice()),
                is_passthrough,
                *format,
            ),
            OwnedBuf::U32(v) => Input::new(
                i as u32,
                BufView::U32(v.as_slice()),
                is_passthrough,
                *format,
            ),
            OwnedBuf::I64(v) => Input::new(
                i as u32,
                BufView::I64(v.as_slice()),
                is_passthrough,
                *format,
            ),
        };

        inputs.push(input);
    }

    // 3) Run
    rknn_model.set_inputs(inputs)?;
    rknn_model.run()?;

    Ok(())
}
