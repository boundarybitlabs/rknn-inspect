use std::io::{BufReader, Cursor};

use {
    rknpu2::query::TensorAttrView,
    stanza::{renderer::Renderer, table::Table},
};

use {
    crate::perf::parsing::parse_perf_data,
    rknpu2::{
        RKNN,
        api::runtime::RuntimeAPI,
        bf16, f16,
        query::{InputAttr, InputOutputNum, PerfDetail},
        tensor::{TensorT, TensorType, builder::TensorBuilder, tensor::Tensor},
    },
};

mod parsing;

pub fn do_perf(
    rknn_model: &RKNN<RuntimeAPI>,
    core_mask: u32,
    console: &dyn Renderer<Output = String>,
    full_name: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    rknn_model.set_core_mask(core_mask)?;

    let io_num = rknn_model.query::<InputOutputNum>()?;

    let mut input_tensors = Vec::<TensorT>::new();

    for i in 0..io_num.input_num() {
        let attr = rknn_model.query_with_input::<InputAttr>(i)?;
        match attr.dtype() {
            rknpu2::tensor::DataTypeKind::Float32(_) => {
                input_tensors.push(build_tensor::<f32>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::Float16(_) => {
                input_tensors.push(build_tensor::<f16>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::BFloat16(_) => {
                input_tensors.push(build_tensor::<bf16>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::Int4(_) => todo!(),
            rknpu2::tensor::DataTypeKind::Int8(_) => {
                input_tensors.push(build_tensor::<i8>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::UInt8(_) => {
                input_tensors.push(build_tensor::<u8>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::Int16(_) => {
                input_tensors.push(build_tensor::<i16>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::UInt16(_) => {
                input_tensors.push(build_tensor::<u16>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::Int32(_) => {
                input_tensors.push(build_tensor::<i32>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::UInt32(_) => {
                input_tensors.push(build_tensor::<u32>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::Int64(_) => {
                input_tensors.push(build_tensor::<i64>(rknn_model, i)?.into())
            }
            rknpu2::tensor::DataTypeKind::Bool(_) => todo!(),
            rknpu2::tensor::DataTypeKind::Max(_) => todo!(),
            rknpu2::tensor::DataTypeKind::Other(_) => todo!(),
        }
    }

    rknn_model.set_inputs(input_tensors)?;
    rknn_model.run()?;

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

fn build_tensor<T: TensorType + Copy>(
    rknn_model: &RKNN<RuntimeAPI>,
    index: u32,
) -> Result<Tensor<T>, Box<dyn std::error::Error>> {
    let mut tensor = TensorBuilder::new_input(rknn_model, index).allocate::<T>()?;
    tensor.fill_with(T::default());

    Ok(tensor)
}
