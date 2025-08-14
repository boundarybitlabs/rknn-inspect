use {
    crate::perf::parsing::parse_perf_data,
    rknpu2::{
        RKNN,
        api::runtime::RuntimeAPI,
        io::{buffer::BufView, input::Input},
        query::{InputAttr, InputOutputNum, PerfDetail, TensorAttrView},
    },
    stanza::{renderer::Renderer, table::Table},
    std::io::{BufReader, Cursor},
};

mod parsing;

pub fn do_perf(
    rknn_model: &RKNN<RuntimeAPI>,
    core_mask: u32,
    console: &dyn Renderer<Output = String>,
    full_name: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    rknn_model.set_core_mask(core_mask)?;

    run_tensors(rknn_model)?;

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

fn run_tensors(rknn_model: &RKNN<RuntimeAPI>) -> Result<(), Box<dyn std::error::Error>> {
    let io_num = rknn_model.query::<InputOutputNum>()?;

    // Hold buffers separately so they live long enough
    let mut bufs = Vec::with_capacity(io_num.input_num() as usize);
    let mut inputs: Vec<Input> = Vec::with_capacity(io_num.input_num() as usize);

    for i in 0..io_num.input_num() {
        let attr = rknn_model.query_with_input::<InputAttr>(i)?;

        bufs.push((vec![0.01f32; attr.num_elements() as usize], attr.format()));
    }

    for (i, slice) in bufs.iter().enumerate() {
        inputs.push(Input::new(
            i as u32,
            BufView::F32(slice.0.as_slice()),
            false,
            slice.1,
        ));
    }

    rknn_model.set_inputs(inputs)?;
    rknn_model.run()?;

    Ok(())
}
