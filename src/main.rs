use {
    clap::Parser,
    prettytable::Table,
    rknpu2::{
        RKNN,
        query::{InputAttr, InputOutputNum, OutputAttr, PerfDetail, SdkVersion},
        rknpu2_sys::RKNN_FLAG_COLLECT_PERF_MASK,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut bytes = std::fs::read(&args.model_path)?;

    let rknn_model = match RKNN::new_with_library(
        "/usr/lib/librknnrt.so",
        &mut bytes,
        RKNN_FLAG_COLLECT_PERF_MASK,
    ) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("Failed to load model: {}", err);
            std::process::exit(1);
        }
    };

    let mut table_full = Table::new();

    if args.sdk {
        let mut table = Table::new();

        let sdk = rknn_model.query::<SdkVersion>()?;
        table.add_row(vec!["SDK".into(), sdk.api_version()].into());
        table.add_row(vec!["Driver".into(), sdk.driver_version()].into());

        table_full.add_row(vec![table.to_string()].into());
    }

    if args.io {
        let mut table_inputs = Table::new();
        table_inputs.add_row(vec!["Name", "Type", "Shape"].into());

        let mut table_outputs = Table::new();
        table_outputs.add_row(vec!["Name", "Type", "Shape"].into());

        let io_num = rknn_model.query::<InputOutputNum>()?;
        for i in 0..io_num.input_num() {
            let input = rknn_model.query_with_input::<InputAttr>(i)?;
            table_inputs.add_row(
                vec![
                    input.name().to_string(),
                    format!("{:?}", input.dtype()),
                    format!("{:?}", input.dims()),
                ]
                .into(),
            );
        }

        for i in 0..io_num.output_num() {
            let output = rknn_model.query_with_input::<OutputAttr>(i)?;
            table_outputs.add_row(
                vec![
                    output.name().to_string(),
                    format!("{:?}", output.dtype()),
                    format!("{:?}", output.dims()),
                ]
                .into(),
            );
        }
        table_full.add_row(vec!["Inputs", "Outputs"].into());
        table_full.add_row(vec![table_inputs.to_string(), table_outputs.to_string()].into());
    }
    let mut table = Table::new();
    if args.perf {
        use rknpu2::tensor::builder::TensorBuilder;

        let io_num = rknn_model.query::<InputOutputNum>()?;

        let mut input_tensors = Vec::new();

        for i in 0..io_num.input_num() {
            let mut tensor = TensorBuilder::new_input(&rknn_model, i).allocate::<half::f16>()?;
            tensor.fill_with(half::f16::ZERO);

            input_tensors.push(tensor);
        }

        rknn_model.set_inputs(&input_tensors)?;
        rknn_model.run()?;

        let perf_info = rknn_model.query::<PerfDetail>()?;

        table.add_row(vec!["Performance Information"].into());
        table.add_row(vec![perf_info.details()].into());
    }

    table_full.printstd();
    table.printstd();

    Ok(())
}

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(help = "Path to the model file")]
    pub model_path: String,
    #[clap(short, long, help = "Show inputs and outputs")]
    pub io: bool,
    #[clap(short, long, help = "Show native input/output information")]
    pub native_io: bool,
    #[clap(short, long, help = "Enable performance profiling")]
    pub perf: bool,
    #[clap(short, long, help = "Show SDK information")]
    pub sdk: bool,
}
