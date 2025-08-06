use {
    rknpu2::{
        RKNN,
        api::runtime::RuntimeAPI,
        query::{InputOutputNum, NativeInputAttr, NativeOutputAttr},
    },
    stanza::{
        renderer::Renderer,
        style::Styles,
        table::{Cell, Row, Table},
    },
    std::error::Error,
};

pub fn do_native_io(
    rknn_model: &RKNN<RuntimeAPI>,
    console: &dyn Renderer<Output = String>,
) -> Result<(), Box<dyn Error>> {
    // Subtable for inputs
    let mut table_inputs = Table::default();
    table_inputs.push_row(vec![
        "Name",
        "Type",
        "Shape",
        "Format",
        "Quantization",
        "Quant Param",
    ]);

    // Subtable for outputs
    let mut table_outputs = Table::default();
    table_outputs.push_row(vec![
        "Name",
        "Type",
        "Shape",
        "Format",
        "Quantization",
        "Quant Param",
    ]);

    // Query number of IOs
    let io_num = rknn_model.query::<InputOutputNum>()?;

    for i in 0..io_num.input_num() {
        let input = rknn_model.query_with_input::<NativeInputAttr>(i)?;
        table_inputs.push_row(vec![
            input.name().to_string(),
            format!("{:?}", input.dtype()),
            format!("{:?}", input.dims()),
            format!("{:?}", input.format()),
            format!("{:?}", input.qnt_type()),
            format!("{:?}", input.affine_asymmetric_param()),
        ]);
    }

    for i in 0..io_num.output_num() {
        let output = rknn_model.query_with_input::<NativeOutputAttr>(i)?;
        table_outputs.push_row(vec![
            output.name().to_string(),
            format!("{:?}", output.dtype()),
            format!("{:?}", output.dims()),
            format!("{:?}", output.format()),
            format!("{:?}", output.qnt_type()),
            format!("{:?}", output.affine_asymmetric_param()),
        ]);
    }

    // Top-level table with nested tables
    let mut table_full = Table::default();
    table_full.push_row(vec!["Native Inputs", "Native Outputs"]);
    table_full.push_row(Row::new(
        Styles::default(),
        vec![Cell::from(table_inputs), Cell::from(table_outputs)],
    ));

    println!("{}", console.render(&table_full).to_string());
    Ok(())
}
