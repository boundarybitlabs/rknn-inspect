use {
    crate::utils::push_attr_row,
    rknpu2::{
        RKNN,
        api::runtime::RuntimeAPI,
        query::{InputOutputNum, QueryWithInput, TensorAttrView},
    },
    stanza::{renderer::Renderer, table::Table},
    std::error::Error,
};

pub fn do_io<
    I: QueryWithInput<Input = u32> + TensorAttrView,
    O: QueryWithInput<Input = u32> + TensorAttrView,
>(
    rknn_model: &RKNN<RuntimeAPI>,
    console: &dyn Renderer<Output = String>,
    full: bool,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let mut tbl = Table::default();
    if full {
        tbl.push_row(vec![
            "IO",
            "Name",
            "DType",
            "Bit",
            "Signed",
            "Shape",
            (prefix.to_string() + "Format").as_str(),
            "Quant",
            "Scale",
            "ZeroPt",
            "DFP(fl)",
            "Qmin",
            "Qmax",
            "FloatRange≈",
            "Size(B)",
            "Size+Stride(B)",
            "Contig",
            "Stride N,H,W,C (bytes)",
            "Notes",
        ]);
    } else {
        tbl.push_row(vec![
            "IO",
            "Name",
            "DType",
            "Shape",
            (prefix.to_string() + "Format").as_str(),
            "Quant",
            "Scale",
            "ZeroPt",
            "QRange",
            "Size(B)",
            "Stride N,H,W,C",
            "Notes",
        ]);
    }

    let io_num = rknn_model.query::<InputOutputNum>()?;

    for i in 0..io_num.input_num() {
        let a = rknn_model.query_with_input::<I>(i)?;
        push_attr_row(&mut tbl, &a, full);
    }
    for i in 0..io_num.output_num() {
        let a = rknn_model.query_with_input::<O>(i)?;
        push_attr_row(&mut tbl, &a, full);
    }

    println!("{}", console.render(&tbl).to_string());
    Ok(())
}
