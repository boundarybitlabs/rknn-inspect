use rknpu2::{
    RKNN,
    api::runtime::RuntimeAPI,
    query::{InputOutputNum, PerfDetail},
    tensor::builder::TensorBuilder,
};

pub fn do_perf(rknn_model: &RKNN<RuntimeAPI>) -> Result<(), Box<dyn std::error::Error>> {
    let io_num = rknn_model.query::<InputOutputNum>()?;

    let mut input_tensors = Vec::new();

    for i in 0..io_num.input_num() {
        let mut tensor = TensorBuilder::new_input(&rknn_model, i).allocate::<i8>()?;
        tensor.fill_with(0i8);

        input_tensors.push(tensor);
    }

    rknn_model.set_inputs(&input_tensors)?;
    rknn_model.run()?;

    let perf_info = rknn_model.query::<PerfDetail>()?;
    println!("{}", perf_info.details());

    Ok(())
}
