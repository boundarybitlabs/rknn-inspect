use rknpu2::{
    RKNN,
    api::runtime::RuntimeAPI,
    bf16, f16,
    query::{InputAttr, InputOutputNum, PerfDetail},
    tensor::{TensorT, TensorType, builder::TensorBuilder, tensor::Tensor},
};

pub fn do_perf(rknn_model: &RKNN<RuntimeAPI>) -> Result<(), Box<dyn std::error::Error>> {
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
    println!("{}", perf_info.details());

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
