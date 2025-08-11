use {
    clap::{Parser, ValueEnum},
    rknpu2::rknpu2_sys::_rknn_core_mask::{
        RKNN_NPU_CORE_0, RKNN_NPU_CORE_0_1, RKNN_NPU_CORE_0_1_2, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
        RKNN_NPU_CORE_ALL, RKNN_NPU_CORE_AUTO,
    },
};

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(help = "Path to the model file")]
    pub model_path: String,

    #[clap(long, default_value_t = 0, help = "Which library path to use")]
    pub lib_index: usize,

    #[clap(short, long, help = "Show inputs and outputs")]
    pub io: bool,

    #[clap(short = 'n', long, help = "Show native input/output information")]
    pub native_io: bool,

    #[clap(long, help = "Show native input/output information in NHWC format")]
    pub native_nhwc_io: bool,

    #[clap(long, help = "Show native input/output information in NC1HWC2 format")]
    pub native_nc1hwc2_io: bool,

    #[clap(short, long, help = "Enable performance profiling")]
    pub perf: bool,

    #[clap(short, long, help = "Show SDK information")]
    pub sdk: bool,

    #[clap(long, value_enum, default_value_t = NpuCore::Auto, help = "Select NPU cores to use")]
    pub npu_cores: NpuCore,

    #[clap(long, default_value_t = false, help = "Output in Markdown format")]
    pub markdown: bool,

    #[clap(
        long,
        default_value_t = false,
        help = "Show full name of the op in the --perf output"
    )]
    pub full_name: bool,

    #[clap(long, default_value_t = false, help = "Show full io information")]
    pub full: bool,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum NpuCore {
    Core0,
    Core1,
    Core2,
    CoreAll,
    Core0_1,
    Core0_1_2,
    Auto,
}

impl NpuCore {
    pub fn as_rknn_const(&self) -> u32 {
        match self {
            NpuCore::Core0 => RKNN_NPU_CORE_0,
            NpuCore::Core1 => RKNN_NPU_CORE_1,
            NpuCore::Core2 => RKNN_NPU_CORE_2,
            NpuCore::CoreAll => RKNN_NPU_CORE_ALL,
            NpuCore::Core0_1 => RKNN_NPU_CORE_0_1,
            NpuCore::Core0_1_2 => RKNN_NPU_CORE_0_1_2,
            NpuCore::Auto => RKNN_NPU_CORE_AUTO,
        }
    }
}
