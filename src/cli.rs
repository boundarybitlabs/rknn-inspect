use clap::Parser;

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
