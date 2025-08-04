use {
    crate::{cli::Args, io::do_io, perf::do_perf, sdk::do_sdk},
    clap::Parser,
    rknpu2::{RKNN, rknpu2_sys::RKNN_FLAG_COLLECT_PERF_MASK},
    stanza::renderer::console::Console,
};

mod cli;
mod io;
mod native_io;
mod perf;
mod sdk;

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

    let console = Console::default();

    if args.sdk {
        if let Err(e) = do_sdk(&rknn_model, &console) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.io {
        if let Err(e) = do_io(&rknn_model, &console) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.native_io {
        if let Err(e) = native_io::do_native_io(&rknn_model, &console) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.perf {
        let core_mask = args.npu_cores.as_rknn_const();

        if let Err(e) = do_perf(&rknn_model, core_mask) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
