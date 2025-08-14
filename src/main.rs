use {
    crate::{cli::Args, io::do_io, perf::do_perf, sdk::do_sdk},
    clap::Parser,
    rknpu2::{
        RKNN,
        query::{
            InputAttr, NativeInputAttr, NativeNC1HWC2InputAttr, NativeNC1HWC2OutputAttr,
            NativeNHWCInputAttr, NativeNHWCOutputAttr, NativeOutputAttr, OutputAttr,
        },
        rknpu2_sys::RKNN_FLAG_COLLECT_PERF_MASK,
        utils::find_rknn_library,
    },
    stanza::{
        renderer::{
            Renderer,
            console::{Console, Decor, Line},
            markdown::Markdown,
        },
        table::Table,
    },
};

mod cli;
mod io;
mod perf;
mod sdk;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let console: Box<dyn Renderer<Output = String>> = if args.markdown {
        Box::new(Markdown::default())
    } else {
        let decor = Decor {
            remap_thin_to: Line::Thin,
            remap_bold_to: Line::Thin,
            print_escape_codes: false,
            draw_outer_border: true,
            ..Decor::default()
        };

        Box::new(Console(decor))
    };

    let library_paths = find_rknn_library().collect::<Vec<_>>();

    if library_paths.is_empty() {
        eprintln!("No RKNN library found");
        std::process::exit(1);
    }

    let mut table = Table::default();
    table.push_row(["Index", "Library Path"]);

    for (i, path) in library_paths.iter().enumerate() {
        table.push_row([i.to_string(), path.display().to_string()]);
    }

    println!("{}", console.render(&table).to_string());

    let mut bytes = std::fs::read(&args.model_path)?;

    let library_path = match library_paths.get(args.lib_index) {
        Some(path) => path.clone(),
        None => {
            eprintln!("Invalid library index");
            std::process::exit(1);
        }
    };

    let rknn_model =
        match RKNN::new_with_library(library_path, &mut bytes, RKNN_FLAG_COLLECT_PERF_MASK) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("Failed to load model: {}", err);
                std::process::exit(1);
            }
        };

    if args.sdk {
        if let Err(e) = do_sdk(&rknn_model, &*console) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.io {
        if let Err(e) = do_io::<InputAttr, OutputAttr>(&rknn_model, &*console, args.full, "") {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.native_io {
        if let Err(e) =
            do_io::<NativeInputAttr, NativeOutputAttr>(&rknn_model, &*console, args.full, "Native ")
        {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.native_nhwc_io {
        if let Err(e) = do_io::<NativeNHWCInputAttr, NativeNHWCOutputAttr>(
            &rknn_model,
            &*console,
            args.full,
            "Native NHWC ",
        ) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.native_nc1hwc2_io {
        if let Err(e) = do_io::<NativeNC1HWC2InputAttr, NativeNC1HWC2OutputAttr>(
            &rknn_model,
            &*console,
            args.full,
            "Native NC1HWC2 ",
        ) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.perf {
        let core_mask = args.npu_cores.as_rknn_const();

        if let Err(e) = do_perf(&rknn_model, core_mask, &*console, args.full_name) {
            println!("Error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
