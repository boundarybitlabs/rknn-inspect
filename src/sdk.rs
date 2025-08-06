use {
    rknpu2::{RKNN, api::runtime::RuntimeAPI, query::SdkVersion},
    stanza::{renderer::Renderer, table::Table},
};

pub fn do_sdk(
    rknn: &RKNN<RuntimeAPI>,
    console: &dyn Renderer<Output = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut table = Table::default().with_row(vec!["Component", "Version"]);
    let sdk = rknn.query::<SdkVersion>()?;
    table.push_row(vec!["SDK".into(), sdk.api_version()]);
    table.push_row(vec!["Driver".into(), sdk.driver_version()]);

    println!("{}", console.render(&table).to_string());
    Ok(())
}
