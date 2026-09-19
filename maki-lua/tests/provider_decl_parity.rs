//! The `synthetic` port, proved by equality.
//!
//! maki authors a declaration for the slug in Rust and bundles a Lua plugin
//! that authors the same one. Because that provider needs no hook at all, the
//! two [`ProviderDecl`]s being equal is the *whole* of the equivalence: there
//! is no behaviour left for the comparison to miss, and no golden, transcript
//! or wire assertion could say more than this does.
//!
//! The registry, the environment and `providers.toml` are process-global, so
//! each test here boots its own host and leans on `cargo nextest` giving every
//! test its own process.

use std::collections::HashMap;
use std::sync::Arc;

use maki_agent::tools::ToolRegistry;
use maki_config::PluginsConfig;
use maki_lua::{PluginHost, PluginPermissions};
use maki_providers::plugin::{self, ProviderDecl};
use tempfile::TempDir;

const SLUG: &str = "synthetic";
const NET_HOST: &str = "api.synthetic.new";
const PROVIDERS_FILE: &str = "providers.toml";
/// A `providers.toml` entry that picks the author and defines nothing, which
/// is the whole of the escape hatch.
const PINNED_TO_RUST: &str = "[synthetic]\nimpl = \"rust\"\n";
/// Not the origin either declaration carries, so a Lua decl that won the slug
/// would be visible rather than indistinguishable.
const DEVIANT_BASE_URL: &str = "https://api.synthetic.new/openai/v2";

const HOST_FAILED: &str = "the plugin host did not start";
const LOAD_FAILED: &str = "the synthetic plugin did not load";
const TEMPDIR_FAILED: &str = "no temporary state directory";
const CONFIG_DIR_FAILED: &str = "the isolated config directory did not resolve";
const WRITE_FAILED: &str = "the providers config could not be written";
const NO_RUST_DECL: &str = "synthetic is missing from the declarations maki authors";
const NO_REGISTERED_DECL: &str = "no declaration serves synthetic after the load";
const NOT_RENDERABLE: &str = "a declaration is always serialisable";
const DECLS_DIFFER: &str = "the two authorings of synthetic declare different providers";

/// Points every base directory at a throwaway tree, so neither the real
/// `providers.toml` nor this machine's credentials reach the registration.
fn isolated_state() -> TempDir {
    let dir = TempDir::new().expect(TEMPDIR_FAILED);
    for var in [
        "HOME",
        "XDG_STATE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
    ] {
        unsafe { std::env::set_var(var, dir.path()) };
    }
    dir
}

fn write_providers_config(contents: &str) {
    let dir = maki_storage::paths::config_dir().expect(CONFIG_DIR_FAILED);
    std::fs::write(dir.join(PROVIDERS_FILE), contents).expect(WRITE_FAILED);
}

/// Only the one bundled plugin, so what registers the slug is unambiguous.
fn only_synthetic() -> PluginsConfig {
    PluginsConfig {
        enabled: true,
        names: vec![SLUG.to_owned()],
        packages: Vec::new(),
        opts: HashMap::new(),
    }
}

fn host() -> PluginHost {
    PluginHost::new(Arc::new(ToolRegistry::new())).expect(HOST_FAILED)
}

fn deviant_plugin() -> String {
    format!(
        r#"
maki.provider.register({{
  slug = "{SLUG}",
  codec = "openai",
  base_url = "{DEVIANT_BASE_URL}",
}})
"#
    )
}

/// `ProviderDecl` is `Serialize` and not `Debug`, and the serialised form is
/// the one worth reading on a failure anyway: it is the declaration as every
/// other reader of it sees it.
fn assert_same_decl(left: &ProviderDecl, right: &ProviderDecl) {
    let render = |decl| serde_json::to_string_pretty(decl).expect(NOT_RENDERABLE);
    assert!(
        left == right,
        "{DECLS_DIFFER}\n--- registered\n{}\n--- authored in rust\n{}",
        render(left),
        render(right)
    );
}

fn permissions_for(host: &str) -> PluginPermissions {
    let mut permissions = PluginPermissions::trusted();
    permissions.set_net_hosts(Some(Arc::from(vec![host.to_owned()])));
    permissions
}

/// Deliberately without `plugin::begin_load`, which would stage the
/// Rust-authored declaration first: nothing but the bundled plugin registers
/// here, so the declaration standing at the end is provably the Lua one and
/// the comparison cannot pass by finding maki's own on both sides.
#[test]
fn the_bundled_lua_decl_is_the_declaration_maki_authors() {
    let _state = isolated_state();
    let mut host = host();
    host.load_builtins(&only_synthetic()).expect(LOAD_FAILED);
    plugin::commit_load();

    let lua = plugin::registered_decl(SLUG).expect(NO_REGISTERED_DECL);
    assert_same_decl(&lua, &plugin::rust_decl(SLUG).expect(NO_RUST_DECL));
}

/// The escape hatch: `impl = "rust"` pins the shipped declaration for a user
/// the Lua one misbehaves for. The plugin still loads and still registers --
/// it simply loses the slug -- so the origin it declares is what tells the two
/// apart.
#[test]
fn the_configured_impl_keeps_the_declaration_maki_ships() {
    let _state = isolated_state();
    write_providers_config(PINNED_TO_RUST);

    let host = host();
    plugin::begin_load();
    host.load_source_with_permissions(SLUG, &deviant_plugin(), permissions_for(NET_HOST))
        .expect(LOAD_FAILED);
    plugin::commit_load();

    let standing = plugin::registered_decl(SLUG).expect(NO_REGISTERED_DECL);
    assert_same_decl(&standing, &plugin::rust_decl(SLUG).expect(NO_RUST_DECL));
    assert_ne!(standing.base_url.as_deref(), Some(DEVIANT_BASE_URL));
}
