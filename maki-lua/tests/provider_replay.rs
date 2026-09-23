//! The bundled provider plugins: the declaration each one authors, and the
//! recorded exchanges each one replays.
//!
//! maki writes every ported provider twice, once in Rust and once as a plugin
//! bundled in the binary, and the bundled one outranks the other at every real
//! startup. Two things follow. The two declarations have to agree, which for a
//! provider with no hooks at all is the whole story. And the plugin as shipped
//! has to replay the same recorded exchanges maki's own authoring is pinned
//! to: the goldens live over in `maki-providers` next to the artifacts, both
//! authorings assert against those same files, and neither reading dies when
//! the other one is deleted.
//!
//! Recorded on loopback, which is what `<SLUG>_BASE_URL` points at here. The
//! plugins declare their real hosts and nothing else, so reaching the recorded
//! server is the same widening a user with a gateway relies on.
//!
//! The registry, the environment and `providers.toml` are all process wide, so
//! every test boots its own host and counts on `cargo nextest` giving each test
//! a process of its own.

use std::collections::HashMap;
use std::sync::Arc;

use maki_agent::tools::ToolRegistry;
use maki_config::PluginsConfig;
use maki_lua::{PluginHost, PluginPermissions};
use maki_providers::plugin::{self, ProviderDecl};
use maki_providers::replay::{self, Fixture};
use maki_providers::{
    ThinkingConfig, deepseek_fixtures as deepseek, synthetic_fixtures as synthetic,
};
use serde_json::Value;
use tempfile::TempDir;
use test_case::test_case;

const SYNTHETIC: &str = "synthetic";
const SYNTHETIC_HOST: &str = "api.synthetic.new";
const DEEPSEEK: &str = "deepseek";
const DEEPSEEK_HOST: &str = "api.deepseek.com";
const LOOPBACK_HOST: &str = "127.0.0.1";

const PROVIDERS_FILE: &str = "providers.toml";
/// The whole of the escape hatch: it picks the author and defines nothing.
const PIN_TO_RUST: &str = "impl = \"rust\"";
/// An origin no authoring declares, so a decl carrying it is visibly the
/// deviant plugin's rather than maki's.
const DEVIANT_PATH: &str = "/deviant";

const HOST_FAILED: &str = "the plugin host did not start";
const LOAD_FAILED: &str = "the bundled provider plugin did not load";
const TEMPDIR_FAILED: &str = "no temporary state directory";
const CONFIG_DIR_FAILED: &str = "the isolated config directory did not resolve";
const WRITE_FAILED: &str = "the providers config could not be written";
const NO_RUST_DECL: &str = "the slug is missing from the declarations maki authors";
const NO_REGISTERED_DECL: &str = "no declaration serves the slug after the load";
const NOT_RENDERABLE: &str = "a declaration is always serialisable";
const DECLS_DIFFER: &str = "the two authorings declare different providers";

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

fn plugin_host() -> PluginHost {
    PluginHost::new(Arc::new(ToolRegistry::new())).expect(HOST_FAILED)
}

/// One bundled plugin and no others, so what registers the slug is never
/// ambiguous, loaded with the permissions its own `plugin.toml` declares, so
/// what is under test is the plugin exactly as it ships.
///
/// Loopback is waved past the SSRF guard because that is where the recorded
/// server listens. The host comes back instead of being dropped, since a hook
/// whose host has died answers nothing.
fn load_bundled(slug: &str) -> PluginHost {
    maki_lua::set_allowed_private_hosts(&[LOOPBACK_HOST.to_owned()]);
    let mut host = plugin_host();
    host.load_builtins(&PluginsConfig {
        enabled: true,
        names: vec![slug.to_owned()],
        packages: Vec::new(),
        opts: HashMap::new(),
    })
    .expect(LOAD_FAILED);
    host
}

/// The same load, handed to the replay harness so it runs inside the harness's
/// own window, over the Rust declaration staged first. That ordering is the
/// precedence a real startup applies.
fn bundled(slug: &'static str) -> impl FnOnce() -> PluginHost {
    move || load_bundled(slug)
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

/// Deliberately without `plugin::begin_load`, which would stage the
/// Rust-authored declaration first. Nothing but the bundled plugin registers
/// here, so the declaration standing at the end is provably the Lua one and the
/// comparison cannot pass by finding maki's own on both sides.
#[test_case(SYNTHETIC ; "synthetic")]
#[test_case(DEEPSEEK ; "deepseek")]
fn the_bundled_lua_decl_is_the_declaration_maki_authors(slug: &str) {
    let _state = isolated_state();
    let _host = load_bundled(slug);
    plugin::commit_load();

    let lua = plugin::registered_decl(slug).expect(NO_REGISTERED_DECL);
    assert_same_decl(&lua, &plugin::rust_decl(slug).expect(NO_RUST_DECL));
}

/// The escape hatch for a user the Lua authoring misbehaves for: `impl =
/// "rust"` in `providers.toml` pins the shipped declaration. The plugin still
/// loads and still registers and simply loses the slug, so the origin it wanted
/// to declare is what tells the two apart.
#[test_case(SYNTHETIC, SYNTHETIC_HOST ; "synthetic")]
#[test_case(DEEPSEEK, DEEPSEEK_HOST ; "deepseek")]
fn the_configured_impl_keeps_the_declaration_maki_ships(slug: &str, net_host: &str) {
    let _state = isolated_state();
    let config_dir = maki_storage::paths::config_dir().expect(CONFIG_DIR_FAILED);
    std::fs::write(
        config_dir.join(PROVIDERS_FILE),
        format!("[{slug}]\n{PIN_TO_RUST}\n"),
    )
    .expect(WRITE_FAILED);

    let deviant = format!("https://{net_host}{DEVIANT_PATH}");
    let mut permissions = PluginPermissions::trusted();
    permissions.set_net_hosts(Some(Arc::from(vec![net_host.to_owned()])));

    let source = format!(
        r#"maki.provider.register({{ slug = "{slug}", codec = "openai", base_url = "{deviant}" }})"#
    );

    let host = plugin_host();
    plugin::begin_load();
    host.load_source_with_permissions(slug, &source, permissions)
        .expect(LOAD_FAILED);
    plugin::commit_load();

    let standing = plugin::registered_decl(slug).expect(NO_REGISTERED_DECL);
    assert_same_decl(&standing, &plugin::rust_decl(slug).expect(NO_RUST_DECL));
    assert_ne!(standing.base_url.as_deref(), Some(deviant.as_str()));
}

#[test_case(&synthetic::SUCCESS ; "success")]
#[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
#[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
#[test_case(&replay::RATE_LIMITED ; "rate_limited")]
#[test_case(&replay::SERVER_ERROR ; "server_error")]
#[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
#[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
#[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
fn the_bundled_synthetic_plugin_replays_the_recorded_exchange(fixture: &Fixture) {
    replay::declared(bundled(SYNTHETIC), SYNTHETIC, fixture, &synthetic::model());
}

#[test_case(&deepseek::SUCCESS ; "success")]
#[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
#[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
#[test_case(&replay::RATE_LIMITED ; "rate_limited")]
#[test_case(&replay::SERVER_ERROR ; "server_error")]
#[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
#[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
#[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
#[test_case(&deepseek::THINKING_OFF ; "thinking_off")]
#[test_case(&deepseek::THINKING_ADAPTIVE ; "thinking_adaptive")]
fn the_bundled_deepseek_plugin_replays_the_recorded_exchange(fixture: &Fixture) {
    replay::declared(
        bundled(DEEPSEEK),
        DEEPSEEK,
        fixture,
        &deepseek::model(deepseek::FLASH_SPEC),
    );
}

/// The `reasoning_content` back-fill, which needs a history to act on and a
/// tool list to be allowed to.
#[test_case("padding_with_tools", deepseek::FLASH_SPEC, true ; "a turn with no reasoning gets some")]
#[test_case("padding_without_tools", deepseek::FLASH_SPEC, false ; "no tools, nothing added")]
#[test_case("reasoner_with_tools", deepseek::REASONER_SPEC, true ; "the model that refuses the field")]
fn the_bundled_deepseek_plugin_pads_the_same_turns(
    name: &'static str,
    spec: &str,
    with_tools: bool,
) {
    let fixture = Fixture {
        name,
        script: deepseek::SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(deepseek::EFFORT),
        session: None,
    };
    let tools = if with_tools {
        replay::tools()
    } else {
        Value::Array(Vec::new())
    };
    replay::declared_with(
        bundled(DEEPSEEK),
        DEEPSEEK,
        &fixture,
        &deepseek::model(spec),
        &deepseek::history(),
        &tools,
    );
}

/// The balance endpoint, which is the one hook that leaves the codec's request
/// path: the Lua authoring reaches it through `maki.net`, under the plugin's
/// own declared hosts, and the golden says it asked the same server the same
/// question as the Rust one.
#[test]
fn the_bundled_deepseek_plugin_reads_the_balance_endpoint() {
    replay::declared_usage(bundled(DEEPSEEK), DEEPSEEK, &deepseek::BALANCE);
}
