//! What a `providers.toml` entry means for a slug a declaration serves.
//!
//! Out of line rather than beside [`maki_providers::plugin`]'s own tests
//! because the config is read from the process-wide home directory: each case
//! needs its own, and `cargo nextest` is what gives a test its own process.

use maki_config::providers::Protocol;
use maki_providers::plugin::{
    self, DeclAuthority, DeclSource, ProviderDecl, RegisterError, Registration,
};
use maki_providers::spec::Owner;
use tempfile::TempDir;

const HOME_VARS: &[&str] = &[
    "HOME",
    "XDG_STATE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
];

const PROVIDERS_FILE: &str = "providers.toml";

/// A built-in slug that has moved onto a declaration, and the gateway a user
/// points it at.
const CLAIMED_SLUG: &str = "deepseek";
const GATEWAY_URL: &str = "https://gateway.example/v1";

const OWN_SLUG: &str = "configured-provider";
const OWN_PROTOCOL: &str = "openai";
const OWN_BASE_URL: &str = "https://configured.example/v1";
const OWN_HOST: &str = "declared.example";
const OWN_DISPLAY_NAME: &str = "Configured";

const TEMPDIR_FAILED: &str = "no temporary state directory";
const CONFIG_DIR_FAILED: &str = "the isolated config directory did not resolve";
const WRITE_FAILED: &str = "providers.toml could not be written";
const PROVIDER_LOST: &str = "a providers.toml overlay took the slug off its declaration";

/// Points every base directory at a throwaway tree holding `providers.toml`,
/// so no case reads this machine's config. The file goes wherever maki's own
/// path resolution says it lives, so a test can never quietly write somewhere
/// nothing reads.
fn isolated(providers_toml: &str) -> TempDir {
    let dir = TempDir::new().expect(TEMPDIR_FAILED);
    for var in HOME_VARS {
        unsafe { std::env::set_var(var, dir.path()) };
    }
    let config = maki_storage::paths::config_dir().expect(CONFIG_DIR_FAILED);
    std::fs::write(config.join(PROVIDERS_FILE), providers_toml).expect(WRITE_FAILED);
    dir
}

fn declaration() -> Registration {
    Registration {
        decl: ProviderDecl {
            slug: OWN_SLUG.to_owned(),
            display_name: Some(OWN_DISPLAY_NAME.to_owned()),
            codec: Some(Protocol::Openai),
            base: None,
            base_url: None,
            api_key_env: None,
            system_prefix: None,
            max_tokens_field: None,
            include_stream_usage: None,
            thinking_dialect: None,
            models: Vec::new(),
            net_hosts: vec![OWN_HOST.to_owned()],
        },
        hooks: plugin::ProviderHooks::default(),
    }
}

/// The documented overlay: `[<builtin>] base_url` points the shipped provider
/// at a gateway. The built-in owns the slug, so the entry is not a second
/// provider competing for it and must not cost the declaration its slug, which
/// would leave the user with a models.dev stub in place of the provider the
/// overlay was written for.
#[test]
fn an_overlay_on_a_built_in_slug_keeps_the_declaration() {
    let _home = isolated(&format!("[{CLAIMED_SLUG}]\nbase_url = \"{GATEWAY_URL}\"\n"));
    plugin::begin_load();
    plugin::commit_load();

    assert!(
        matches!(Owner::of(CLAIMED_SLUG), Owner::Plugin),
        "{PROVIDER_LOST}"
    );
    assert_eq!(
        plugin::effective_base_url(CLAIMED_SLUG).as_deref(),
        Some(GATEWAY_URL)
    );
}

/// A slug maki has no row for is a different question: nothing is inherited,
/// the entry is the whole provider, and a declaration claiming it would be a
/// second definition of the same name.
#[test]
fn a_slug_providers_toml_defines_is_still_refused() {
    let _home = isolated(&format!(
        "[{OWN_SLUG}]\nprotocol = \"{OWN_PROTOCOL}\"\nbase_url = \"{OWN_BASE_URL}\"\n"
    ));
    plugin::begin_load();
    let error =
        plugin::register(declaration(), DeclSource::Lua, DeclAuthority::ThirdParty).unwrap_err();
    plugin::commit_load();

    assert!(matches!(error, RegisterError::ConfiguredSlug(_)), "{error}");
}
