use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use flume::Sender;
use maki_config::host_allowed;
use maki_config::providers::{
    ImplChoice, Protocol, ProviderDef, ProvidersConfig, configured_base_url,
};
use maki_storage::StateDir;
use maki_storage::auth::lock_credentials;
use maki_storage::id::SessionRef;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, warn};
use url::{Host, Url};

use crate::model::{
    Model, ModelInfo, ModelPricing, ModelTier, Prefixed, ThinkingSupport, longest_prefix_match,
};
use crate::provider::{BoxFuture, Provider};
use crate::spec::{ProviderRegistry, ProviderSpec};
use crate::types::ThinkingFields;
use crate::{AgentError, Message, ProviderEvent, ProviderUsage, RequestOptions, StreamResponse};

use super::codec::{self, BodyHook, CodecOptions, RequestCtx};
pub use super::codec::{EffortField, OpenAiWire, SessionCarrier, ThinkingWire};
use super::{
    KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts, deepseek, mistral, openrouter, regolo,
    requesty, synthetic, tensorx,
};

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 16384;
const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;
const BUILD_BODY_OPTION: &str = "build_body hook";
const SYSTEM_PREFIX_OPTION: &str = "system_prefix";
const OPENAI_OPTION: &str = "openai";
const DISPLAY_NAME_FIELD: &str = "display_name";
const API_KEY_ENV_FIELD: &str = "api_key_env";
const MODELS_FIELD: &str = "models";
const IMPL_FIELD: &str = "impl";
const HTTPS_SCHEME: &str = "https";
const HTTP_SCHEME: &str = "http";
const LOCALHOST: &str = "localhost";

/// One plugin-supplied callback. Generic both ways so every hook on
/// [`ProviderHooks`] has the same shape and presence is one `Option` check.
pub trait Hook<In, Out>: Send + Sync {
    fn call(&self, input: In) -> BoxFuture<'_, Result<Out, AgentError>>;
}

#[derive(Default, Clone)]
pub struct ProviderHooks {
    pub auth: Option<Arc<dyn Hook<AuthPurpose, PluginAuth>>>,
    pub list_models: Option<Arc<dyn Hook<(), Vec<ModelInfo>>>>,
    pub build_body: Option<Arc<dyn Hook<BodyInput, Value>>>,
    pub map_error: Option<Arc<dyn Hook<ApiError, Option<ApiError>>>>,
    pub fetch_usage: Option<Arc<dyn Hook<(), Option<ProviderUsage>>>>,
    pub login: Option<Arc<dyn Hook<(), ()>>>,
    pub logout: Option<Arc<dyn Hook<(), ()>>>,
}

/// Why the auth hook is being asked for credentials. `Reload` only re-reads
/// what a login wrote, which is what lets it skip the cross-process lock.
#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum AuthPurpose {
    Resolve,
    Refresh,
    Reload,
}

/// The request as it goes on the wire, plus the things a plugin branches on.
/// `thinking` is rendered text and not structure, because the hook is a
/// wire-level escape hatch and not a second place to model effort.
#[derive(Serialize)]
pub struct BodyInput {
    pub body: Value,
    pub model: String,
    pub thinking: String,
    /// The [`ModelInfo::extra`] this slug's `list_models` attached to the
    /// model, absent when discovery said nothing about it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<Value>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

#[derive(Deserialize)]
pub struct PluginAuth {
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
}

impl PluginAuth {
    /// The one door plugin-supplied credentials come through, so a plugin
    /// cannot point maki's tokens at a host it never declared.
    pub fn into_resolved(self, slug: &str, hosts: &[String]) -> Result<ResolvedAuth, AgentError> {
        let base_url = declared_base_url(slug, self.base_url, hosts)
            .map_err(|message| AgentError::Config { message })?;
        Ok(ResolvedAuth::new(slug, self.headers.into_iter().collect())?.with_base_url(base_url))
    }
}

/// The only door an origin a plugin chose comes through, whether it arrived
/// with the registration or from an auth hook. Both end up holding the token
/// maki sends, so both answer to the same host list.
///
/// The scheme is half of the promise: a declared host reached over plaintext
/// still puts the token on the wire in the clear. Only `https` passes, plus
/// `http` on loopback, where a self-hosted provider has no wire to listen on.
fn declared_base_url(
    slug: &str,
    base_url: Option<String>,
    hosts: &[String],
) -> Result<Option<String>, String> {
    let Some(url) = &base_url else {
        return Ok(None);
    };
    let parsed = Url::parse(url)
        .map_err(|e| format!("provider '{slug}': base_url '{url}' is not a url: {e}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| format!("provider '{slug}': base_url '{url}' has no host"))?;
    let scheme = parsed.scheme();
    if scheme != HTTPS_SCHEME && !(scheme == HTTP_SCHEME && is_loopback(&parsed)) {
        return Err(format!(
            "provider '{slug}': base_url '{url}' would send credentials over '{scheme}'; use \
             https, or http only for loopback"
        ));
    }
    if !host_allowed(host, hosts) {
        return Err(format!(
            "provider '{slug}': base_url host '{host}' is not in the declared net hosts"
        ));
    }
    Ok(base_url)
}

fn is_loopback(url: &Url) -> bool {
    match url.host() {
        Some(Host::Domain(name)) => name == LOCALHOST,
        Some(Host::Ipv4(addr)) => addr.is_loopback(),
        Some(Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}

/// One declared model row. `Serialize` mirrors `Deserialize` field for field,
/// defaults included, so a row survives a round trip and two spellings of the
/// same provider compare equal whichever way they were authored.
#[derive(Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PluginModel {
    /// Every id this row answers for. `prefixes[0]` is the canonical id,
    /// used wherever a concrete model has to be named.
    pub prefixes: Vec<String>,
    #[serde(default = "default_tier")]
    pub tier: ModelTier,
    #[serde(default)]
    pub supports_tool_examples: Option<bool>,
    #[serde(default)]
    pub supports_thinking: Option<bool>,
    #[serde(default)]
    pub requires_thinking: bool,
    #[serde(default)]
    pub supports_vision: Option<bool>,
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
    #[serde(default)]
    pub pricing: Option<ModelPricing>,
    #[serde(default)]
    pub thinking_fields: Option<ThinkingFields>,
}

impl Prefixed for PluginModel {
    fn prefixes(&self) -> impl Iterator<Item = &str> {
        self.prefixes.iter().map(String::as_str)
    }
}

impl PluginModel {
    fn canonical_id(&self) -> Option<&str> {
        self.prefixes.first().map(String::as_str)
    }

    fn to_model(&self, slug: &str, base: &'static ProviderSpec, id: String) -> Model {
        Model {
            id,
            provider: Arc::from(slug),
            tier: self.tier,
            family: base.family,
            supports_tool_examples_override: self.supports_tool_examples,
            thinking_override: ThinkingSupport::from_flags(
                self.supports_thinking,
                self.requires_thinking,
            ),
            supports_vision_override: self.supports_vision,
            supports_fast_override: None,
            pricing: self.pricing.clone().unwrap_or_default(),
            subsidised_by: None,
            discovered_free: false,
            max_output_tokens: Some(self.max_output_tokens),
            turn_output_tokens: None,
            context_window: self.context_window,
            thinking_fields: self.thinking_fields.clone().map(Box::new),
        }
    }

    /// This row as the catalogue reports it. The `supports_*` flags are
    /// `Option` on both sides, so an unstated one stays unstated rather than
    /// becoming a published negative. `provider_info` is a stash only the Rust
    /// provider that filled it reads back, so a declared row never has one.
    fn to_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.canonical_id().unwrap_or_default().to_string(),
            context_window: Some(self.context_window),
            max_output_tokens: Some(self.max_output_tokens),
            pricing: self.pricing.clone(),
            supports_thinking: self.supports_thinking,
            supports_vision: self.supports_vision,
            tier: Some(self.tier),
            provider_info: None,
            extra: None,
            effort: None,
        }
    }
}

fn default_tier() -> ModelTier {
    ModelTier::Medium
}

fn default_max_output_tokens() -> u32 {
    DEFAULT_MAX_OUTPUT_TOKENS
}

fn default_context_window() -> u32 {
    DEFAULT_CONTEXT_WINDOW
}

/// One provider, as data: everything maki needs to build it except the
/// callbacks. Keeping the data on its own is what lets two spellings of the
/// same provider be compared and dumped, and for a provider with no hooks that
/// comparison is the whole proof that the Rust and the Lua authoring agree.
///
/// `PartialEq` and not `Eq` because a declared [`ModelPricing`] is four `f64`
/// rates. Comparing them bitwise is exactly the question being asked: two
/// decls state the same rate or they do not.
///
/// Decoded by serde straight off the authoring surface, and a key it does not
/// know is an error: a typo'd option would otherwise be an option silently
/// left out.
#[derive(Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDecl {
    pub slug: String,
    /// `None` only for a decl that claims a built-in slug, which inherits the
    /// name off the spec row rather than restating it.
    pub display_name: Option<String>,
    pub codec: Option<Protocol>,
    pub base: Option<String>,
    /// The declaration's static origin, below the user's `<SLUG>_BASE_URL` and
    /// `providers.toml`. An origin a hook returns is a different thing and
    /// outranks both, see [`CodecOptions::base_url`].
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub system_prefix: Option<String>,
    #[serde(default)]
    pub models: Vec<PluginModel>,
    /// Only for `codec = "openai"`: grouped under the codec that honours them,
    /// so one rule refuses every option a different target would ignore.
    pub openai: Option<OpenAiWire>,
    /// Never written by the author: a Lua plugin's come from its
    /// `plugin.toml`, where the permission to reach them is granted. Still
    /// dumped, because two decls are compared on it too.
    #[serde(skip_deserializing)]
    pub net_hosts: Vec<String>,
}

/// A declaration plus the callbacks that go with it. The split is the point:
/// the data half is comparable and dumpable, the behaviour half is neither.
/// They still travel together, so nothing builds a provider from data that was
/// never checked against its hooks.
pub struct Registration {
    pub decl: ProviderDecl,
    pub hooks: ProviderHooks,
}

/// Which language a declaration was authored in, the only axis a provider
/// varies on once it is data. Recorded rather than inferred, because nothing
/// else about an entry says where it came from and every log line about one
/// should be able to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclSource {
    Rust,
    Lua,
}

/// Who stands behind a declaration. A different question from which language
/// wrote it, and the only one that decides whether a slug maki already ships
/// may be taken over.
///
/// Claiming a built-in slug inherits that row's `api_key_env`, so the key the
/// user set for the built-in is resolved into the claimant's credentials and
/// sent to the origin the claimant declared. Exactly right for the bundled
/// plugin that *is* the provider, credential theft from anything else. A
/// package asks for `net` and a list of hosts, and nothing in that grant says
/// "and the built-in providers' keys".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclAuthority {
    /// maki's own: a Rust declaration, or a plugin compiled into the binary.
    Bundled,
    /// Anything that arrived after the build: a package, a local plugin
    /// directory, `init.lua`.
    ThirdParty,
}

/// Whether the declaration that just registered is the one serving its slug.
///
/// Answered rather than assumed, because losing is not failing: `impl` in
/// `providers.toml` picks between two authors and the loser is still a
/// well-formed declaration. A caller that grants credentials or egress off the
/// back of a registration has to grant them to the winner only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Registered {
    Serving,
    /// Another author's declaration holds the slug, see [`wins`].
    Shadowed,
}

#[derive(Debug, thiserror::Error)]
pub enum RegisterError {
    #[error(
        "invalid provider slug '{0}': must start with a letter or digit and hold only letters, digits, '_' and '-'"
    )]
    InvalidSlug(String),
    #[error("provider slug '{0}' is already defined in providers.toml")]
    ConfiguredSlug(String),
    #[error(
        "provider slug '{0}' belongs to a built-in provider and cannot be declared by a plugin maki does not ship"
    )]
    ReservedSlug(String),
    #[error("provider '{slug}': {message}")]
    Credentials { slug: String, message: String },
    #[error("provider '{0}' is already registered")]
    DuplicateSlug(String),
    #[error("provider '{0}' must set a display_name")]
    NoDisplayName(String),
    #[error(
        "provider '{slug}': {field} is inherited from the built-in provider of the same name and must not be restated"
    )]
    Restated { slug: String, field: &'static str },
    #[error("provider '{0}' must set exactly one of `codec` or `base`")]
    CodecOrBase(String),
    #[error("provider '{slug}': base '{base}' is not a native provider")]
    UnknownBase { slug: String, base: String },
    #[error("provider '{0}' must declare at least one net host")]
    NoNetHosts(String),
    #[error("{0}")]
    UndeclaredBaseUrl(String),
    #[error("provider '{0}' cannot register outside a plugin load")]
    Closed(String),
    #[error("provider '{slug}': {option} is not supported by {target}")]
    Unsupported {
        slug: String,
        option: &'static str,
        target: String,
    },
}

/// What a registered slug builds its requests with. Exactly one of the two, so
/// the impossible "neither" is unrepresentable past registration.
#[derive(Clone, Copy)]
enum Target {
    Base(&'static ProviderSpec),
    Codec(Protocol),
}

impl Target {
    /// The native spec behind this target: model family, fallbacks and the
    /// model table a plugin that curates none borrows.
    fn spec(self) -> Option<&'static ProviderSpec> {
        match self {
            Self::Base(spec) => Some(spec),
            Self::Codec(protocol) => codec::protocol_spec(protocol),
        }
    }

    fn describe(self) -> String {
        match self {
            Self::Base(spec) => format!("base '{}'", spec.slug),
            Self::Codec(protocol) => format!("codec {protocol:?}"),
        }
    }
}

/// Only the openai codecs thread a body hook through [`codec::build`].
///
/// An exhaustive match rather than a list of the ones that work, so a new
/// codec breaks this line and someone has to answer for it. An option a codec
/// cannot honour is a registration error, never a no-op.
fn honours_build_body(target: Target) -> bool {
    match target {
        Target::Codec(Protocol::Openai | Protocol::OpenaiResponses) => true,
        Target::Codec(Protocol::Anthropic | Protocol::Google) | Target::Base(_) => false,
    }
}

/// The `openai` table is the chat codec's own vocabulary. The responses codec
/// spells neither effort nor extra body the same way, so it is refused there
/// rather than half honoured.
fn honours_openai_wire(target: Target) -> bool {
    match target {
        Target::Codec(Protocol::Openai) => true,
        Target::Codec(Protocol::OpenaiResponses | Protocol::Anthropic | Protocol::Google)
        | Target::Base(_) => false,
    }
}

/// Google drops the system prefix and always has (see `super::google`), so a
/// plugin that sets one against it is told instead of ignored. Asked of the
/// spec behind the target, because `codec = "google"` and `base = "google"`
/// reach the same constructor and must answer alike.
fn honours_system_prefix(target: Target) -> bool {
    !target
        .spec()
        .is_some_and(|spec| spec.slug == super::google::SLUG)
}

fn is_valid_slug(s: &str) -> bool {
    !s.is_empty()
        && s.as_bytes()[0].is_ascii_alphanumeric()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

struct PluginEntry {
    decl: ProviderDecl,
    /// The built-in spec row this decl claimed, `None` for a slug maki did not
    /// already know. What a claim leaves out is read back off the row by
    /// [`PluginEntry::spec`] and [`PluginEntry::display_name`], so nobody else
    /// has to ask whether an entry is a claim.
    claimed: Option<&'static ProviderSpec>,
    source: DeclSource,
    /// The plugin that registered this, `None` for maki's own Rust decls.
    owner: Option<Arc<str>>,
    target: Target,
    hooks: ProviderHooks,
    /// Shared with every other entry this slug ever had: an entry is replaced
    /// on reload, its credentials are not. Held here rather than looked up
    /// beside the entry, so "registered" and "has credentials" cannot come
    /// apart at a call site.
    auth: Arc<AuthState>,
}

type Registry = HashMap<Box<str>, Arc<PluginEntry>>;

/// What every read answers. Only ever replaced whole, by [`commit_load`], so
/// there is no moment at which a reader can see a half-built registry: during
/// a load the previous generation is still the published one.
static PROVIDERS: LazyLock<RwLock<Arc<Registry>>> = LazyLock::new(RwLock::default);
/// The load in progress. `Some` between [`begin_load`] and [`commit_load`],
/// and open at process start so a host that loads plugins without driving the
/// phases (tests, embedders) still registers.
static STAGING: LazyLock<Mutex<Option<Registry>>> =
    LazyLock::new(|| Mutex::new(Some(Registry::new())));
/// Append-only for the life of the process: see [`shared_auth`].
static AUTH: LazyLock<RwLock<HashMap<Box<str>, Arc<AuthState>>>> = LazyLock::new(RwLock::default);

/// Every declaration maki itself authors, staged by [`begin_load`] before any
/// plugin gets to speak. Written out rather than collected with `inventory`
/// for the reason [`crate::spec::ProviderRegistry::builtins`] gives, and
/// because one list read top to bottom is how anyone tells which providers are
/// ported. Each port appends its constructor here, and this list, the seeding
/// in [`begin_load`] and the precedence in [`wins`] all go away with the last
/// provider that still needs them.
const RUST_DECLS: &[fn() -> Registration] = &[
    || Registration {
        decl: synthetic::decl(),
        hooks: ProviderHooks::default(),
    },
    || Registration {
        decl: deepseek::decl(),
        hooks: deepseek::hooks(),
    },
    || Registration {
        decl: mistral::decl(),
        hooks: mistral::hooks(),
    },
    || Registration {
        decl: tensorx::decl(),
        hooks: tensorx::hooks(),
    },
    || Registration {
        decl: regolo::decl(),
        hooks: regolo::hooks(),
    },
    || Registration {
        decl: requesty::decl(),
        hooks: requesty::hooks(),
    },
    || Registration {
        decl: openrouter::decl(),
        hooks: openrouter::hooks(),
    },
];

/// Opens the registration window, at the top of every plugin load.
///
/// Nothing published goes away here. Dropping the old entries before the
/// replacements exist would make every reader answer "unknown provider" for as
/// long as the load takes, so [`commit_load`] swaps the whole map in one step
/// instead.
///
/// The Rust-authored decls are staged first, so a Lua decl that never arrives,
/// or arrives broken, leaves the shipped one standing.
pub fn begin_load() {
    *STAGING.lock().unwrap() = Some(Registry::new());
    RUST_DECLS.iter().for_each(|declare| stage_rust(declare()));
}

fn stage_rust(reg: Registration) {
    let slug = reg.decl.slug.clone();
    if let Err(error) = register(reg, DeclSource::Rust, DeclAuthority::Bundled) {
        warn!(slug, %error, "built-in provider declaration was rejected");
    }
}

/// Publishes what this load registered. A load that registered nothing
/// publishes nothing, which is how a `/reload` drops a plugin that was removed.
///
/// Credentials are re-pointed here and not at registration, so they only ever
/// follow a declaration that is actually going to serve: one a failed plugin
/// staged and [`discard`] took back never touched them.
pub fn commit_load() {
    let Some(staged) = STAGING.lock().unwrap().take() else {
        return;
    };
    for entry in staged.values() {
        let slug = &entry.decl.slug;
        let api_key_env = api_key_env(&entry.decl, entry.claimed);
        if let Err(error) = entry
            .auth
            .redeclare(slug, api_key_env, &entry.decl.net_hosts)
        {
            warn!(slug, %error, "provider credentials were not re-declared");
        }
    }
    *PROVIDERS.write().unwrap() = Arc::new(staged);
}

/// Takes back everything `plugin` staged in the load in progress, for a plugin
/// whose load failed after it registered. Whatever it displaced is staged
/// again, so the slug ends up where it would be had the plugin never loaded.
pub fn discard(plugin: &str) {
    let dropped: Vec<Box<str>> = {
        let mut staging = STAGING.lock().unwrap();
        let Some(staged) = staging.as_mut() else {
            return;
        };
        let dropped: Vec<Box<str>> = staged
            .iter()
            .filter(|(_, entry)| entry.owner.as_deref() == Some(plugin))
            .map(|(slug, _)| slug.clone())
            .collect();
        for slug in &dropped {
            staged.remove(slug);
        }
        dropped
    };
    if !dropped.is_empty() {
        debug!(plugin, slugs = ?dropped, "discarded the declarations of a failed plugin load");
    }
    RUST_DECLS
        .iter()
        .map(|declare| declare())
        .filter(|reg| dropped.iter().any(|slug| **slug == reg.decl.slug))
        .for_each(stage_rust);
}

/// Which declaration serves a slug both a Rust and a Lua author declared:
/// `true` when `challenger` takes the slug off `incumbent`.
///
/// The only axis is who wrote the declaration, since both drive the same
/// [`codec::build`] call. Lua wins by default, which keeps everyday use on the
/// path plugin authors write against, and `impl = "rust"` in `providers.toml`
/// pins the shipped one for a user the Lua decl misbehaves for.
///
/// Two decls from the *same* source never reach here: that is
/// [`RegisterError::DuplicateSlug`], and stays one.
fn wins(incumbent: DeclSource, challenger: DeclSource, configured: Option<ImplChoice>) -> bool {
    let preferred = match configured {
        Some(ImplChoice::Rust) => DeclSource::Rust,
        Some(ImplChoice::Lua) | None => DeclSource::Lua,
    };
    challenger == preferred && incumbent != preferred
}

/// Who holds this slug in the load in progress. Asked of the staged map and
/// not of what is published, because a reload re-registers every slug it
/// registered last time and [`begin_load`] has already staged the Rust decls.
fn staged_source(slug: &str) -> Result<Option<DeclSource>, RegisterError> {
    let staging = STAGING.lock().unwrap();
    let staged = staging
        .as_ref()
        .ok_or_else(|| RegisterError::Closed(slug.to_string()))?;
    Ok(staged.get(slug).map(|entry| entry.source))
}

/// Registers a declaration, or leaves the slug to the one that beat it.
///
/// A declaration rejected while another author's is standing costs a warning
/// and not a provider: the incumbent keeps serving the slug. Safe only because
/// both declarations drive the same code path, and loud because a silent swap
/// leaves "why is my provider behaving oddly" with nothing to go on. The error
/// still goes back to the caller, so a plugin that failed is reported failed.
pub fn register(
    reg: Registration,
    source: DeclSource,
    authority: DeclAuthority,
) -> Result<Registered, RegisterError> {
    register_owned(reg, source, authority, None)
}

/// [`register`] for a Lua plugin, remembered as `plugin`'s so a load of it
/// that fails later can be taken back with [`discard`].
pub fn register_plugin(
    plugin: Arc<str>,
    reg: Registration,
    authority: DeclAuthority,
) -> Result<Registered, RegisterError> {
    register_owned(reg, DeclSource::Lua, authority, Some(plugin))
}

fn register_owned(
    reg: Registration,
    source: DeclSource,
    authority: DeclAuthority,
    owner: Option<Arc<str>>,
) -> Result<Registered, RegisterError> {
    let slug = reg.decl.slug.clone();
    let result = register_decl(reg, source, authority, owner);
    if let Err(error) = &result
        && let Ok(Some(standing)) = staged_source(&slug)
        && standing != source
    {
        warn!(
            slug,
            ?source,
            ?standing,
            %error,
            "declaration rejected, the standing declaration keeps this slug"
        );
    }
    result
}

fn register_decl(
    reg: Registration,
    source: DeclSource,
    authority: DeclAuthority,
    owner: Option<Arc<str>>,
) -> Result<Registered, RegisterError> {
    let Registration { decl, hooks } = reg;
    let slug = decl.slug.clone();
    if !is_valid_slug(&slug) {
        return Err(RegisterError::InvalidSlug(slug));
    }
    // Asked before the declaration is examined at all: one that loses is never
    // built, so a defect in it is not worth a word, and an explicit `impl`
    // choice cannot be turned into a rejection by a check the winner would
    // never have reached.
    let config = ProvidersConfig::load();
    let config_entry = config.get(&slug);
    let configured = config_entry.and_then(|def| def.r#impl);
    if let Some(incumbent) = staged_source(&slug)? {
        if incumbent == source {
            return Err(RegisterError::DuplicateSlug(slug));
        }
        if !wins(incumbent, source, configured) {
            debug!(
                slug,
                ?incumbent,
                rejected = ?source,
                ?configured,
                "kept the standing provider declaration"
            );
            return Ok(Registered::Shadowed);
        }
    }
    // maki's own declarations may claim a built-in slug, because a declaration
    // is how a built-in provider is written now and the spec row it claims
    // stays the single source of truth for everything [`inherited`] lists.
    //
    // For anyone else the slug stays reserved, and the rejection is here
    // rather than a silent demotion to a non-claim: a third party that took
    // `anthropic` without inheriting the row would still have taken
    // `anthropic`, which is the part that matters. See [`DeclAuthority`]. A
    // slug maki has no row for is nobody's to claim either, there would be
    // nothing to inherit and the name would collide with a future built-in.
    let claimed = match (ProviderRegistry::get(&slug), authority) {
        (Some(spec), DeclAuthority::Bundled) => Some(spec),
        (Some(_), DeclAuthority::ThirdParty) => return Err(RegisterError::ReservedSlug(slug)),
        (None, _) => None,
    };
    // Only a slug maki does not already own can be lost to `providers.toml`.
    // An entry under a built-in slug has never defined a provider: the built-in
    // keeps the slug and the entry overlays what
    // [`maki_config::providers::ignored_builtin_fields`] does not name, which
    // is how `[deepseek] base_url` points the shipped provider at a gateway.
    // Rejecting the claim over one would delete the provider the overlay was
    // written for, and would delete it for the Rust-authored declaration too.
    if claimed.is_none() && config_entry.is_some_and(defines_provider) {
        return Err(RegisterError::ConfiguredSlug(slug));
    }
    if decl.net_hosts.is_empty() {
        return Err(RegisterError::NoNetHosts(slug));
    }
    inherited(&decl, claimed)?;
    // Vetted here and then dropped: a declared origin is the codec's last
    // resort (see [`CodecOptions::base_url`]), not a credential, so it is the
    // one thing checked at registration and read back at `create`.
    declared_base_url(&slug, decl.base_url.clone(), &decl.net_hosts)
        .map_err(RegisterError::UndeclaredBaseUrl)?;
    let target = target_of(&decl, &hooks)?;
    // Checked now and applied at [`commit_load`], so a bad `[<slug>.headers]`
    // still fails the plugin that registered the slug. A built-in's fails only
    // its own [`create`], as it did natively: failing the bundled load would
    // keep maki from starting over a provider the user may never pick.
    if claimed.is_none() {
        ResolvedAuth::new(&slug, Vec::new()).map_err(|e| RegisterError::Credentials {
            slug: slug.clone(),
            message: e.to_string(),
        })?;
    }

    let mut staging = STAGING.lock().unwrap();
    let Some(staged) = staging.as_mut() else {
        return Err(RegisterError::Closed(slug));
    };
    let auth = shared_auth(&slug);
    debug!(
        slug,
        ?source,
        ?authority,
        claims_builtin = claimed.is_some(),
        codec = ?decl.codec,
        base = decl.base.as_deref(),
        models = decl.models.len(),
        hooked_credentials = hooks.auth.is_some(),
        "registered provider declaration"
    );
    staged.insert(
        slug.as_str().into(),
        Arc::new(PluginEntry {
            decl,
            claimed,
            source,
            owner,
            target,
            hooks,
            auth,
        }),
    );
    Ok(Registered::Serving)
}

/// What this declaration builds its requests with, once every option it stated
/// has been checked against what that target can honour. An option the target
/// would quietly drop is a rejection, for the reason [`honours_build_body`]
/// gives.
fn target_of(decl: &ProviderDecl, hooks: &ProviderHooks) -> Result<Target, RegisterError> {
    let target = match (decl.codec, &decl.base) {
        (Some(protocol), None) => Target::Codec(protocol),
        (None, Some(base)) => Target::Base(
            ProviderRegistry::get(base)
                .filter(|spec| spec.is_native())
                .ok_or_else(|| RegisterError::UnknownBase {
                    slug: decl.slug.clone(),
                    base: base.clone(),
                })?,
        ),
        _ => return Err(RegisterError::CodecOrBase(decl.slug.clone())),
    };
    let unsupported = |option| RegisterError::Unsupported {
        slug: decl.slug.clone(),
        option,
        target: target.describe(),
    };
    if hooks.build_body.is_some() && !honours_build_body(target) {
        return Err(unsupported(BUILD_BODY_OPTION));
    }
    if decl.system_prefix.is_some() && !honours_system_prefix(target) {
        return Err(unsupported(SYSTEM_PREFIX_OPTION));
    }
    if decl.openai.is_some() && !honours_openai_wire(target) {
        return Err(unsupported(OPENAI_OPTION));
    }
    Ok(target)
}

/// One cell of credentials per slug, for the life of the process. Reusing it
/// is what carries a token across a reload, and what stops a reload from
/// minting a second [`RefreshGate`] for a slug whose token is in flight.
fn shared_auth(slug: &str) -> Arc<AuthState> {
    let mut states = AUTH.write().unwrap();
    if let Some(state) = states.get(slug) {
        return Arc::clone(state);
    }
    let state = Arc::new(AuthState::new());
    states.insert(slug.into(), Arc::clone(&state));
    state
}

/// Whether a `providers.toml` entry defines a provider of its own, as opposed
/// to only naming which of two declarations serves the slug. An `impl` line is
/// there to choose between them, so counting it as a definition would reject
/// both and leave the slug with nothing.
///
/// Asked of the serialised form rather than field by field: every field of
/// [`ProviderDef`] is skipped when unset, so one added later counts without
/// being listed here.
fn defines_provider(def: &ProviderDef) -> bool {
    let Ok(Value::Object(fields)) = serde_json::to_value(def) else {
        return true;
    };
    fields.keys().any(|field| field != IMPL_FIELD)
}

/// The env var this declaration's key comes out of: its own, or the claimed
/// row's. Empty means the provider has no key env at all (Ollama's host,
/// Aperture's gateway), which is the same as declaring none.
fn api_key_env(decl: &ProviderDecl, claimed: Option<&'static ProviderSpec>) -> Option<String> {
    decl.api_key_env
        .as_deref()
        .or(claimed.map(|spec| spec.api_key_env))
        .filter(|env_var| !env_var.is_empty())
        .map(str::to_owned)
}

/// What a claim may not restate, and what a non-claim may not leave out.
///
/// Restating an inherited field is refused rather than ignored, for the reason
/// [`honours_build_body`] gives: an option that cannot be honoured is a
/// registration error, never a no-op. Two homes for `display_name` is exactly
/// what a silent override would bring back.
fn inherited(
    decl: &ProviderDecl,
    claimed: Option<&'static ProviderSpec>,
) -> Result<(), RegisterError> {
    if claimed.is_none() {
        return match decl.display_name {
            Some(_) => Ok(()),
            None => Err(RegisterError::NoDisplayName(decl.slug.clone())),
        };
    }
    let restated = [
        (DISPLAY_NAME_FIELD, decl.display_name.is_some()),
        (API_KEY_ENV_FIELD, decl.api_key_env.is_some()),
        (MODELS_FIELD, !decl.models.is_empty()),
    ];
    match restated.into_iter().find(|(_, stated)| *stated) {
        Some((field, _)) => Err(RegisterError::Restated {
            slug: decl.slug.clone(),
            field,
        }),
        None => Ok(()),
    }
}

fn entries() -> Arc<Registry> {
    Arc::clone(&PROVIDERS.read().unwrap())
}

fn entry(slug: &str) -> Option<Arc<PluginEntry>> {
    entries().get(slug).cloned()
}

fn unknown(slug: &str) -> AgentError {
    AgentError::Config {
        message: format!("unknown plugin provider '{slug}'"),
    }
}

/// What a declaration's `api_key_env` resolved to. A decl that names one
/// answers for its key up front, the way every built-in does, so availability
/// follows from the declaration rather than from who wrote it.
#[derive(Clone)]
enum DeclaredKeys {
    /// No `api_key_env`: the credentials come from an auth hook, which cannot
    /// run on a synchronous path, so nothing is knowable until one does.
    Hooked,
    Resolved {
        env_var: String,
        pool: KeyPool,
    },
    /// [`KeyPool::resolve`] failed. The message is kept rather than the error
    /// because [`AgentError`] is not `Clone`, and `create` has to answer with
    /// it verbatim every time so the picker hides the provider instead of
    /// listing one that cannot serve a request.
    Missing {
        env_var: String,
        message: String,
    },
}

impl DeclaredKeys {
    /// The same call every built-in makes, so env / saved-credential /
    /// `providers.toml` precedence and the "run `maki auth login`" message are
    /// identical by construction rather than by copying.
    fn resolve(slug: &str, env_var: Option<String>) -> Self {
        let Some(env_var) = env_var else {
            return Self::Hooked;
        };
        match KeyPool::resolve(slug, &env_var) {
            Ok(pool) => Self::Resolved { env_var, pool },
            Err(e) => Self::Missing {
                env_var,
                message: e.to_string(),
            },
        }
    }

    fn env_var(&self) -> Option<&str> {
        match self {
            Self::Hooked => None,
            Self::Resolved { env_var, .. } | Self::Missing { env_var, .. } => Some(env_var),
        }
    }

    /// Resolves `env_var` afresh and adopts the answer only if it says
    /// something new: another variable, other keys, or a key appearing or
    /// going. The same keys keep the pool they had, so re-reading never walks a
    /// rotating pool back to its first key. Answers whether anything moved.
    fn refresh(&mut self, slug: &str, env_var: Option<String>) -> bool {
        let fresh = Self::resolve(slug, env_var);
        let unchanged = match (&*self, &fresh) {
            (Self::Hooked, Self::Hooked) => true,
            (
                Self::Resolved { env_var, pool },
                Self::Resolved {
                    env_var: read,
                    pool: read_pool,
                },
            ) => env_var == read && pool.same_keys(read_pool),
            (
                Self::Missing { env_var, message },
                Self::Missing {
                    env_var: read,
                    message: read_message,
                },
            ) => env_var == read && message == read_message,
            _ => false,
        };
        if !unchanged {
            *self = fresh;
        }
        !unchanged
    }

    /// The credentials a provider starts with, before any hook has run.
    fn initial_auth(&self, slug: &str) -> Result<ResolvedAuth, RegisterError> {
        let auth = match self {
            Self::Resolved { pool, .. } => ResolvedAuth::bearer(slug, pool.current()),
            Self::Hooked | Self::Missing { .. } => ResolvedAuth::new(slug, Vec::new()),
        };
        // `ResolvedAuth` only fails over `[<slug>.headers]` in providers.toml,
        // which a declaration claiming a built-in slug is allowed to have.
        auth.map_err(|e| RegisterError::Credentials {
            slug: slug.to_string(),
            message: e.to_string(),
        })
    }
}

/// Auth for one slug, kept in its own map so a reload cannot drop a token or a
/// refresh in flight. The codec reads this cell per request, so whatever the
/// hook last wrote is what goes on the wire, with nothing rebuilt or copied
/// back.
struct AuthState {
    current: Arc<Mutex<ResolvedAuth>>,
    /// The latest registration's egress list, not the one the entry a caller
    /// happens to hold was built with. A refresh that started before a reload
    /// still lands its answer here, so vetting it against anything older would
    /// admit an origin the current declaration no longer covers.
    hosts: Mutex<Arc<[String]>>,
    keys: Mutex<DeclaredKeys>,
    gate: RefreshGate,
}

impl AuthState {
    /// Starts with no credentials and no declared hosts. [`Self::redeclare`]
    /// runs before anything can read this, so starting empty fails closed if
    /// it ever did not, or if the declaration's `[<slug>.headers]` did not
    /// resolve (which [`create`] then reports).
    fn new() -> Self {
        Self {
            current: Arc::new(Mutex::new(ResolvedAuth::withheld())),
            hosts: Mutex::default(),
            keys: Mutex::new(DeclaredKeys::Hooked),
            gate: RefreshGate::default(),
        }
    }

    fn hosts(&self) -> Arc<[String]> {
        Arc::clone(&self.hosts.lock().unwrap())
    }

    /// The pool `create` hands the provider for rotation, or the error the
    /// declared env var resolved to.
    ///
    /// Re-read on every call, the way a built-in rebuilt per `create` re-reads
    /// its key: one that appeared or was replaced since, say by a `maki auth
    /// login` after the old key was revoked, is what the next provider sends.
    fn declared_pool(&self, slug: &str) -> Result<Option<KeyPool>, AgentError> {
        let mut keys = self.keys.lock().unwrap();
        let env_var = keys.env_var().map(str::to_owned);
        // Only over credentials nothing has minted: a token a hook produced is
        // newer than anything an env var can offer.
        if keys.refresh(slug, env_var)
            && !self.gate.ran()
            && let Ok(declared) = keys.initial_auth(slug)
        {
            *self.current.lock().unwrap() = declared;
        }
        match &*keys {
            DeclaredKeys::Hooked => Ok(None),
            DeclaredKeys::Resolved { pool, .. } => Ok(Some(pool.clone())),
            DeclaredKeys::Missing { env_var, message } => {
                debug!(slug, env_var, "declared api key did not resolve");
                Err(AgentError::Config {
                    message: message.clone(),
                })
            }
        }
    }

    /// Re-points one slug's credentials at what the newest registration says.
    ///
    /// Three things move on a reload and none may be ignored: the declared
    /// hosts, which every later hook answer is vetted against, the declared key
    /// env var, and the static credentials, which are the only ones a plugin
    /// without an auth hook ever has. A token a hook already minted stays,
    /// unless the new declaration stopped covering the origin it points at.
    fn redeclare(
        &self,
        slug: &str,
        api_key_env: Option<String>,
        hosts: &[String],
    ) -> Result<(), RegisterError> {
        *self.hosts.lock().unwrap() = hosts.into();
        let mut keys = self.keys.lock().unwrap();
        keys.refresh(slug, api_key_env);
        let declared = keys.initial_auth(slug)?;
        let mut current = self.current.lock().unwrap();
        let undeclared = declared_base_url(slug, current.base_url.clone(), hosts).is_err();
        if !self.gate.ran() || undeclared {
            *current = declared;
        }
        Ok(())
    }

    /// [`Self::mint`] under the cross-process lock: serialised against other
    /// maki processes on the same credentials, and re-entrant in this one so
    /// the hook can store what it minted.
    async fn locked_mint(
        &self,
        slug: &str,
        hook: &dyn Hook<AuthPurpose, PluginAuth>,
        purpose: AuthPurpose,
    ) -> Result<(), AgentError> {
        let owned = slug.to_owned();
        let _lock = smol::unblock(move || {
            StateDir::resolve()
                .ok()
                .map(|dir| lock_credentials(&dir, &owned))
        })
        .await;
        self.mint(slug, hook, purpose).await
    }

    /// Runs the auth hook and makes what it answered the credentials every
    /// request to the slug carries.
    async fn mint(
        &self,
        slug: &str,
        hook: &dyn Hook<AuthPurpose, PluginAuth>,
        purpose: AuthPurpose,
    ) -> Result<(), AgentError> {
        let mut fresh = hook
            .call(purpose)
            .await?
            .into_resolved(slug, &self.hosts())?;
        let mut current = self.current.lock().unwrap();
        // A hook that omits base_url keeps the resolved one. Falling back to
        // the provider's default origin would silently repoint the token.
        if fresh.base_url.is_none() {
            fresh.base_url = current.base_url.take();
        }
        *current = fresh;
        Ok(())
    }
}

impl PluginEntry {
    /// The spec row this entry answers inherited questions from: the built-in
    /// it claimed, else the native spec behind its target. Model family,
    /// fallbacks and the curated table a declaration with none of its own
    /// borrows all come from here, so no caller has to ask what kind of entry
    /// it is holding.
    fn spec(&self) -> Option<&'static ProviderSpec> {
        self.claimed.or_else(|| self.target.spec())
    }

    fn display_name(&self) -> &str {
        self.decl
            .display_name
            .as_deref()
            .or(self.claimed.map(|spec| spec.display_name))
            // Registration refuses a declaration with neither, so a real entry
            // never reaches this arm.
            .unwrap_or(&self.decl.slug)
    }

    /// Resolve once, lazily, from async code. Every fallible provider method
    /// starts here, so no synchronous path ever has to reach a hook.
    async fn ensure_auth(&self) -> Result<(), AgentError> {
        if self.auth.gate.ran() {
            return Ok(());
        }
        self.run_auth(AuthPurpose::Resolve).await.map(drop)
    }

    /// The only place the auth hook is called. A plugin without one keeps the
    /// credentials its registration declared.
    ///
    /// Answers whether credentials were actually minted, which is a different
    /// question from whether the call failed: with no auth hook this succeeds
    /// having changed nothing, and a caller that replayed a request on that
    /// would only re-send the credentials it already had.
    ///
    /// Callable from the plugin host's own thread, because the hook goes to the
    /// host's priority lane and the dispatch loop serves it while this future
    /// is parked. What may not reach here is a *blocking* caller, and that is
    /// held by construction rather than by a check: every path in is `async`,
    /// and `create` builds a provider without running a hook at all.
    async fn run_auth(&self, purpose: AuthPurpose) -> Result<bool, AgentError> {
        let Some(hook) = self.hooks.auth.clone() else {
            return Ok(false);
        };
        let slug = self.decl.slug.clone();
        // A reload re-reads what a login wrote, so it spends no token. It
        // waits for neither the gate nor the cross-process lock, and it runs
        // under `block_on` on the ui thread, where either wait would freeze the
        // ui.
        if matches!(purpose, AuthPurpose::Reload) {
            self.auth.mint(&slug, hook.as_ref(), purpose).await?;
            return Ok(true);
        }
        // Everything else spends a single-use token, so it runs to completion
        // on its own task. A caller that goes away mid-flight, an Esc or a turn
        // that timed out, must neither release the gate and the lock while the
        // hook still holds the token nor drop what the hook minted with it.
        let auth = Arc::clone(&self.auth);
        let (done, outcome) = flume::bounded(1);
        smol::spawn(async move {
            let minted = auth
                .gate
                .single_flight(auth.locked_mint(&slug, hook.as_ref(), purpose))
                .await;
            let _ = done.send(minted);
        })
        .detach();
        outcome
            .recv_async()
            .await
            .map_err(|_| AgentError::Channel)??;
        Ok(true)
    }
}

/// Single-flight around the plugin's auth hook. Every `create` mints a fresh
/// `PluginProvider`, so sub-agents running their own model would each spend the
/// plugin's rotating refresh token, and a spent one taken twice costs the whole
/// token family. They queue here instead, and the late arrival returns to find
/// the shared credentials already holding what the winner minted. That is why
/// the gate hangs off the per-slug auth state rather than the provider.
#[derive(Default)]
struct RefreshGate {
    lock: smol::lock::Mutex<()>,
    /// Counted, not compared: a refresh can hand back byte-identical
    /// credentials, so the count is the only thing that can tell a parked
    /// caller the work is already done. Written only under `lock`.
    runs: AtomicU64,
}

impl RefreshGate {
    /// Whether the hook has ever run to completion, which is also what makes
    /// the lazy first resolve happen once.
    fn ran(&self) -> bool {
        self.runs.load(Ordering::Acquire) > 0
    }

    async fn single_flight(
        &self,
        work: impl Future<Output = Result<(), AgentError>>,
    ) -> Result<(), AgentError> {
        let before = self.runs.load(Ordering::Acquire);
        let _guard = self.lock.lock().await;
        if self.runs.load(Ordering::Acquire) != before {
            debug!("peer refreshed while we waited, skipping auth hook");
            return Ok(());
        }
        work.await?;
        self.runs.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

struct BodyAdapter(Arc<dyn Hook<BodyInput, Value>>);

impl BodyHook for BodyAdapter {
    fn call<'a>(
        &'a self,
        body: Value,
        ctx: &RequestCtx<'_>,
    ) -> BoxFuture<'a, Result<Value, AgentError>> {
        self.0.call(BodyInput {
            body,
            model: ctx.model.id.clone(),
            thinking: ctx.opts.thinking.to_string(),
            model_info: ctx.discovered.as_ref().and_then(|info| info.extra.clone()),
        })
    }
}

struct PluginProvider {
    entry: Arc<PluginEntry>,
    /// Cloned out of the auth state at `create`, so rotation shares the index
    /// with every other provider built for this slug. `None` for a declaration
    /// whose credentials come from a hook: there is no pool to walk.
    pool: Option<KeyPool>,
    inner: Box<dyn Provider>,
}

impl PluginProvider {
    /// The single place `map_error` is applied, so it cannot cover streaming
    /// and miss the rest. The hook may restate the status and the message and
    /// nothing else: `retry_after` is what the server actually asked for, and
    /// retryability is derived from the status by `retry_kind`.
    fn mapped<'a, T: Send + 'a>(
        &'a self,
        result: Result<T, AgentError>,
    ) -> BoxFuture<'a, Result<T, AgentError>> {
        Box::pin(async move {
            let Some(hook) = &self.entry.hooks.map_error else {
                return result;
            };
            let Err(AgentError::Api {
                status,
                message,
                retry_after,
            }) = result
            else {
                return result;
            };
            let original = ApiError { status, message };
            let replacement = match hook.call(original.clone()).await {
                Ok(mapped) => mapped,
                Err(e) => {
                    warn!(error = %e, "map_error hook failed, keeping the original error");
                    None
                }
            };
            let ApiError { status, message } = replacement.unwrap_or(original);
            Err(AgentError::Api {
                status,
                message,
                retry_after,
            })
        })
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, AgentError> {
        self.entry.ensure_auth().await?;
        if let Some(hook) = &self.entry.hooks.list_models {
            return hook.call(()).await;
        }
        // A declaration with no table of its own, a claim included, falls
        // through to the inner provider, which is built from the very codec
        // the built-in uses. So asking it is asking the built-in.
        if self.entry.decl.models.is_empty() {
            return self.inner.list_models().await;
        }
        Ok(self
            .entry
            .decl
            .models
            .iter()
            .map(PluginModel::to_info)
            .collect())
    }

    async fn usage(&self) -> Result<Option<ProviderUsage>, AgentError> {
        self.entry.ensure_auth().await?;
        match &self.entry.hooks.fetch_usage {
            Some(hook) => hook.call(()).await,
            None => self.inner.fetch_usage().await,
        }
    }
}

#[warn(clippy::missing_trait_methods)]
impl Provider for PluginProvider {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        opts: RequestOptions,
        session_id: Option<&'a SessionRef>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            let result = async {
                self.entry.ensure_auth().await?;

                // First attempt streams through a counting relay: a 401 is only
                // retried when it preceded every event. Replaying deltas onto a
                // channel that already delivered part of the answer would
                // duplicate text in the UI and, after a cancel, in the history.
                let (tx, rx) = flume::unbounded();
                let attempt = async {
                    let result = self
                        .inner
                        .stream_message(model, messages, system, tools, &tx, opts, session_id)
                        .await;
                    drop(tx);
                    result
                };
                let forward = async move {
                    let mut forwarded = false;
                    while let Ok(ev) = rx.recv_async().await {
                        forwarded = true;
                        if event_tx.send_async(ev).await.is_err() {
                            break;
                        }
                    }
                    forwarded
                };
                let (result, forwarded) = futures_lite::future::zip(attempt, forward).await;
                match result {
                    // The plugin mints credentials without the user, so an
                    // expired token costs one silent refresh instead of a
                    // re-login prompt. Only a refresh that minted something
                    // earns the replay: a declaration keyed off an
                    // `api_key_env` has no hook to run, so a second try would
                    // re-send the key the server just rejected.
                    Err(e) if e.is_auth_error() && !forwarded => {
                        debug!(error = %e, "auth error, refreshing plugin-backed credentials");
                        match self.entry.run_auth(AuthPurpose::Refresh).await {
                            Ok(true) => {
                                self.inner
                                    .stream_message(
                                        model, messages, system, tools, event_tx, opts, session_id,
                                    )
                                    .await
                            }
                            Ok(false) => Err(e),
                            Err(refresh_err) => {
                                warn!(error = %refresh_err, "silent refresh failed, falling back to re-login");
                                Err(e)
                            }
                        }
                    }
                    result => result,
                }
            }
            .await;
            self.mapped(result).await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let result = self.models().await;
            self.mapped(result).await
        })
    }

    fn fetch_usage(&self) -> BoxFuture<'_, Result<Option<ProviderUsage>, AgentError>> {
        Box::pin(async move {
            let result = self.usage().await;
            self.mapped(result).await
        })
    }

    fn refresh_auth(&self) -> BoxFuture<'_, Result<(), AgentError>> {
        Box::pin(async move {
            let result = self.entry.run_auth(AuthPurpose::Refresh).await.map(drop);
            self.mapped(result).await
        })
    }

    fn reload_auth(&self) -> BoxFuture<'_, Result<(), AgentError>> {
        Box::pin(async move {
            let result = self.entry.run_auth(AuthPurpose::Reload).await.map(drop);
            self.mapped(result).await
        })
    }

    /// Answering `None` here is not free: the retry loop rotates keys without
    /// spending its budget, so a provider that hides its pool turns every
    /// rotation into a retry.
    fn keys(&self) -> Option<KeyRotation<'_>> {
        Some(KeyRotation::new(
            self.pool.as_ref()?,
            &self.entry.auth.current,
            KeyHeader::Bearer,
        ))
    }

    /// Whatever the target knows about its models, a `base`'s own rules
    /// included, still applies under a declaration that wraps it.
    fn adjust_model(&self, model: &mut Model) {
        self.inner.adjust_model(model);
    }
}

/// Builds from registry data alone: no hook runs here, because this is called
/// from synchronous code and a hook means re-entering the plugin host.
///
/// A declaration that names an `api_key_env` and resolved no key fails here
/// rather than at the first request, which is what makes `provider_available`
/// tell the truth about it and the picker hide it. A declaration without one
/// gets its credentials from a hook and cannot know yet, so it stays lazy. The
/// rule keys off the declared field, never off who wrote the declaration.
pub fn create(slug: &str, timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    let entry = entry(slug).ok_or_else(|| unknown(slug))?;
    // Re-resolved per build, as a native constructor did, so a bad
    // `[<slug>.headers]` a built-in's registration let through fails here.
    ResolvedAuth::new(slug, Vec::new())?;
    let pool = entry.auth.declared_pool(slug)?;
    // The same handle the codec reads per request, so credentials resolved by a
    // later `ensure_auth` land without rebuilding anything.
    let shared = entry.auth.current.clone();
    let prefix = entry.decl.system_prefix.clone();

    let inner = match entry.target {
        Target::Base(spec) => {
            let native = spec.native.ok_or_else(|| AgentError::Config {
                message: format!("base provider '{}' has no constructor", spec.slug),
            })?;
            (native.with_auth)(shared, timeouts, prefix)
        }
        Target::Codec(protocol) => codec::build(codec_options(&entry, protocol), shared, timeouts),
    };

    debug!(slug, source = ?entry.source, rotating_keys = pool.is_some(), "built plugin provider");
    Ok(Box::new(PluginProvider { entry, pool, inner }))
}

/// The declaration, as the codec takes it.
///
/// The declared origin lands in `base_url` and not in the auth cell on
/// purpose: `auth.base_url` outranks the user's `<SLUG>_BASE_URL`, which a
/// static declaration must not. `base_url` is read back against env and
/// `providers.toml` inside the compat layer, where nothing here can write it,
/// and `auth.base_url` is only ever written through
/// [`PluginAuth::into_resolved`], the one door that vets an origin against the
/// declared `net_hosts`.
fn codec_options(entry: &PluginEntry, protocol: Protocol) -> CodecOptions {
    let decl = &entry.decl;
    CodecOptions {
        api_key_env: decl.api_key_env.clone().unwrap_or_default().into(),
        base_url: decl.base_url.clone().unwrap_or_default().into(),
        provider_name: entry.display_name().to_owned().into(),
        system_prefix: decl.system_prefix.clone(),
        openai: decl.openai.clone().unwrap_or_default(),
        build_body: entry
            .hooks
            .build_body
            .clone()
            .map(|hook| Arc::new(BodyAdapter(hook)) as Arc<dyn BodyHook>),
        ..CodecOptions::new(protocol, decl.slug.clone())
    }
}

/// Owned, not `&'static`: unlike a script's metadata, a plugin entry can be
/// replaced by a reload while a caller holds the name.
pub fn display_name(slug: &str) -> Option<String> {
    entry(slug).map(|entry| entry.display_name().to_owned())
}

/// The credentials a registered slug currently holds, for a hook that has to
/// reach an endpoint the codec knows nothing about. A snapshot, like the one
/// every codec takes per request, so a refresh landing mid-call cannot swap the
/// headers a request is already building.
pub fn resolved_auth(slug: &str) -> Option<ResolvedAuth> {
    Some(entry(slug)?.auth.current.lock().unwrap().clone())
}

/// [`resolved_auth`] for a built-in's Rust hook. Hooks are handed no
/// credentials, so one that fetches off the codec's request path reads the
/// ones its registration resolved.
pub(crate) fn registered_auth(slug: &str) -> Result<ResolvedAuth, AgentError> {
    resolved_auth(slug).ok_or_else(|| AgentError::Config {
        message: format!("{slug} is not registered"),
    })
}

/// The origin a request to `slug` would reach right now, resolved the way a
/// codec resolves it per request: an auth-supplied origin, then the user's
/// `<SLUG>_BASE_URL` or `providers.toml`, then the declared default.
///
/// A hook reaching an endpoint off the codec's request path has to resolve the
/// same origin the codec would, or a user who points the slug at a gateway has
/// that one call go somewhere else.
pub fn effective_base_url(slug: &str) -> Option<String> {
    resolve_base_url(entry(slug)?.as_ref()).map(|(base_url, _)| base_url)
}

/// The origin of [`effective_base_url`] when the user or maki picked it, never
/// one a third-party plugin authored.
///
/// The codec reaches this origin over the Rust client, which knows no
/// private-address guard, so a provider's own side calls have to reach it on
/// the same terms: a user pointing a slug at a LAN gateway, or sitting behind
/// a fake-IP proxy, gets a listing and not only a chat.
pub fn vouched_origin(slug: &str) -> Option<Url> {
    let entry = entry(slug)?;
    let (base_url, chosen_by) = resolve_base_url(&entry)?;
    if chosen_by == OriginChooser::Plugin && entry.claimed.is_none() {
        return None;
    }
    Url::parse(&base_url).ok()
}

#[derive(PartialEq, Eq)]
enum OriginChooser {
    User,
    Plugin,
}

fn resolve_base_url(entry: &PluginEntry) -> Option<(String, OriginChooser)> {
    if let Some(explicit) = entry.auth.current.lock().unwrap().base_url.clone() {
        return Some((explicit, OriginChooser::Plugin));
    }
    let slug = entry.decl.slug.as_str();
    if let Some(configured) = configured_base_url(slug, ProvidersConfig::load().get(slug)) {
        return Some((configured, OriginChooser::User));
    }
    Some((entry.decl.base_url.clone()?, OriginChooser::Plugin))
}

/// The host of [`effective_base_url`], for a caller deciding whether an
/// outbound request is one this provider is already making. Every origin that
/// can win there was either vetted at registration (the declaration's own, a
/// hook's) or stated by the user (`<SLUG>_BASE_URL`, `providers.toml`).
pub fn effective_host(slug: &str) -> Option<String> {
    let base_url = effective_base_url(slug)?;
    Some(Url::parse(&base_url).ok()?.host_str()?.to_owned())
}

pub fn base_for_slug(slug: &str) -> Option<&'static ProviderSpec> {
    entry(slug)?.spec()
}

/// The declaration maki itself authors for `slug`, which is not always the one
/// serving it: a Lua decl for the same slug outranks it unless `impl = "rust"`
/// says otherwise. Read straight off [`RUST_DECLS`], so it answers before any
/// load has happened.
pub fn rust_decl(slug: &str) -> Option<ProviderDecl> {
    RUST_DECLS
        .iter()
        .map(|declare| declare().decl)
        .find(|decl| decl.slug == slug)
}

/// The declaration serving `slug` right now, whoever authored it.
pub fn registered_decl(slug: &str) -> Option<ProviderDecl> {
    Some(entry(slug)?.decl.clone())
}

/// A provider for `slug` built against auth the caller resolved, for a caller
/// that routes a request onto another provider's wire rather than owning the
/// credentials. `None` when no declaration drives a codec for the slug, which
/// leaves the caller its own fallback.
///
/// `system_prefix` is the routing caller's, and it outranks the declared one
/// for the same reason the native path takes it as an argument: it belongs to
/// the session, not to the provider.
pub fn build_with_auth(
    slug: &str,
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
    system_prefix: Option<String>,
) -> Option<Box<dyn Provider>> {
    let entry = entry(slug)?;
    let Target::Codec(protocol) = entry.target else {
        return None;
    };
    let mut options = codec_options(&entry, protocol);
    options.system_prefix = system_prefix.or(options.system_prefix);
    Some(codec::build(options, auth, timeouts))
}

pub fn lookup_model(slug: &str, model_id: &str) -> Option<Model> {
    let entry = entry(slug)?;
    let model = longest_prefix_match(&entry.decl.models, model_id)?;
    Some(model.to_model(slug, entry.spec()?, model_id.to_string()))
}

pub fn find_model_for_tier(slug: &str, tier: ModelTier) -> Option<Model> {
    let entry = entry(slug)?;
    let model = entry.decl.models.iter().find(|model| model.tier == tier)?;
    Some(model.to_model(slug, entry.spec()?, model.canonical_id()?.to_string()))
}

/// A declaration that curates no table of its own borrows the one behind its
/// spec row, the same table the built-in serves.
pub fn plugin_model_specs_for(slug: &str) -> Vec<String> {
    let Some(entry) = entry(slug) else {
        return Vec::new();
    };
    let ids: Vec<&str> = if entry.decl.models.is_empty() {
        entry
            .spec()
            .map(|spec| {
                spec.models()
                    .iter()
                    .flat_map(|row| row.prefixes.iter().copied())
                    .collect()
            })
            .unwrap_or_default()
    } else {
        entry
            .decl
            .models
            .iter()
            .filter_map(PluginModel::canonical_id)
            .collect()
    };
    ids.into_iter().map(|id| format!("{slug}/{id}")).collect()
}

/// The slugs only a declaration knows about. A claim is left out: it answers
/// for a built-in row, and whatever lists the built-ins already lists it, with
/// the built-in's rule for a provider that has no key.
pub fn unclaimed_slugs() -> Vec<String> {
    entries()
        .values()
        .filter(|entry| entry.claimed.is_none())
        .map(|entry| entry.decl.slug.clone())
        .collect()
}

pub fn auth_providers() -> Vec<(String, String)> {
    entries()
        .values()
        .filter(|entry| entry.hooks.login.is_some())
        .map(|entry| (entry.decl.slug.clone(), entry.display_name().to_owned()))
        .collect()
}

pub fn is_registered(slug: &str) -> bool {
    entries().contains_key(slug)
}

/// Blocks, so it belongs to the cli thread and never to the plugin host's: the
/// hook it drives runs on the host, and waiting for it from there deadlocks.
pub fn login(slug: &str) -> Result<(), AgentError> {
    interactive(slug, |hooks| hooks.login.clone(), "login")
}

pub fn logout(slug: &str) -> Result<(), AgentError> {
    interactive(slug, |hooks| hooks.logout.clone(), "logout")
}

/// The two hooks a person triggers rather than the model.
type InteractiveHook = Option<Arc<dyn Hook<(), ()>>>;

fn interactive(
    slug: &str,
    pick: fn(&ProviderHooks) -> InteractiveHook,
    what: &str,
) -> Result<(), AgentError> {
    let entry = entry(slug).ok_or_else(|| unknown(slug))?;
    let hook = pick(&entry.hooks).ok_or_else(|| AgentError::Config {
        message: format!("provider '{slug}' does not support {what} (uses API key)"),
    })?;
    smol::block_on(hook.call(()))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use futures_lite::future::zip;
    use maki_config::providers::base_url_env_var;
    use test_case::test_case;

    use super::*;
    use crate::retry::RetryKind;
    use crate::test_support::{Canned, serve};
    use crate::types::dialect;

    const AUTH_HEADER: &str = "authorization";
    const CONSTANT_TOKEN: &str = "Bearer constant";
    const FIRST_TOKEN: &str = "Bearer refreshed-1";
    const ROTATED_TOKEN: &str = "Bearer refreshed-2";
    const EXAMPLE_HOST: &str = "example.com";
    const EXAMPLE_BASE_URL: &str = "https://example.com";
    const OTHER_HOST: &str = "other.example";
    const OTHER_BASE_URL: &str = "https://other.example";
    const HOOK_CALLED: &str = "a hook ran on a synchronous path";
    const ONE_HOOK_CALL: &str = "concurrent callers share one auth hook run";
    const SINGLE_FLIGHT_RERUN: &str = "a refresh that overlaps nobody runs the hook again";
    const INNER_UNUSED: &str = "the inner provider is not exercised here";

    const DISPLAY_NAME: &str = "Plugin";
    const SOME_SYSTEM_PREFIX: &str = "You are X.";
    const MODEL_ID: &str = "plug-1";
    const MODEL_ALIAS: &str = "plug";
    /// Matches [`MODEL_ALIAS`] by prefix and [`MODEL_ID`] more tightly, which
    /// is what makes a lookup prove it took the longest match.
    const MODEL_VARIANT: &str = "plug-1-mini";
    const MODEL_TIER: ModelTier = ModelTier::Strong;
    const UNDECLARED_BASE_URL: &str = "https://evil.test/v1";
    /// A native built-in with a curated table, claimed with a codec that is
    /// deliberately not its own so inheritance cannot be mistaken for the
    /// target's fallback.
    const CLAIMED_SLUG: &str = super::super::anthropic::SLUG;
    const NO_CLAIMED_SPEC: &str = "the claimed slug must have a spec row";
    const CLAIM_READS_THE_CODEC_SPEC: &str =
        "a claim resolves through the row it claimed, not through its codec's";
    const CLAIM_LOST_THE_TABLE: &str = "a claim serves the curated table it inherited";
    const RELOAD_DROPS_STALE: &str = "a new load must not inherit the last load's entries";
    const RELOAD_KEEPS_SERVING: &str = "a load in progress must not unpublish what is serving";
    const WRONG_AUTHOR_SERVING: &str = "the wrong author's declaration is serving the slug";
    const PRECEDENCE_MISMATCH: &str = "the lua half of the load was not rejected as expected";
    const FIRST_LUA_DECL_REJECTED: &str = "the first lua decl must take the slug";
    const LEASE_LOST: &str = "the origin the auth hook leased must reach the auth cell";
    const RESERVED_SLUG_TAKEN: &str = "a refused declaration must not be serving the slug";
    const SHADOWED_SERVES: &str = "the decl that lost its slug reported itself as serving";

    fn decl(slug: &str) -> ProviderDecl {
        ProviderDecl {
            slug: slug.to_string(),
            display_name: Some(DISPLAY_NAME.to_string()),
            codec: Some(Protocol::Openai),
            base: None,
            base_url: Some(EXAMPLE_BASE_URL.to_string()),
            api_key_env: None,
            system_prefix: None,
            openai: None,
            models: vec![
                serde_json::from_value(serde_json::json!({
                    "prefixes": [MODEL_ID, MODEL_ALIAS],
                    "tier": MODEL_TIER,
                }))
                .unwrap(),
            ],
            net_hosts: vec![EXAMPLE_HOST.to_string()],
        }
    }

    fn registration(slug: &str) -> Registration {
        Registration {
            decl: decl(slug),
            hooks: ProviderHooks::default(),
        }
    }

    /// A test registers the way a plugin load does: open the window, register,
    /// publish. Entries from the previous load go, exactly as on a `/reload`.
    fn register_loaded(reg: Registration) -> Result<(), RegisterError> {
        register_loaded_as(reg, DeclAuthority::ThirdParty)
    }

    /// [`register_loaded`] for the declarations maki itself ships, which are
    /// the only ones that may take a built-in slug.
    fn register_loaded_as(
        reg: Registration,
        authority: DeclAuthority,
    ) -> Result<(), RegisterError> {
        begin_load();
        let result = register(reg, DeclSource::Lua, authority);
        commit_load();
        result.map(drop)
    }

    struct CountingAuth {
        calls: AtomicUsize,
        rotating: bool,
        /// The origin this hook leases. A hook is the only way an origin ever
        /// reaches the auth cell, a declared one being the codec's last resort.
        base_url: Option<String>,
    }

    impl CountingAuth {
        fn new(rotating: bool) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                rotating,
                base_url: None,
            }
        }

        fn leasing(base_url: &str) -> Self {
            Self {
                base_url: Some(base_url.to_string()),
                ..Self::new(false)
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    impl Hook<AuthPurpose, PluginAuth> for CountingAuth {
        /// Yields before counting, so a caller that reaches here is observably
        /// in flight when its peer is next polled. Without that the fence would
        /// only hold for as long as [`PluginEntry::run_auth`] keeps an await
        /// ahead of the hook, which is the caller's business and not the gate's.
        fn call(&self, _purpose: AuthPurpose) -> BoxFuture<'_, Result<PluginAuth, AgentError>> {
            Box::pin(async move {
                smol::future::yield_now().await;
                let run = self.calls.fetch_add(1, Ordering::AcqRel) + 1;
                Ok(PluginAuth {
                    base_url: self.base_url.clone(),
                    headers: HashMap::from([(
                        AUTH_HEADER.to_string(),
                        if self.rotating {
                            format!("Bearer refreshed-{run}")
                        } else {
                            CONSTANT_TOKEN.to_string()
                        },
                    )]),
                })
            })
        }
    }

    /// Says when it has started, then holds on until released, so a test can
    /// drop the caller while the hook is known to be mid-flight.
    struct ParkedAuth {
        entered: flume::Sender<()>,
        released: flume::Receiver<()>,
        calls: AtomicUsize,
    }

    impl Hook<AuthPurpose, PluginAuth> for ParkedAuth {
        fn call(&self, _purpose: AuthPurpose) -> BoxFuture<'_, Result<PluginAuth, AgentError>> {
            Box::pin(async move {
                let _ = self.entered.send(());
                let _ = self.released.recv_async().await;
                self.calls.fetch_add(1, Ordering::AcqRel);
                Ok(PluginAuth {
                    base_url: None,
                    headers: HashMap::from([(AUTH_HEADER.to_string(), FIRST_TOKEN.to_string())]),
                })
            })
        }
    }

    struct PanicHook;

    impl<In, Out> Hook<In, Out> for PanicHook {
        fn call(&self, _input: In) -> BoxFuture<'_, Result<Out, AgentError>> {
            panic!("{HOOK_CALLED}");
        }
    }

    struct RemapHook(Option<ApiError>);

    impl Hook<ApiError, Option<ApiError>> for RemapHook {
        fn call(&self, _input: ApiError) -> BoxFuture<'_, Result<Option<ApiError>, AgentError>> {
            let mapped = self.0.clone();
            Box::pin(async move { Ok(mapped) })
        }
    }

    struct UnusedProvider;

    impl Provider for UnusedProvider {
        fn stream_message<'a>(
            &'a self,
            _model: &'a Model,
            _messages: &'a [Message],
            _system: &'a str,
            _tools: &'a Value,
            _event_tx: &'a Sender<ProviderEvent>,
            _opts: RequestOptions,
            _session_id: Option<&'a SessionRef>,
        ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
            Box::pin(async {
                Err(AgentError::Config {
                    message: INNER_UNUSED.to_string(),
                })
            })
        }

        fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
            Box::pin(async {
                Err(AgentError::Config {
                    message: INNER_UNUSED.to_string(),
                })
            })
        }
    }

    fn entry_with(hooks: ProviderHooks) -> Arc<PluginEntry> {
        const SLUG: &str = "in-memory";
        let decl = decl(SLUG);
        let auth = AuthState::new();
        auth.redeclare(SLUG, None, &decl.net_hosts).unwrap();
        Arc::new(PluginEntry {
            decl,
            claimed: None,
            source: DeclSource::Lua,
            owner: None,
            target: Target::Codec(Protocol::Openai),
            hooks,
            auth: Arc::new(auth),
        })
    }

    fn provider_with(hooks: ProviderHooks) -> PluginProvider {
        PluginProvider {
            entry: entry_with(hooks),
            pool: None,
            inner: Box::new(UnusedProvider),
        }
    }

    fn token(entry: &PluginEntry) -> String {
        entry.auth.current.lock().unwrap().headers[0].1.clone()
    }

    fn base_url(slug: &str) -> Option<String> {
        let entry = entry(slug).unwrap();
        let auth = entry.auth.current.lock().unwrap();
        auth.base_url.clone()
    }

    #[test_case("myslug", true ; "valid_simple")]
    #[test_case("my-slug", true ; "valid_hyphen")]
    #[test_case("my_slug", true ; "valid_underscore")]
    #[test_case("A1", true ; "valid_upper")]
    #[test_case("", false ; "empty")]
    #[test_case("-bad", false ; "leading_hyphen")]
    #[test_case("has.dot", false ; "has_dot")]
    #[test_case("has/slash", false ; "has_slash")]
    #[test_case("has space", false ; "has_space")]
    fn slug_validation(input: &str, expected: bool) {
        assert_eq!(is_valid_slug(input), expected);
    }

    /// Two callers on one slug, because sub-agents each build their own
    /// provider over the same per-slug credentials. Only one may spend a
    /// rotating refresh token, and the parked caller finds the winner's token
    /// already in the shared cell instead of being handed a copy of it. A
    /// non-rotating hook repeats the same bytes, which is how we pin down that
    /// the gate trusts its counter and not a diff.
    #[test_case(true, FIRST_TOKEN, ROTATED_TOKEN ; "rotating_token")]
    #[test_case(false, CONSTANT_TOKEN, CONSTANT_TOKEN ; "unchanged_token")]
    fn refresh_gate_single_flights_concurrent_callers(
        rotating: bool,
        after_one: &str,
        after_two: &str,
    ) {
        let hook = Arc::new(CountingAuth::new(rotating));
        let entry = entry_with(ProviderHooks {
            auth: Some(hook.clone()),
            ..ProviderHooks::default()
        });

        smol::block_on(async {
            let (a, b) = zip(
                entry.run_auth(AuthPurpose::Refresh),
                entry.run_auth(AuthPurpose::Refresh),
            )
            .await;
            a.unwrap();
            b.unwrap();

            assert_eq!(hook.calls(), 1, "{ONE_HOOK_CALL}");
            assert_eq!(token(&entry), after_one);

            // The late caller snapshots the count before locking, so a refresh
            // that overlaps nobody still runs the hook.
            entry.run_auth(AuthPurpose::Refresh).await.unwrap();
        });

        assert_eq!(hook.calls(), 2, "{SINGLE_FLIGHT_RERUN}");
        assert_eq!(token(&entry), after_two);
    }

    #[test]
    fn reload_keeps_the_auth_state() {
        const SLUG: &str = "reload-plugin";
        let hook = Arc::new(CountingAuth::new(true));
        let with_hook = || {
            let mut reg = registration(SLUG);
            reg.hooks.auth = Some(hook.clone());
            reg
        };

        register_loaded(with_hook()).unwrap();
        let before = entry(SLUG).unwrap();
        register_loaded(with_hook()).unwrap();
        let after = entry(SLUG).unwrap();

        assert!(
            Arc::ptr_eq(&before.auth, &after.auth),
            "the new entry picks up the credentials the old one was using"
        );
        smol::block_on(async {
            let (a, b) = zip(
                before.run_auth(AuthPurpose::Refresh),
                after.run_auth(AuthPurpose::Refresh),
            )
            .await;
            a.unwrap();
            b.unwrap();
        });

        assert_eq!(hook.calls(), 1, "{ONE_HOOK_CALL}");
    }

    /// The defect [`begin_load`] exists to prevent: a `/reload` builds a new
    /// plugin host, and an entry the old one left behind answers on a channel
    /// nobody serves. A load that registers nothing must leave nothing.
    ///
    /// And the defect publishing-on-commit exists to prevent: the old entries
    /// only go once the replacements are ready, so a read landing while the
    /// load runs still gets the generation that is actually serving.
    #[test]
    fn a_new_load_drops_the_entries_of_the_previous_one() {
        const SLUG: &str = "stale-plugin";
        const CREDENTIALS_DROPPED: &str = "credentials must outlive the load that registered them";
        register_loaded(registration(SLUG)).unwrap();

        begin_load();
        assert!(is_registered(SLUG), "{RELOAD_KEEPS_SERVING}");
        commit_load();

        assert!(!is_registered(SLUG), "{RELOAD_DROPS_STALE}");
        assert!(
            AUTH.read().unwrap().contains_key(SLUG),
            "{CREDENTIALS_DROPPED}"
        );
    }

    /// The fence behind "no synchronous path calls a plugin": every hook here
    /// panics, and every synchronous entry point still answers.
    #[test]
    fn create_calls_no_hook() {
        const SLUG: &str = "sync-plugin";
        let mut reg = registration(SLUG);
        reg.hooks = ProviderHooks {
            auth: Some(Arc::new(PanicHook)),
            list_models: Some(Arc::new(PanicHook)),
            build_body: Some(Arc::new(PanicHook)),
            map_error: Some(Arc::new(PanicHook)),
            fetch_usage: Some(Arc::new(PanicHook)),
            login: Some(Arc::new(PanicHook)),
            logout: Some(Arc::new(PanicHook)),
        };
        register_loaded(reg).unwrap();

        create(SLUG, Timeouts::default()).unwrap();
        assert_eq!(display_name(SLUG).as_deref(), Some(DISPLAY_NAME));
        assert!(base_for_slug(SLUG).is_some());
        assert_eq!(lookup_model(SLUG, MODEL_VARIANT).unwrap().id, MODEL_VARIANT);
        assert_eq!(find_model_for_tier(SLUG, MODEL_TIER).unwrap().id, MODEL_ID);
        assert_eq!(plugin_model_specs_for(SLUG), [format!("{SLUG}/{MODEL_ID}")]);
        assert_eq!(
            auth_providers(),
            [(SLUG.to_string(), DISPLAY_NAME.to_string())]
        );
    }

    /// Availability follows the declaration and not who wrote it: a decl that
    /// names an `api_key_env` answers for its key, so `create` fails while
    /// there is none and the picker hides the provider rather than listing one
    /// that cannot serve a request. A key that appears later needs no reload,
    /// and neither does one that replaces a key the provider already had,
    /// which is what a user does after the old one was revoked.
    #[test]
    fn a_declared_key_env_decides_availability() {
        const SLUG: &str = "keyed-plugin";
        const ENV_VAR: &str = "MAKI_TEST_KEYED_PLUGIN_KEY";
        const KEY: &str = "sk-keyed";
        const REPLACED_KEY: &str = "sk-replaced";
        const LISTED_BUT_KEYLESS: &str = "a declared key env with no key must not build";
        const STALE_KEY: &str = "a replaced key must reach the next provider built";
        let mut reg = registration(SLUG);
        reg.decl.api_key_env = Some(ENV_VAR.to_string());
        register_loaded(reg).unwrap();

        let error = create(SLUG, Timeouts::default())
            .err()
            .expect(LISTED_BUT_KEYLESS);
        assert!(error.to_string().contains(ENV_VAR), "{error}");

        unsafe { std::env::set_var(ENV_VAR, KEY) };
        create(SLUG, Timeouts::default()).unwrap();
        assert_eq!(token(&entry(SLUG).unwrap()), format!("Bearer {KEY}"));

        unsafe { std::env::set_var(ENV_VAR, REPLACED_KEY) };
        create(SLUG, Timeouts::default()).unwrap();
        assert_eq!(
            token(&entry(SLUG).unwrap()),
            format!("Bearer {REPLACED_KEY}"),
            "{STALE_KEY}"
        );
    }

    /// A claim answers for a built-in row, so the built-ins' listing already
    /// covers it; listing it again as a plugin doubled every model it has and,
    /// with no key, offered the models anyway under a warning.
    #[test]
    fn only_a_slug_no_builtin_owns_is_listed_as_a_plugin() {
        const SLUG: &str = "unclaimed-plugin";
        begin_load();
        register(
            registration(SLUG),
            DeclSource::Lua,
            DeclAuthority::ThirdParty,
        )
        .unwrap();
        register(claim(CLAIMED_SLUG), DeclSource::Lua, DeclAuthority::Bundled).unwrap();
        commit_load();

        assert_eq!(unclaimed_slugs(), [SLUG]);
    }

    /// A plugin whose load fails after registering must leave no provider
    /// behind: a slug only it declared goes, and a Rust decl it displaced
    /// serves again, exactly as if the plugin had never loaded.
    #[test]
    fn a_discarded_plugin_leaves_nothing_staged() {
        const PLUGIN: &str = "failed-plugin";
        const SLUG: &str = "failed-plugin-slug";
        const DISCARDED_SERVES: &str = "a failed plugin's provider was published";
        const DISPLACED_NOT_RESTORED: &str = "the decl a failed plugin displaced must serve again";
        let displaced = deepseek::decl().slug;
        begin_load();
        register_plugin(
            Arc::from(PLUGIN),
            registration(SLUG),
            DeclAuthority::ThirdParty,
        )
        .unwrap();
        let takeover = Registration {
            decl: deepseek::decl(),
            hooks: deepseek::hooks(),
        };
        register_plugin(Arc::from(PLUGIN), takeover, DeclAuthority::Bundled).unwrap();
        discard(PLUGIN);
        commit_load();

        assert!(!is_registered(SLUG), "{DISCARDED_SERVES}");
        assert_eq!(
            source_of(&displaced),
            Some(DeclSource::Rust),
            "{DISPLACED_NOT_RESTORED}"
        );
    }

    /// A refresh spends a single-use token, so the caller going away (an Esc,
    /// a turn that timed out) must not abandon it halfway: the hook runs to
    /// the end, what it minted lands, and the gate stays shut until it has.
    #[test]
    fn a_refresh_outlives_the_caller_that_started_it() {
        const MINT_DROPPED: &str = "what the abandoned refresh minted never landed";
        let (entered_tx, entered) = flume::bounded(1);
        let (release, released) = flume::bounded(1);
        let hook = Arc::new(ParkedAuth {
            entered: entered_tx,
            released,
            calls: AtomicUsize::new(0),
        });
        let entry = entry_with(ProviderHooks {
            auth: Some(hook.clone()),
            ..ProviderHooks::default()
        });

        smol::block_on(async {
            let abandoned =
                futures_lite::future::poll_once(entry.run_auth(AuthPurpose::Refresh)).await;
            assert!(abandoned.is_none());
            entered.recv_async().await.unwrap();
            release.send(()).unwrap();
            let _gate = entry.auth.gate.lock.lock().await;
        });

        assert_eq!(hook.calls.load(Ordering::Acquire), 1, "{ONE_HOOK_CALL}");
        assert_eq!(token(&entry), FIRST_TOKEN, "{MINT_DROPPED}");
    }

    /// The 401 replay is a credential refresh, not a retry: a declaration
    /// whose key comes from an `api_key_env` has no hook to mint a new one, so
    /// a second request would only spend the key the server just rejected.
    #[test]
    fn a_hookless_decl_does_not_replay_a_401() {
        const SLUG: &str = "hookless-401-plugin";
        const ENV_VAR: &str = "MAKI_TEST_HOOKLESS_401_KEY";
        /// `<SLUG>_BASE_URL`, which is how the codec reaches loopback.
        const BASE_URL_ENV: &str = "HOOKLESS_401_PLUGIN_BASE_URL";
        const KEY: &str = "sk-rejected";
        const PROMPT: &str = "read a.txt";
        const UNAUTHORIZED: u16 = 401;
        const UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"invalid api key"}}"#;
        const REJECTED_TWICE: &str = "a decl with no auth hook re-sent the rejected key";
        const STILL_AUTHORIZED: &str = "the auth error must reach the caller";
        /// Two answers so a replaying provider is recorded rather than parked
        /// on an `accept` that never returns.
        const SCRIPT: &[Canned] = &[
            Canned::json(UNAUTHORIZED, UNAUTHORIZED_BODY),
            Canned::json(UNAUTHORIZED, UNAUTHORIZED_BODY),
        ];

        let (base_url, requests) = serve(SCRIPT);
        unsafe {
            std::env::set_var(ENV_VAR, KEY);
            std::env::set_var(BASE_URL_ENV, &base_url);
        }
        let mut reg = registration(SLUG);
        reg.decl.api_key_env = Some(ENV_VAR.to_string());
        register_loaded(reg).unwrap();

        let provider = create(SLUG, Timeouts::default()).unwrap();
        let model = lookup_model(SLUG, MODEL_ID).unwrap();
        let messages = [Message::user(PROMPT.to_owned())];
        let (tx, _rx) = flume::unbounded();
        let result = smol::block_on(provider.stream_message(
            &model,
            &messages,
            "",
            &serde_json::json!([]),
            &tx,
            RequestOptions::default(),
            None,
        ));

        assert!(
            result.as_ref().err().is_some_and(AgentError::is_auth_error),
            "{STILL_AUTHORIZED}"
        );
        assert_eq!(requests.lock().unwrap().len(), 1, "{REJECTED_TWICE}");
    }

    /// Only an origin the user or maki chose skips the side-call guard, never
    /// the one a third-party decl wrote.
    #[test_case(DeclAuthority::ThirdParty, None, None ; "third_party_declared_origin")]
    #[test_case(DeclAuthority::ThirdParty, Some(OTHER_BASE_URL), Some(OTHER_BASE_URL) ; "third_party_user_origin")]
    #[test_case(DeclAuthority::Bundled, None, Some(EXAMPLE_BASE_URL) ; "bundled_declared_origin")]
    fn vouched_origin_is_the_users_or_makis(
        authority: DeclAuthority,
        configured: Option<&str>,
        expected: Option<&str>,
    ) {
        const THIRD_PARTY_SLUG: &str = "vouch-plugin";
        let (slug, reg) = match authority {
            DeclAuthority::Bundled => (CLAIMED_SLUG, claim(CLAIMED_SLUG)),
            DeclAuthority::ThirdParty => (THIRD_PARTY_SLUG, registration(THIRD_PARTY_SLUG)),
        };
        if let Some(configured) = configured {
            unsafe { std::env::set_var(base_url_env_var(slug), configured) };
        }
        register_loaded_as(reg, authority).unwrap();

        let expected = expected.map(|url| Url::parse(url).unwrap());
        assert_eq!(vouched_origin(slug), expected);
    }

    /// A decl that claims a built-in slug states only what the row does not
    /// already answer, and every inherited answer comes off the row: the name,
    /// the curated table, and the spec behind it, which here is *not* the one
    /// the declared codec would have resolved to.
    #[test]
    fn a_claim_inherits_the_spec_row() {
        let spec = ProviderRegistry::get(CLAIMED_SLUG).expect(NO_CLAIMED_SPEC);
        register_loaded_as(claim(CLAIMED_SLUG), DeclAuthority::Bundled).unwrap();

        assert_eq!(
            display_name(CLAIMED_SLUG).as_deref(),
            Some(spec.display_name)
        );
        assert_eq!(
            base_for_slug(CLAIMED_SLUG).map(|spec| spec.slug),
            Some(CLAIMED_SLUG),
            "{CLAIM_READS_THE_CODEC_SPEC}"
        );
        assert_eq!(
            plugin_model_specs_for(CLAIMED_SLUG).len(),
            spec.models()
                .iter()
                .map(|entry| entry.prefixes.len())
                .sum::<usize>(),
            "{CLAIM_LOST_THE_TABLE}"
        );
    }

    /// A dialect is named, not spelled out: it resolves by name, an unknown
    /// name says which names there are, and a decl dumps the name it came
    /// from rather than storing it twice.
    #[test]
    fn a_dialect_is_resolved_by_name() {
        const KNOWN: &str = "deepseek";
        const UNKNOWN: &str = "not-a-dialect";
        let authored = |name: &str| {
            serde_json::json!({
                "slug": "dialect-plugin",
                "codec": "openai",
                "openai": { "thinking": { "dialect": name } },
            })
        };

        let decl: ProviderDecl = serde_json::from_value(authored(KNOWN)).unwrap();
        let thinking = decl.openai.as_ref().and_then(|wire| wire.thinking.as_ref());
        assert_eq!(
            thinking.map(|thinking| thinking.dialect),
            Some(&dialect::DEEPSEEK)
        );

        let error = serde_json::from_value::<ProviderDecl>(authored(UNKNOWN))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains(UNKNOWN) && error.contains(KNOWN), "{error}");

        let dumped = serde_json::to_value(&decl).unwrap();
        assert_eq!(
            dumped["openai"]["thinking"]["dialect"],
            serde_json::json!(KNOWN)
        );
    }

    /// A decl that claims a built-in slug: it states only what the spec row
    /// does not already answer.
    fn claim(slug: &str) -> Registration {
        let mut reg = registration(slug);
        reg.decl.display_name = None;
        reg.decl.models.clear();
        reg
    }

    fn bad_slug(reg: &mut Registration) {
        reg.decl.slug = "has.dot".to_string();
    }

    fn no_display_name(reg: &mut Registration) {
        reg.decl.display_name = None;
    }

    fn claim_restating_the_display_name(reg: &mut Registration) {
        *reg = claim(CLAIMED_SLUG);
        reg.decl.display_name = Some(DISPLAY_NAME.to_string());
    }

    fn claim_restating_the_api_key_env(reg: &mut Registration) {
        *reg = claim(CLAIMED_SLUG);
        reg.decl.api_key_env = Some("PLUGIN_API_KEY".to_string());
    }

    /// A claim keeping the table [`claim`] drops.
    fn claim_restating_the_model_table(reg: &mut Registration) {
        *reg = registration(CLAIMED_SLUG);
        reg.decl.display_name = None;
    }

    fn already_registered(reg: &mut Registration) {
        register(
            registration(&reg.decl.slug),
            DeclSource::Lua,
            DeclAuthority::Bundled,
        )
        .unwrap();
    }

    fn codec_and_base(reg: &mut Registration) {
        reg.decl.base = Some("openai".to_string());
    }

    fn missing_base(reg: &mut Registration) {
        reg.decl.codec = None;
        reg.decl.base = Some("not-a-provider".to_string());
    }

    fn no_net_hosts(reg: &mut Registration) {
        reg.decl.net_hosts.clear();
    }

    fn outside_a_load(_reg: &mut Registration) {
        commit_load();
    }

    fn body_hook_on_anthropic(reg: &mut Registration) {
        reg.decl.codec = Some(Protocol::Anthropic);
        reg.hooks.build_body = Some(Arc::new(PanicHook));
    }

    /// The table the chat codec reads, on a codec that would read none of it.
    fn openai_wire_on_anthropic(reg: &mut Registration) {
        reg.decl.codec = Some(Protocol::Anthropic);
        reg.decl.openai = Some(OpenAiWire::default());
    }

    fn system_prefix_on_google(reg: &mut Registration) {
        reg.decl.codec = Some(Protocol::Google);
        reg.decl.system_prefix = Some(SOME_SYSTEM_PREFIX.to_string());
    }

    /// `base = "google"` reaches the same constructor as `codec = "google"`,
    /// which drops the prefix, so it has to be refused just as loudly.
    fn system_prefix_on_the_google_base(reg: &mut Registration) {
        reg.decl.codec = None;
        reg.decl.base = Some(super::super::google::SLUG.to_string());
        reg.decl.system_prefix = Some(SOME_SYSTEM_PREFIX.to_string());
    }

    fn base_url_off_the_declared_hosts(reg: &mut Registration) {
        reg.decl.base_url = Some(UNDECLARED_BASE_URL.to_string());
    }

    /// A declared host reached over plaintext still puts the token on the wire
    /// in the clear, so the host list alone is not the whole question.
    fn base_url_over_plain_http(reg: &mut Registration) {
        reg.decl.base_url = Some(format!("http://{EXAMPLE_HOST}/v1"));
    }

    /// `ConfiguredSlug` and `Credentials` have no row: both need a
    /// `providers.toml` entry for the slug, and the process-wide config is not
    /// a test fixture. They are covered out of line, in
    /// `tests/configured_overlay.rs`.
    #[test_case(bad_slug, |e| matches!(e, RegisterError::InvalidSlug(_)) ; "invalid_slug")]
    #[test_case(no_display_name, |e| matches!(e, RegisterError::NoDisplayName(_)) ; "display_name_is_mandatory_without_a_row_to_inherit_it_from")]
    #[test_case(claim_restating_the_display_name, |e| matches!(e, RegisterError::Restated { field, .. } if *field == DISPLAY_NAME_FIELD) ; "claim_restates_display_name")]
    #[test_case(claim_restating_the_api_key_env, |e| matches!(e, RegisterError::Restated { field, .. } if *field == API_KEY_ENV_FIELD) ; "claim_restates_api_key_env")]
    #[test_case(claim_restating_the_model_table, |e| matches!(e, RegisterError::Restated { field, .. } if *field == MODELS_FIELD) ; "claim_restates_models")]
    #[test_case(already_registered, |e| matches!(e, RegisterError::DuplicateSlug(_)) ; "duplicate")]
    #[test_case(codec_and_base, |e| matches!(e, RegisterError::CodecOrBase(_)) ; "codec_and_base_together")]
    #[test_case(missing_base, |e| matches!(e, RegisterError::UnknownBase { .. }) ; "unknown_base")]
    #[test_case(no_net_hosts, |e| matches!(e, RegisterError::NoNetHosts(_)) ; "net_hosts_empty")]
    #[test_case(outside_a_load, |e| matches!(e, RegisterError::Closed(_)) ; "registration_outside_a_load")]
    #[test_case(body_hook_on_anthropic, |e| matches!(e, RegisterError::Unsupported { .. }) ; "build_body_needs_an_openai_codec")]
    #[test_case(openai_wire_on_anthropic, |e| matches!(e, RegisterError::Unsupported { option, .. } if *option == OPENAI_OPTION) ; "openai_table_needs_the_openai_codec")]
    #[test_case(system_prefix_on_google, |e| matches!(e, RegisterError::Unsupported { .. }) ; "google_drops_the_system_prefix")]
    #[test_case(system_prefix_on_the_google_base, |e| matches!(e, RegisterError::Unsupported { .. }) ; "so_does_the_google_base")]
    #[test_case(base_url_off_the_declared_hosts, |e| matches!(e, RegisterError::UndeclaredBaseUrl(_)) ; "base_url_must_be_declared")]
    #[test_case(base_url_over_plain_http, |e| matches!(e, RegisterError::UndeclaredBaseUrl(_)) ; "base_url_must_be_https")]
    fn registration_rejects(mutate: fn(&mut Registration), expected: fn(&RegisterError) -> bool) {
        const SLUG: &str = "rejected-plugin";
        begin_load();
        let mut reg = registration(SLUG);
        mutate(&mut reg);
        let error = register(reg, DeclSource::Lua, DeclAuthority::Bundled).unwrap_err();
        assert!(expected(&error), "{error}");
        commit_load();
    }

    /// A built-in slug is maki's to declare and nobody else's. Claiming one
    /// inherits its `api_key_env`, so a package that took `anthropic` would
    /// have the user's Anthropic key resolved into its own credentials and
    /// sent to the origin it declared. That is reach a grant of `net` plus a
    /// host list never bought, under a name the picker still labels
    /// "Anthropic".
    ///
    /// Refused outright rather than demoted to a non-claim, because a third
    /// party serving `anthropic` on its own terms is the same theft without
    /// the inheritance.
    #[test]
    fn a_third_party_cannot_declare_a_builtin_slug() {
        let error = register_loaded(registration(CLAIMED_SLUG)).unwrap_err();

        assert!(matches!(error, RegisterError::ReservedSlug(_)), "{error}");
        assert!(!is_registered(CLAIMED_SLUG), "{RESERVED_SLUG_TAKEN}");
    }

    /// A decl that loses its slug is well formed, since `impl = "rust"` is a
    /// user choice and not a defect, so it registers without error. It must
    /// still say that it is not the one serving, because every grant a caller
    /// hangs off a registration belongs to the winner alone.
    #[test]
    fn a_shadowed_decl_registers_without_serving() {
        const SLUG: &str = "shadowed-decl";
        load_with_rust_decl(SLUG);

        let first = register(
            registration(SLUG),
            DeclSource::Lua,
            DeclAuthority::ThirdParty,
        );
        // Lua wins by default, so the rust decl is the one shadowed on the way
        // back: same slug, same call, the other author.
        let second = register(registration(SLUG), DeclSource::Rust, DeclAuthority::Bundled);
        commit_load();

        assert_eq!(first.unwrap(), Registered::Serving);
        assert_eq!(second.unwrap(), Registered::Shadowed, "{SHADOWED_SERVES}");
    }

    /// Stands in for what [`begin_load`] does with [`RUST_DECLS`], for a slug
    /// maki ships no declaration for: staged before any plugin gets to speak.
    fn load_with_rust_decl(slug: &str) {
        begin_load();
        register(registration(slug), DeclSource::Rust, DeclAuthority::Bundled).unwrap();
    }

    fn source_of(slug: &str) -> Option<DeclSource> {
        entry(slug).map(|entry| entry.source)
    }

    /// Both declarations reach [`codec::build`] the same way, so the config key
    /// picks an author and nothing else. Driven through [`wins`] rather than a
    /// registration because the process-wide config is not a test fixture.
    #[test_case(None, DeclSource::Lua ; "lua_unless_asked_otherwise")]
    #[test_case(Some(ImplChoice::Lua), DeclSource::Lua ; "lua")]
    #[test_case(Some(ImplChoice::Rust), DeclSource::Rust ; "rust")]
    fn the_configured_impl_picks_the_author(configured: Option<ImplChoice>, expected: DeclSource) {
        let served = if wins(DeclSource::Rust, DeclSource::Lua, configured) {
            DeclSource::Lua
        } else {
            DeclSource::Rust
        };
        assert_eq!(served, expected);
    }

    /// The escape hatch has to survive the check that follows it: `impl` is
    /// how a user picks between two declarations, so an entry carrying only
    /// that is not a `providers.toml` provider and must not reject both.
    #[test_case(ProviderDef::default(), false ; "an_empty_entry_defines_nothing")]
    #[test_case(ProviderDef { r#impl: Some(ImplChoice::Rust), ..ProviderDef::default() }, false ; "impl_alone_only_picks_the_author")]
    #[test_case(ProviderDef { base_url: Some(EXAMPLE_BASE_URL.to_owned()), r#impl: Some(ImplChoice::Rust), ..ProviderDef::default() }, true ; "anything_else_is_a_definition")]
    fn a_configured_entry_defines_a_provider(def: ProviderDef, expected: bool) {
        assert_eq!(defines_provider(&def), expected);
    }

    fn lua_says_nothing(_slug: &str) -> Option<RegisterError> {
        None
    }

    fn one_lua_decl(slug: &str) -> Option<RegisterError> {
        lua_decl(registration(slug))
    }

    fn two_lua_decls(slug: &str) -> Option<RegisterError> {
        assert!(one_lua_decl(slug).is_none(), "{FIRST_LUA_DECL_REJECTED}");
        one_lua_decl(slug)
    }

    fn a_broken_lua_decl(slug: &str) -> Option<RegisterError> {
        let mut reg = registration(slug);
        no_net_hosts(&mut reg);
        lua_decl(reg)
    }

    fn lua_decl(reg: Registration) -> Option<RegisterError> {
        register(reg, DeclSource::Lua, DeclAuthority::ThirdParty).err()
    }

    /// Who ends up serving a slug maki already declared, once the Lua half of
    /// the load has had its say. Lua wins by default, which keeps everyday use
    /// on the authoring surface plugin authors write against, and a Lua decl
    /// that fails costs the user nothing: the shipped decl it never displaced
    /// keeps serving, and `register` warns so that "why is this provider odd"
    /// has something to go on.
    ///
    /// A second decl from the *same* author is a collision either way. The
    /// author decides precedence, never the slug.
    #[test_case(lua_says_nothing, DeclSource::Rust, None ; "silence_leaves_the_rust_decl")]
    #[test_case(one_lua_decl, DeclSource::Lua, None ; "a_lua_decl_takes_the_slug")]
    #[test_case(two_lua_decls, DeclSource::Lua, Some(|e| matches!(e, RegisterError::DuplicateSlug(_))) ; "a_second_lua_decl_collides")]
    #[test_case(a_broken_lua_decl, DeclSource::Rust, Some(|e| matches!(e, RegisterError::NoNetHosts(_))) ; "a_broken_lua_decl_changes_nothing")]
    fn the_lua_half_of_a_load_decides_the_author(
        lua: fn(&str) -> Option<RegisterError>,
        serving: DeclSource,
        rejected: Option<fn(&RegisterError) -> bool>,
    ) {
        const SLUG: &str = "contested-slug";
        load_with_rust_decl(SLUG);
        let rejection = lua(SLUG);
        commit_load();

        assert_eq!(
            rejection.is_some(),
            rejected.is_some(),
            "{PRECEDENCE_MISMATCH}: {rejection:?}"
        );
        if let (Some(expected), Some(error)) = (rejected, &rejection) {
            assert!(expected(error), "{error}");
        }
        assert_eq!(source_of(SLUG), Some(serving), "{WRONG_AUTHOR_SERVING}");
    }

    const UPSTREAM_SAID_NO: &str = "upstream said no";
    /// A status maki never retries, so a mapping that changes nothing is
    /// visible in `retry_kind` as well as in the status.
    const NOT_RETRYABLE: u16 = 418;
    const RATE_LIMITED: u16 = 429;

    fn api_error(status: u16, retry_after: Option<Duration>) -> AgentError {
        AgentError::Api {
            status,
            message: UPSTREAM_SAID_NO.to_string(),
            retry_after,
        }
    }

    #[test]
    fn map_error_absent_leaves_the_error_untouched() {
        let provider = provider_with(ProviderHooks::default());
        let error =
            smol::block_on(provider.mapped::<()>(Err(api_error(NOT_RETRYABLE, None)))).unwrap_err();
        assert!(matches!(
            error,
            AgentError::Api {
                status: NOT_RETRYABLE,
                ..
            }
        ));
        assert_eq!(error.retry_kind(), None);
    }

    /// The hook restates status and message. Retryability and `Retry-After`
    /// stay maki's to decide.
    #[test]
    fn map_error_remaps_the_status_only() {
        const RETRY_AFTER: Duration = Duration::from_secs(7);
        const REMAPPED: &str = "slow down";
        let provider = provider_with(ProviderHooks {
            map_error: Some(Arc::new(RemapHook(Some(ApiError {
                status: RATE_LIMITED,
                message: REMAPPED.to_string(),
            })))),
            ..ProviderHooks::default()
        });

        let error =
            smol::block_on(provider.mapped::<()>(Err(api_error(NOT_RETRYABLE, Some(RETRY_AFTER)))))
                .unwrap_err();

        assert!(
            matches!(&error, AgentError::Api { status: RATE_LIMITED, message, .. } if message == REMAPPED)
        );
        assert_eq!(error.retry_kind(), Some(RetryKind::RateLimit));
        assert_eq!(error.retry_after(), Some(RETRY_AFTER));
    }

    #[test]
    fn plugin_auth_rejects_an_undeclared_base_url() {
        const SLUG: &str = "egress-plugin";
        let hosts = vec![EXAMPLE_HOST.to_string()];
        let declared = format!("{EXAMPLE_BASE_URL}/v1");
        let auth = |base_url: &str| PluginAuth {
            base_url: Some(base_url.to_string()),
            headers: HashMap::new(),
        };

        assert!(
            auth(UNDECLARED_BASE_URL)
                .into_resolved(SLUG, &hosts)
                .is_err()
        );
        assert_eq!(
            auth(&declared)
                .into_resolved(SLUG, &hosts)
                .unwrap()
                .base_url
                .as_deref(),
            Some(declared.as_str())
        );
    }

    /// `http` is admitted for loopback alone, so a provider served on the same
    /// machine still works without opening plaintext egress to the internet.
    #[test_case("https://example.com/v1", EXAMPLE_HOST, true ; "https_to_a_declared_host")]
    #[test_case("http://example.com/v1", EXAMPLE_HOST, false ; "plaintext_to_a_remote_host")]
    #[test_case("http://localhost:8080/v1", LOCALHOST, true ; "plaintext_to_localhost")]
    #[test_case("http://127.0.0.1:8080/v1", "127.0.0.1", true ; "plaintext_to_a_loopback_address")]
    #[test_case("ftp://example.com/v1", EXAMPLE_HOST, false ; "a_scheme_that_is_neither")]
    fn base_url_scheme(url: &str, host: &str, accepted: bool) {
        const SLUG: &str = "scheme-plugin";
        let hosts = vec![host.to_string()];
        assert_eq!(
            declared_base_url(SLUG, Some(url.to_string()), &hosts).is_ok(),
            accepted,
            "{url}"
        );
    }

    /// What the per-slug auth cell must and must not carry across a reload: a
    /// slug that has minted nothing takes the fresh declaration, an origin a
    /// hook leased survives as long as the new declaration still covers it,
    /// and it goes the moment a narrowed host list stops covering it.
    ///
    /// The origin is always the hook's. A declared `base_url` is the codec's
    /// last resort and never reaches the auth cell, which is what keeps a
    /// plugin from outranking the user's `<SLUG>_BASE_URL`.
    #[test_case(false, true, None ; "an_unminted_slug_takes_the_fresh_declaration")]
    #[test_case(true, true, Some(OTHER_BASE_URL) ; "a_leased_origin_survives_the_reload")]
    #[test_case(true, false, None ; "narrowing_the_hosts_drops_an_origin_they_no_longer_cover")]
    fn a_reload_re_reads_the_declaration(
        minted: bool,
        still_covered: bool,
        expected: Option<&str>,
    ) {
        const SLUG: &str = "redeclare-plugin";
        let declaration = |covers_the_lease: bool| {
            let mut reg = registration(SLUG);
            if covers_the_lease {
                reg.decl.net_hosts.push(OTHER_HOST.to_string());
            }
            reg
        };

        let mut first = declaration(true);
        if minted {
            first.hooks.auth = Some(Arc::new(CountingAuth::leasing(OTHER_BASE_URL)));
        }
        register_loaded(first).unwrap();
        if minted {
            smol::block_on(entry(SLUG).unwrap().ensure_auth()).unwrap();
            assert_eq!(
                base_url(SLUG).as_deref(),
                Some(OTHER_BASE_URL),
                "{LEASE_LOST}"
            );
        }

        register_loaded(declaration(still_covered)).unwrap();

        assert_eq!(base_url(SLUG).as_deref(), expected);
    }
}
