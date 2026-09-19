use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use flume::Sender;
use maki_config::host_allowed;
use maki_config::providers::{ImplChoice, Protocol, ProviderDef, ProvidersConfig};
use maki_storage::StateDir;
use maki_storage::auth::lock_credentials;
use maki_storage::id::SessionRef;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use tracing::{debug, warn};
use url::{Host, Url};

use crate::model::{
    Model, ModelInfo, ModelPricing, ModelTier, Prefixed, ThinkingSupport, longest_prefix_match,
};
use crate::provider::{BoxFuture, Provider};
use crate::spec::{ProviderRegistry, ProviderSpec};
use crate::types::{EffortDialect, ThinkingFields, dialect};
use crate::{AgentError, Message, ProviderEvent, ProviderUsage, RequestOptions, StreamResponse};

use super::codec::{self, BodyHook, CodecOptions};
use super::{KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts, synthetic};

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 16384;
const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;
const BUILD_BODY_OPTION: &str = "build_body hook";
const SYSTEM_PREFIX_OPTION: &str = "system_prefix";
const DISPLAY_NAME_FIELD: &str = "display_name";
const API_KEY_ENV_FIELD: &str = "api_key_env";
const MODELS_FIELD: &str = "models";
const IMPL_FIELD: &str = "impl";
const HTTPS_SCHEME: &str = "https";
const HTTP_SCHEME: &str = "http";
const LOCALHOST: &str = "localhost";

/// One plugin-supplied callback. Generic in both directions so every hook on
/// [`ProviderHooks`] is the same shape, and `Option::is_some` is the only
/// presence question the registry ever asks.
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

/// The request as it goes on the wire, plus the two things a plugin branches
/// on. `thinking` is rendered, not structured, because the hook is a wire-level
/// escape hatch and not a second place to model effort.
#[derive(Serialize)]
pub struct BodyInput {
    pub body: Value,
    pub model: String,
    pub thinking: String,
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

/// The only place an origin a plugin chose is admitted, whether it arrived with
/// the registration or from an auth hook. Both paths end up holding the token
/// maki sends, so both ask the same question of the same list.
///
/// Origin, not host: the scheme is half of what a declaration promises. A
/// declared host reached over plaintext puts the token on the wire in the
/// clear, so only `https` is admitted. `http` stays open for loopback, where a
/// self-hosted provider has no wire to listen on.
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
/// defaults included, so a row that went through either survives a round trip
/// and two decls of the same provider compare equal whichever way they were
/// authored.
#[derive(Clone, PartialEq, Deserialize, Serialize)]
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

    fn to_model(
        &self,
        slug: &str,
        base: &'static ProviderSpec,
        id: String,
        tier: ModelTier,
    ) -> Model {
        Model {
            id,
            provider: Arc::from(slug),
            tier,
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

    /// This row as the catalogue reports it. Every field the declaration
    /// states is carried, `supports_*` included: they are `Option` on both
    /// sides, so an unstated one stays unstated rather than becoming a
    /// published negative. `provider_info` is a stash only the Rust provider
    /// that filled it can read back, so a declared row never has one.
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

/// One provider, as data. Everything maki needs to build it except the
/// callbacks, which is what lets two declarations of the same provider be
/// compared and dumped: for a provider with no hooks, `PartialEq` over this is
/// a complete equivalence proof between the Rust-authored and the Lua-authored
/// spelling of it.
///
/// `PartialEq` and not `Eq` because a declared [`ModelPricing`] is four `f64`
/// rates, which have no total equality. Comparing them bitwise is exactly the
/// question being asked -- two decls state the same rate or they do not.
#[derive(Clone, PartialEq, Serialize)]
pub struct ProviderDecl {
    pub slug: String,
    /// `None` only for a decl that claims a built-in slug, which inherits the
    /// name off the spec row rather than restating it.
    pub display_name: Option<String>,
    pub codec: Option<Protocol>,
    pub base: Option<String>,
    /// The declaration's static origin, below the user's `<SLUG>_BASE_URL` and
    /// `providers.toml`. An origin a hook returns is a different thing and
    /// outranks both; see [`CodecOptions::base_url`].
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub system_prefix: Option<String>,
    /// `None` is `max_tokens`; see [`CodecOptions::max_tokens_field`].
    pub max_tokens_field: Option<String>,
    /// `None` asks for streamed usage; see
    /// [`CodecOptions::include_stream_usage`].
    pub include_stream_usage: Option<bool>,
    /// How this provider's API spells reasoning effort, when it has its own
    /// word for it.
    #[serde(serialize_with = "serialize_dialect")]
    pub thinking_dialect: Option<&'static EffortDialect<'static>>,
    pub models: Vec<PluginModel>,
    pub net_hosts: Vec<String>,
}

/// Serialised as the dialect's name, so the table stays the one source of
/// truth for what a dialect is and a decl carries the name once, in the
/// dialect it already holds.
fn serialize_dialect<S: Serializer>(
    thinking_dialect: &Option<&'static EffortDialect<'static>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    thinking_dialect
        .and_then(dialect::name_of)
        .serialize(serializer)
}

/// A declaration plus the callbacks that go with it. The split is the point:
/// the data half is comparable and dumpable, the behaviour half is neither.
pub struct Registration {
    pub decl: ProviderDecl,
    pub hooks: ProviderHooks,
}

/// Which language a declaration was authored in, which is the only axis a
/// provider varies on once it is data. Recorded rather than inferred: nothing
/// about a registered entry says where it came from, and every log line about
/// one should.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclSource {
    Rust,
    Lua,
}

#[derive(Debug, thiserror::Error)]
pub enum RegisterError {
    #[error(
        "invalid provider slug '{0}': must start with a letter or digit and hold only letters, digits, '_' and '-'"
    )]
    InvalidSlug(String),
    #[error("provider slug '{0}' is already defined in providers.toml")]
    ConfiguredSlug(String),
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
    #[error("provider '{slug}': unknown thinking dialect '{name}' (expected one of {expected})")]
    UnknownDialect {
        slug: String,
        name: String,
        expected: String,
    },
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

/// Resolves the dialect a declaration names, for an authoring surface that
/// carries names rather than consts.
pub fn thinking_dialect(
    slug: &str,
    name: &str,
) -> Result<&'static EffortDialect<'static>, RegisterError> {
    dialect::by_name(name).ok_or_else(|| RegisterError::UnknownDialect {
        slug: slug.to_string(),
        name: name.to_string(),
        expected: dialect::NAMES.join(", "),
    })
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
/// Written as an exhaustive match rather than a list of the ones that work, so
/// a new codec breaks this line and someone has to answer for it. An option a
/// codec cannot honour is a registration error, never a no-op.
fn honours_build_body(target: Target) -> bool {
    match target {
        Target::Codec(Protocol::Openai | Protocol::OpenaiResponses) => true,
        Target::Codec(Protocol::Anthropic | Protocol::Google) | Target::Base(_) => false,
    }
}

/// Google drops the system prefix and always has (see `super::google`), so a
/// plugin that sets one against it is told instead of ignored. Asked of the
/// spec behind the target rather than of the target, because `codec = "google"`
/// and `base = "google"` reach the same constructor and must answer alike.
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
    /// already know. Everything a claim does not restate is read back off it
    /// through [`PluginEntry::spec`] and [`PluginEntry::display_name`], which
    /// is why inheritance needs no "am I a claim" branch anywhere else.
    claimed: Option<&'static ProviderSpec>,
    source: DeclSource,
    target: Target,
    hooks: ProviderHooks,
    /// Shared with every other entry this slug ever had: an entry is replaced
    /// on reload, its credentials are not. Held here rather than looked up
    /// beside the entry so "registered" and "has credentials" cannot come
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
/// Append-only for the life of the process: see the note in [`register`].
static AUTH: LazyLock<RwLock<HashMap<Box<str>, Arc<AuthState>>>> = LazyLock::new(RwLock::default);

/// Every declaration maki itself authors, staged by [`begin_load`] before a
/// plugin gets to speak. Written out rather than collected with `inventory`
/// for the reason [`crate::spec::ProviderRegistry::builtins`] gives, and
/// because one list read top to bottom is how anyone tells which providers are
/// ported.
///
/// Each port appends its constructor here (`anthropic::decl`), and the list --
/// with the seeding in [`begin_load`] and the precedence in [`wins`] -- goes
/// away with the last provider that still needs it.
const RUST_DECLS: &[fn() -> Registration] = &[|| Registration {
    decl: synthetic::decl(),
    hooks: ProviderHooks::default(),
}];

/// Opens the registration window, at the top of every plugin load.
///
/// Nothing published goes away here. A `/reload` builds a new plugin host, and
/// an entry left behind by the old one holds hooks that answer on a channel
/// nobody serves any more. Dropping them before the replacements exist would be
/// worse: every reader in the process would answer "unknown provider" for as
/// long as the load takes. The staged map replaces the published one in one
/// step at [`commit_load`] instead.
///
/// The Rust-authored decls are staged first, so every slug maki ships one for
/// is already answered when the load starts: a Lua decl that never arrives, or
/// arrives broken, leaves that one standing instead of leaving the slug with
/// nothing.
pub fn begin_load() {
    *STAGING.lock().unwrap() = Some(Registry::new());
    for declare in RUST_DECLS {
        let reg = declare();
        let slug = reg.decl.slug.clone();
        if let Err(error) = register(reg, DeclSource::Rust) {
            warn!(slug, %error, "built-in provider declaration was rejected");
        }
    }
}

/// Publishes what this load registered. A load that registered nothing
/// publishes nothing, which is how a `/reload` drops a plugin that was removed.
pub fn commit_load() {
    let Some(staged) = STAGING.lock().unwrap().take() else {
        return;
    };
    *PROVIDERS.write().unwrap() = Arc::new(staged);
}

/// Which declaration serves a slug both a Rust and a Lua author declared:
/// `true` when `challenger` takes the slug off `incumbent`.
///
/// The only axis is who wrote the declaration -- both drive the same
/// [`codec::build`] call -- so Lua wins by default and the field exercises the
/// path the Lua authoring surface uses. `impl = "rust"` in `providers.toml`
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

/// Who holds this slug in the load in progress, asked of the staged map rather
/// than of what is published: a reload re-registers every slug it registered
/// last time, and [`begin_load`] has already staged the Rust-authored decls.
fn staged_source(slug: &str) -> Result<Option<DeclSource>, RegisterError> {
    let staging = STAGING.lock().unwrap();
    let staged = staging
        .as_ref()
        .ok_or_else(|| RegisterError::Closed(slug.to_string()))?;
    Ok(staged.get(slug).map(|entry| entry.source))
}

/// Registers a declaration, or leaves the slug to the one that beat it.
///
/// A declaration rejected while another source's is standing costs a warning
/// and not a provider: the incumbent keeps serving the slug. That is a safe
/// fallback only because both declarations drive the same code path -- the
/// choice is of author, never of implementation -- and it is a loud one
/// because a silent swap leaves "why is my provider behaving oddly" with
/// nothing to go on. The error still goes back to the caller, so the plugin
/// that failed is still reported as failed.
pub fn register(reg: Registration, source: DeclSource) -> Result<(), RegisterError> {
    let slug = reg.decl.slug.clone();
    let result = register_decl(reg, source);
    if let Err(error) = &result
        && let Some(standing) = staged_source(&slug).ok().flatten().filter(|s| *s != source)
    {
        warn!(
            slug,
            ?source,
            ?standing,
            %error,
            "declaration rejected; the standing declaration keeps this slug"
        );
    }
    result
}

fn register_decl(reg: Registration, source: DeclSource) -> Result<(), RegisterError> {
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
            return Ok(());
        }
    }
    // A built-in slug is claimable rather than reserved: a declaration is how
    // a built-in provider is written now, and the spec row it claims stays the
    // single source of truth for everything [`inherited`] lists. A slug maki
    // has no row for is still nobody's to claim -- there would be nothing to
    // inherit and the name would collide with a future built-in.
    let claimed = ProviderRegistry::get(&slug);
    // Only a slug maki does not already own can be lost to `providers.toml`.
    // An entry under a built-in slug has never defined a provider -- the
    // built-in keeps the slug and the entry overlays what
    // [`maki_config::providers::ignored_builtin_fields`] does not name, which
    // is how `[synthetic] base_url` points the shipped provider at a gateway.
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

    let target = match (decl.codec, &decl.base) {
        (Some(protocol), None) => Target::Codec(protocol),
        (None, Some(base)) => Target::Base(
            ProviderRegistry::get(base)
                .filter(|spec| spec.is_native())
                .ok_or_else(|| RegisterError::UnknownBase {
                    slug: slug.clone(),
                    base: base.clone(),
                })?,
        ),
        _ => return Err(RegisterError::CodecOrBase(slug)),
    };
    let unsupported = |option| RegisterError::Unsupported {
        slug: slug.clone(),
        option,
        target: target.describe(),
    };
    if hooks.build_body.is_some() && !honours_build_body(target) {
        return Err(unsupported(BUILD_BODY_OPTION));
    }
    if decl.system_prefix.is_some() && !honours_system_prefix(target) {
        return Err(unsupported(SYSTEM_PREFIX_OPTION));
    }

    let api_key_env = api_key_env(&decl, claimed);
    let mut staging = STAGING.lock().unwrap();
    let Some(staged) = staging.as_mut() else {
        return Err(RegisterError::Closed(slug));
    };
    // Credentials survive a reload because of *which map they live in*: a
    // get-or-insert by slug, so a reload cannot mint a second `RefreshGate` for
    // a slug whose token is in flight. What the new registration *declares* is
    // handed to the state either way, which is what keeps an edited
    // `api_key_env` from being silently ignored until the next restart.
    let fresh = Arc::new(AuthState::new(&slug, &decl.net_hosts)?);
    let auth = Arc::clone(
        AUTH.write()
            .unwrap()
            .entry(slug.as_str().into())
            .or_insert(fresh),
    );
    auth.redeclare(&slug, api_key_env, &decl.net_hosts)?;
    debug!(
        slug,
        ?source,
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
            target,
            hooks,
            auth,
        }),
    );
    Ok(())
}

/// The env var this declaration's key comes out of: its own, or the claimed
/// row's. Empty means the provider has no key env at all (Ollama's host,
/// Aperture's gateway), which is the same as declaring none.
/// Whether a `providers.toml` entry defines a provider of its own, as opposed
/// to only naming which of two declarations serves the slug. An `impl` line is
/// there to choose between them, so counting it as a definition would reject
/// both and leave the slug with nothing.
///
/// Asked of the serialised form rather than field by field: every field of
/// [`ProviderDef`] is skipped when unset, so one added later counts without
/// being listed here, and anything that does not serialise to an object counts
/// as a definition.
fn defines_provider(def: &ProviderDef) -> bool {
    let Ok(Value::Object(fields)) = serde_json::to_value(def) else {
        return true;
    };
    fields.keys().any(|field| field != IMPL_FIELD)
}

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
    let restated = |field| RegisterError::Restated {
        slug: decl.slug.clone(),
        field,
    };
    if decl.display_name.is_some() {
        return Err(restated(DISPLAY_NAME_FIELD));
    }
    if decl.api_key_env.is_some() {
        return Err(restated(API_KEY_ENV_FIELD));
    }
    // The curated table is the other half of the spec row's checklist, and a
    // claim reads it through [`PluginEntry::spec`].
    if !decl.models.is_empty() {
        return Err(restated(MODELS_FIELD));
    }
    Ok(())
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
    /// because [`AgentError`] is not `Clone` and `create` has to answer with
    /// it verbatim, every time, so the picker hides the provider instead of
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
/// refresh in flight. One cell per slug for the life of the process: the codec
/// reads it per request, so whatever the hook last wrote is what goes on the
/// wire, without anything being rebuilt or copied back.
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
    /// Starts with no credentials at all: the first [`Self::redeclare`] runs
    /// before anything can read this, and it is the one place a declaration
    /// becomes credentials.
    fn new(slug: &str, hosts: &[String]) -> Result<Self, RegisterError> {
        Ok(Self {
            current: Arc::new(Mutex::new(DeclaredKeys::Hooked.initial_auth(slug)?)),
            hosts: Mutex::new(hosts.into()),
            keys: Mutex::new(DeclaredKeys::Hooked),
            gate: RefreshGate::default(),
        })
    }

    fn hosts(&self) -> Arc<[String]> {
        Arc::clone(&self.hosts.lock().unwrap())
    }

    /// The pool `create` hands the provider for rotation, or the error the
    /// declared env var resolved to.
    ///
    /// Retried here rather than only at registration, because a built-in is
    /// constructed once per `create` and so picks up a key that appeared since
    /// -- a `maki auth login` in this very process -- without waiting for a
    /// reload.
    fn declared_pool(&self, slug: &str) -> Result<Option<KeyPool>, AgentError> {
        let mut keys = self.keys.lock().unwrap();
        if let DeclaredKeys::Missing { env_var, .. } = &*keys {
            *keys = DeclaredKeys::resolve(slug, Some(env_var.clone()));
            // Only over credentials nothing has minted: a token a hook
            // produced is newer than anything an env var can offer.
            if !self.gate.ran()
                && let Ok(declared) = keys.initial_auth(slug)
            {
                *self.current.lock().unwrap() = declared;
            }
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
    /// Three things change on a reload and none may be ignored: the declared
    /// hosts, which are what every later hook answer is vetted against, the
    /// declared key env var, and the static credentials, which are the only
    /// ones a plugin without an auth hook ever has. A token a hook already
    /// minted stays, unless the new declaration stopped covering the origin it
    /// is pointed at.
    fn redeclare(
        &self,
        slug: &str,
        api_key_env: Option<String>,
        hosts: &[String],
    ) -> Result<(), RegisterError> {
        *self.hosts.lock().unwrap() = hosts.into();
        let mut keys = self.keys.lock().unwrap();
        // Re-resolved only when the declaration moved: resolving reads the
        // credential file, and a reload that changed nothing must not walk a
        // rotating pool back to its first key.
        if keys.env_var() != api_key_env.as_deref() {
            *keys = DeclaredKeys::resolve(slug, api_key_env);
        }
        let declared = keys.initial_auth(slug)?;
        let mut current = self.current.lock().unwrap();
        let undeclared = declared_base_url(slug, current.base_url.clone(), hosts).is_err();
        if !self.gate.ran() || undeclared {
            *current = declared;
        }
        Ok(())
    }
}

impl PluginEntry {
    /// The spec row this entry answers inherited questions from: the built-in
    /// it claimed, else the native spec behind its target (model family,
    /// fallbacks, and the curated table a declaration that has none borrows).
    /// The one place a claim's inheritance lives, so no caller asks whether it
    /// is one.
    fn spec(&self) -> Option<&'static ProviderSpec> {
        self.claimed.or_else(|| self.target.spec())
    }

    fn display_name(&self) -> &str {
        self.decl
            .display_name
            .as_deref()
            .or(self.claimed.map(|spec| spec.display_name))
            // Registration refuses a declaration with neither, so this arm is
            // for the type and not for a real entry.
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
    /// Answers whether credentials were actually minted, which is not the same
    /// question as whether the call failed: a declaration with no auth hook
    /// succeeds here having changed nothing, and a caller that reads that as a
    /// refresh retries with the credentials it already had.
    ///
    /// Callable from the plugin host's own thread: the hook goes to the host's
    /// priority lane and the dispatch loop serves it while this future is
    /// parked, which is how a subagent driven from Lua refreshes a token. What
    /// may not reach here is a *blocking* caller, and that is held by
    /// construction instead of by a check: every path in is `async`, and
    /// `create` builds a provider without running a hook at all.
    async fn run_auth(&self, purpose: AuthPurpose) -> Result<bool, AgentError> {
        let Some(hook) = &self.hooks.auth else {
            return Ok(false);
        };
        // One question, asked once, because both answers follow from it: a
        // reload re-reads what a login wrote, so it spends no token. It waits
        // for neither the gate nor the cross-process lock, and it runs under
        // `block_on` on the ui thread, where either wait would freeze the ui.
        let spends_a_token = !matches!(purpose, AuthPurpose::Reload);
        let work = async {
            // Serialised against other maki processes on the same credentials,
            // and re-entrant in this one so the hook can store what it minted.
            let _lock = if spends_a_token {
                let slug = self.decl.slug.clone();
                smol::unblock(move || {
                    StateDir::resolve()
                        .ok()
                        .map(|dir| lock_credentials(&dir, &slug))
                })
                .await
            } else {
                None
            };
            let mut fresh = hook
                .call(purpose)
                .await?
                .into_resolved(&self.decl.slug, &self.auth.hosts())?;
            let mut guard = self.auth.current.lock().unwrap();
            // A hook that omits base_url keeps the resolved one; falling back
            // to the provider's default origin would silently repoint the token.
            if fresh.base_url.is_none() {
                fresh.base_url = guard.base_url.take();
            }
            *guard = fresh;
            Ok(())
        };
        if spends_a_token {
            self.auth.gate.single_flight(work).await?;
        } else {
            work.await?;
        }
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
        model: &'a Model,
        opts: RequestOptions,
    ) -> BoxFuture<'a, Result<Value, AgentError>> {
        self.0.call(BodyInput {
            body,
            model: model.id.clone(),
            thinking: opts.thinking.to_string(),
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
        // A claim never declares a table (it inherits the curated one), and
        // the inner provider it falls through to is built from the very codec
        // the built-in used, so asking it is asking the built-in.
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
                    // re-login prompt. Only a refresh that actually minted
                    // something earns the replay: a declaration whose key comes
                    // from an `api_key_env` has no hook to run, so retrying
                    // would re-send the very key the server just rejected and
                    // hand the caller the second 401 instead of the first.
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
}

/// Builds from registry data alone: no hook runs here, because this is called
/// from synchronous code and a hook means re-entering the plugin host.
///
/// A declaration that names an `api_key_env` and resolved no key fails here
/// rather than at the first request, which is what makes `provider_available`
/// tell the truth about it and the picker hide it. A declaration without one
/// gets its credentials from a hook and so cannot know yet; it stays lazy. The
/// rule keys off the declared field, never off who wrote the declaration.
pub fn create(slug: &str, timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    let entry = entry(slug).ok_or_else(|| unknown(slug))?;
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
/// static declaration must not. Provenance is carried by the two fields
/// themselves and needs no newtype and no carve-out: `resolved_base_url` is
/// read off the user's env and `providers.toml` inside the compat layer, where
/// nothing here can write it, and `auth.base_url` is only ever written through
/// [`PluginAuth::into_resolved`], the one door that vets an origin against the
/// declared `net_hosts`.
fn codec_options(entry: &PluginEntry, protocol: Protocol) -> CodecOptions {
    let decl = &entry.decl;
    CodecOptions {
        api_key_env: decl.api_key_env.clone().unwrap_or_default().into(),
        base_url: decl.base_url.clone().unwrap_or_default().into(),
        max_tokens_field: decl.max_tokens_field.clone().map(Into::into),
        include_stream_usage: decl.include_stream_usage,
        provider_name: entry.display_name().to_owned().into(),
        system_prefix: decl.system_prefix.clone(),
        thinking_dialect: decl.thinking_dialect,
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

pub fn base_for_slug(slug: &str) -> Option<&'static ProviderSpec> {
    entry(slug)?.spec()
}

/// The declaration maki itself authors for `slug`, which is not always the one
/// serving it: a Lua author's decl for the same slug outranks it unless
/// `impl = "rust"` says otherwise. Read straight off [`RUST_DECLS`], so it
/// answers whether or not a load has happened.
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
    Some(model.to_model(slug, entry.spec()?, model_id.to_string(), model.tier))
}

pub fn find_model_for_tier(slug: &str, tier: ModelTier) -> Option<Model> {
    let entry = entry(slug)?;
    let model = entry.decl.models.iter().find(|model| model.tier == tier)?;
    Some(model.to_model(slug, entry.spec()?, model.canonical_id()?.to_string(), tier))
}

pub fn plugin_model_specs_for(slug: &str) -> Vec<String> {
    let Some(entry) = entry(slug) else {
        return Vec::new();
    };
    if entry.decl.models.is_empty() {
        return entry
            .spec()
            .map(|spec| {
                spec.models()
                    .iter()
                    .flat_map(|entry| entry.prefixes.iter())
                    .map(|prefix| format!("{slug}/{prefix}"))
                    .collect()
            })
            .unwrap_or_default();
    }
    entry
        .decl
        .models
        .iter()
        .filter_map(PluginModel::canonical_id)
        .map(|id| format!("{slug}/{id}"))
        .collect()
}

pub fn registered_slugs() -> Vec<String> {
    entries().keys().map(|slug| slug.to_string()).collect()
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
    use test_case::test_case;

    use super::*;
    use crate::retry::RetryKind;
    use crate::test_support::{Canned, serve};

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
    const RUST_DECL_LOST: &str = "the rust decl must keep a slug no lua decl took";
    const LUA_DECL_LOST: &str = "a lua decl takes the slug off the rust one by default";
    const STANDING_RUST: &str = "standing=Rust";
    const SILENT_FALLBACK: &str = "a fallback must name the slug, the sources and the error";

    fn decl(slug: &str) -> ProviderDecl {
        ProviderDecl {
            slug: slug.to_string(),
            display_name: Some(DISPLAY_NAME.to_string()),
            codec: Some(Protocol::Openai),
            base: None,
            base_url: Some(EXAMPLE_BASE_URL.to_string()),
            api_key_env: None,
            system_prefix: None,
            max_tokens_field: None,
            include_stream_usage: None,
            thinking_dialect: None,
            models: vec![
                serde_json::from_value(serde_json::json!({
                    "prefixes": ["plug-1", "plug"],
                    "tier": "strong"
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
        begin_load();
        let result = register(reg, DeclSource::Lua);
        commit_load();
        result
    }

    struct CountingAuth {
        calls: AtomicUsize,
        rotating: bool,
        /// The origin this hook leases, which is the only way an origin ever
        /// reaches the auth cell now that a declared one is the codec's last
        /// resort instead.
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
        /// only hold for as long as `run_auth_hook` keeps an await ahead of the
        /// hook, which is an implementation detail of the caller, not of the
        /// gate under test.
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
        let auth = AuthState::new(SLUG, &decl.net_hosts).unwrap();
        auth.redeclare(SLUG, None, &decl.net_hosts).unwrap();
        Arc::new(PluginEntry {
            decl,
            claimed: None,
            source: DeclSource::Lua,
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
    fn a_new_load_drops_the_previous_load_s_entries() {
        const SLUG: &str = "stale-plugin";
        register_loaded(registration(SLUG)).unwrap();
        assert!(is_registered(SLUG));

        begin_load();
        assert!(is_registered(SLUG), "{RELOAD_KEEPS_SERVING}");
        commit_load();

        assert!(!is_registered(SLUG), "{RELOAD_DROPS_STALE}");
        assert!(
            AUTH.read().unwrap().contains_key(SLUG),
            "credentials outlive the load that registered them"
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
        assert_eq!(lookup_model(SLUG, "plug-1-mini").unwrap().id, "plug-1-mini");
        assert_eq!(
            find_model_for_tier(SLUG, ModelTier::Strong).unwrap().id,
            "plug-1"
        );
        assert_eq!(plugin_model_specs_for(SLUG), [format!("{SLUG}/plug-1")]);
        assert_eq!(
            auth_providers(),
            [(SLUG.to_string(), DISPLAY_NAME.to_string())]
        );
    }

    /// Availability follows the declaration and not who wrote it: a decl that
    /// names an `api_key_env` answers for its key, so `create` fails while
    /// there is none and the picker hides the provider rather than listing one
    /// that cannot serve a request. A key that appears later needs no reload.
    #[test]
    fn a_declared_key_env_decides_availability() {
        const SLUG: &str = "keyed-plugin";
        const ENV_VAR: &str = "MAKI_TEST_KEYED_PLUGIN_KEY";
        const KEY: &str = "sk-keyed";
        const LISTED_BUT_KEYLESS: &str = "a declared key env with no key must not build";
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
        const MODEL_ID: &str = "plug-1";
        const PROMPT: &str = "read a.txt";
        const UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"invalid api key"}}"#;
        const REJECTED_TWICE: &str = "a decl with no auth hook re-sent the rejected key";
        const STILL_AUTHORIZED: &str = "a 401 must reach the caller";
        /// Two answers so a replaying provider is recorded rather than parked
        /// on an `accept` that never returns.
        const SCRIPT: &[Canned] = &[
            Canned::json(401, UNAUTHORIZED_BODY),
            Canned::json(401, UNAUTHORIZED_BODY),
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

    /// A decl that claims a built-in slug states only what the row does not
    /// already answer, and every inherited answer comes off the row: the name,
    /// the curated table, and the spec behind it -- which here is *not* the
    /// one the declared codec would have resolved to.
    #[test]
    fn a_claim_inherits_the_spec_row() {
        let spec = ProviderRegistry::get(CLAIMED_SLUG).expect(NO_CLAIMED_SPEC);
        register_loaded(claim(CLAIMED_SLUG)).unwrap();

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
        const SLUG: &str = "dialect-plugin";
        const KNOWN: &str = "deepseek";
        const UNKNOWN: &str = "not-a-dialect";
        let resolved = thinking_dialect(SLUG, KNOWN).unwrap();
        assert_eq!(resolved, &crate::types::dialect::DEEPSEEK);

        let error = thinking_dialect(SLUG, UNKNOWN).unwrap_err().to_string();
        assert!(error.contains(UNKNOWN) && error.contains(KNOWN), "{error}");

        let mut decl = decl(SLUG);
        decl.thinking_dialect = Some(resolved);
        let dumped = serde_json::to_value(&decl).unwrap();
        assert_eq!(dumped["thinking_dialect"], serde_json::json!(KNOWN));
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

    fn claim_restating_the_model_table(reg: &mut Registration) {
        let models = std::mem::take(&mut reg.decl.models);
        *reg = claim(CLAIMED_SLUG);
        reg.decl.models = models;
    }

    /// A claim is a registration like any other: the checks a new slug passes
    /// are the same ones it passes. From the same author, as a collision is:
    /// two authors for one slug is [`wins`]'s question, not an error.
    fn claim_registered_twice(reg: &mut Registration) {
        *reg = claim(CLAIMED_SLUG);
        register(claim(CLAIMED_SLUG), DeclSource::Lua).unwrap();
    }

    fn claim_without_net_hosts(reg: &mut Registration) {
        *reg = claim(CLAIMED_SLUG);
        reg.decl.net_hosts.clear();
    }

    fn already_registered(reg: &mut Registration) {
        register(registration(&reg.decl.slug), DeclSource::Lua).unwrap();
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
        reg.decl.base_url = Some("https://evil.test/v1".to_string());
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
    #[test_case(claim_registered_twice, |e| matches!(e, RegisterError::DuplicateSlug(_)) ; "claim_duplicate")]
    #[test_case(claim_without_net_hosts, |e| matches!(e, RegisterError::NoNetHosts(_)) ; "claim_net_hosts_empty")]
    #[test_case(already_registered, |e| matches!(e, RegisterError::DuplicateSlug(_)) ; "duplicate")]
    #[test_case(codec_and_base, |e| matches!(e, RegisterError::CodecOrBase(_)) ; "codec_and_base_together")]
    #[test_case(missing_base, |e| matches!(e, RegisterError::UnknownBase { .. }) ; "unknown_base")]
    #[test_case(no_net_hosts, |e| matches!(e, RegisterError::NoNetHosts(_)) ; "net_hosts_empty")]
    #[test_case(outside_a_load, |e| matches!(e, RegisterError::Closed(_)) ; "registration_outside_a_load")]
    #[test_case(body_hook_on_anthropic, |e| matches!(e, RegisterError::Unsupported { .. }) ; "build_body_needs_an_openai_codec")]
    #[test_case(system_prefix_on_google, |e| matches!(e, RegisterError::Unsupported { .. }) ; "google_drops_the_system_prefix")]
    #[test_case(system_prefix_on_the_google_base, |e| matches!(e, RegisterError::Unsupported { .. }) ; "so_does_the_google_base")]
    #[test_case(base_url_off_the_declared_hosts, |e| matches!(e, RegisterError::UndeclaredBaseUrl(_)) ; "base_url_must_be_declared")]
    #[test_case(base_url_over_plain_http, |e| matches!(e, RegisterError::UndeclaredBaseUrl(_)) ; "base_url_must_be_https")]
    fn registration_rejects(mutate: fn(&mut Registration), expected: fn(&RegisterError) -> bool) {
        const SLUG: &str = "rejected-plugin";
        begin_load();
        let mut reg = registration(SLUG);
        mutate(&mut reg);
        let error = register(reg, DeclSource::Lua).unwrap_err();
        assert!(expected(&error), "{error}");
        commit_load();
    }

    /// What [`begin_load`] does with [`RUST_DECLS`], for a slug maki ships no
    /// declaration for: a Rust-authored decl staged before any plugin gets to
    /// speak.
    fn load_with_rust_decl(slug: &str) {
        begin_load();
        register(registration(slug), DeclSource::Rust).unwrap();
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

    #[test]
    fn a_slug_no_plugin_declares_keeps_its_rust_decl() {
        const SLUG: &str = "unclaimed-by-lua";
        load_with_rust_decl(SLUG);
        commit_load();

        assert_eq!(source_of(SLUG), Some(DeclSource::Rust), "{RUST_DECL_LOST}");
    }

    /// The default, and the reason the Lua authoring surface is the one the
    /// field exercises. A second decl from the *same* source is still a
    /// collision: it is the author that decides precedence, never the slug.
    #[test]
    fn a_lua_decl_takes_the_slug_and_a_second_one_collides() {
        const SLUG: &str = "declared-twice";
        load_with_rust_decl(SLUG);

        register(registration(SLUG), DeclSource::Lua).unwrap();
        let error = register(registration(SLUG), DeclSource::Lua).unwrap_err();
        commit_load();

        assert!(matches!(error, RegisterError::DuplicateSlug(_)), "{error}");
        assert_eq!(source_of(SLUG), Some(DeclSource::Lua), "{LUA_DECL_LOST}");
    }

    /// A broken Lua decl costs the plugin author an error and costs the user
    /// nothing: the Rust decl keeps serving the slug. Silently, that would be
    /// a provider behaving unlike the one whose file the user is reading, so
    /// the fallback names the slug, both sources and the load error.
    #[test]
    fn a_lua_decl_that_fails_leaves_the_rust_decl_standing() {
        const SLUG: &str = "broken-lua-decl";
        load_with_rust_decl(SLUG);

        let mut reg = registration(SLUG);
        no_net_hosts(&mut reg);
        let (error, logs) = capturing_warnings(|| register(reg, DeclSource::Lua).unwrap_err());
        commit_load();

        assert!(matches!(error, RegisterError::NoNetHosts(_)), "{error}");
        assert_eq!(source_of(SLUG), Some(DeclSource::Rust), "{RUST_DECL_LOST}");
        let message = error.to_string();
        for expected in [SLUG, STANDING_RUST, message.as_str()] {
            assert!(logs.contains(expected), "{SILENT_FALLBACK}: {logs}");
        }
    }

    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn capturing_warnings<T>(work: impl FnOnce() -> T) -> (T, String) {
        let logs = CapturedLogs::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logs.clone())
            .with_max_level(tracing::Level::WARN)
            .finish();
        let out = tracing::subscriber::with_default(subscriber, work);
        let written = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        (out, written)
    }

    fn api_error(status: u16, retry_after: Option<Duration>) -> AgentError {
        AgentError::Api {
            status,
            message: "upstream said no".to_string(),
            retry_after,
        }
    }

    #[test]
    fn map_error_absent_leaves_the_error_untouched() {
        let provider = provider_with(ProviderHooks::default());
        let error = smol::block_on(provider.mapped::<()>(Err(api_error(418, None)))).unwrap_err();
        assert!(matches!(error, AgentError::Api { status: 418, .. }));
        assert_eq!(error.retry_kind(), None);
    }

    /// The hook restates status and message; retryability and `Retry-After`
    /// stay maki's to decide.
    #[test]
    fn map_error_remaps_the_status_only() {
        const RETRY_AFTER: Duration = Duration::from_secs(7);
        const REMAPPED: &str = "slow down";
        let provider = provider_with(ProviderHooks {
            map_error: Some(Arc::new(RemapHook(Some(ApiError {
                status: 429,
                message: REMAPPED.to_string(),
            })))),
            ..ProviderHooks::default()
        });

        let error = smol::block_on(provider.mapped::<()>(Err(api_error(400, Some(RETRY_AFTER)))))
            .unwrap_err();

        assert!(
            matches!(&error, AgentError::Api { status: 429, message, .. } if message == REMAPPED)
        );
        assert_eq!(error.retry_kind(), Some(RetryKind::RateLimit));
        assert_eq!(error.retry_after(), Some(RETRY_AFTER));
    }

    #[test]
    fn plugin_auth_rejects_an_undeclared_base_url() {
        const SLUG: &str = "egress-plugin";
        let hosts = vec![EXAMPLE_HOST.to_string()];
        let auth = |base_url: &str| PluginAuth {
            base_url: Some(base_url.to_string()),
            headers: HashMap::new(),
        };

        assert!(
            auth("https://evil.test/v1")
                .into_resolved(SLUG, &hosts)
                .is_err()
        );
        assert_eq!(
            auth(&format!("https://{EXAMPLE_HOST}/v1"))
                .into_resolved(SLUG, &hosts)
                .unwrap()
                .base_url
                .as_deref(),
            Some("https://example.com/v1")
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
        let hosts = vec![host.to_string()];
        assert_eq!(
            declared_base_url("scheme-plugin", Some(url.to_string()), &hosts).is_ok(),
            accepted,
            "{url}"
        );
    }

    /// What the append-only auth map must *not* cost: a reload re-reads the
    /// registration, so a provider that has minted nothing takes the fresh
    /// declaration, while an origin a hook leased is left alone as long as the
    /// new declaration still covers it.
    ///
    /// The origin is the hook's, not the declaration's: a declared `base_url`
    /// is the codec's last resort now and never reaches the auth cell, which
    /// is what keeps a plugin from outranking the user's `<SLUG>_BASE_URL`.
    #[test_case(false, None ; "an_unminted_slug_takes_the_fresh_declaration")]
    #[test_case(true, Some(OTHER_BASE_URL) ; "a_leased_origin_survives_the_reload")]
    fn a_reload_re_reads_the_declaration(minted: bool, expected: Option<&str>) {
        const SLUG: &str = "redeclare-plugin";
        let hook = Arc::new(CountingAuth::leasing(OTHER_BASE_URL));
        let declaration = || {
            let mut reg = registration(SLUG);
            reg.decl.net_hosts.push(OTHER_HOST.to_string());
            reg
        };

        let mut first = declaration();
        if minted {
            first.hooks.auth = Some(hook.clone());
        }
        register_loaded(first).unwrap();
        if minted {
            smol::block_on(entry(SLUG).unwrap().ensure_auth()).unwrap();
        }
        register_loaded(declaration()).unwrap();

        let auth = entry(SLUG).unwrap().auth.current.lock().unwrap().clone();
        assert_eq!(auth.base_url.as_deref(), expected);
    }

    /// Credentials that stopped satisfying the declaration are not kept: a
    /// reload that narrows the host list drops an origin it no longer covers,
    /// rather than carrying the token to a host nobody declared any more.
    #[test]
    fn narrowing_the_declared_hosts_drops_an_origin_it_no_longer_covers() {
        const SLUG: &str = "narrowed-plugin";
        let mut wide = registration(SLUG);
        wide.decl.net_hosts.push(OTHER_HOST.to_string());
        wide.hooks.auth = Some(Arc::new(CountingAuth::leasing(OTHER_BASE_URL)));
        register_loaded(wide).unwrap();
        smol::block_on(entry(SLUG).unwrap().ensure_auth()).unwrap();
        assert_eq!(
            entry(SLUG).unwrap().auth.current.lock().unwrap().base_url,
            Some(OTHER_BASE_URL.to_string())
        );

        register_loaded(registration(SLUG)).unwrap();

        let auth = entry(SLUG).unwrap().auth.current.lock().unwrap().clone();
        assert_eq!(auth.base_url, None);
    }
}
