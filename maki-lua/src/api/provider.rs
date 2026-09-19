use std::collections::HashMap;
use std::fmt::Display;
use std::io::{BufRead, Read, Write};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use maki_agent::cancel::CancelToken;
use maki_config::providers::Protocol;
use maki_lua_macro::{lua_fn, lua_table};
use maki_providers::plugin::{
    self, DeclAuthority, Hook, PluginModel, ProviderDecl, ProviderHooks, Registered, Registration,
};
use maki_providers::provider::BoxFuture;
use maki_providers::{AgentError, EffortDialect};
use maki_storage::StateDir;
use maki_storage::auth::{
    delete_plugin_auth, load_plugin_auth, lock_credentials, save_plugin_auth,
};
use mlua::{
    Function, Lua, LuaSerdeExt, MultiValue, RegistryKey, Result as LuaResult, Table,
    Value as LuaValue,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::api::util::convert::{json_to_lua, lua_to_json, lua_to_json_within};
use crate::api::util::pair::{Pair, err_pair, try_pair};
use crate::plugin_permissions::{NetEgress, OwnedSlugs, Permission, PluginPermissions};
use crate::runtime::{DeferredCallback, Request, host_senders};

/// How long the host waits for a hook that runs between turns. Generous,
/// because an auth hook may be talking to an identity provider, but bounded:
/// a plugin that never answers must not park the model behind it forever.
const HOOK_TIMEOUT: Duration = Duration::from_secs(30);
/// The budget for hooks that sit between the user and their first token.
/// Anything slower here is felt as latency on every single request.
const REQUEST_HOOK_TIMEOUT: Duration = Duration::from_secs(5);

const RESOLVE_AUTH: &str = "resolve_auth";
const REFRESH_AUTH: &str = "refresh_auth";
const RELOAD_AUTH: &str = "reload_auth";
const LIST_MODELS: &str = "list_models";
const BUILD_BODY: &str = "build_body";
const MAP_ERROR: &str = "map_error";
const FETCH_USAGE: &str = "fetch_usage";
const LOGIN: &str = "login";
const LOGOUT: &str = "logout";

const SLUG: &str = "slug";
const DISPLAY_NAME: &str = "display_name";
const CODEC: &str = "codec";
const BASE: &str = "base";
const BASE_URL: &str = "base_url";
const API_KEY_ENV: &str = "api_key_env";
const SYSTEM_PREFIX: &str = "system_prefix";
const MAX_TOKENS_FIELD: &str = "max_tokens_field";
const INCLUDE_STREAM_USAGE: &str = "include_stream_usage";
const THINKING_DIALECT: &str = "thinking_dialect";
const MODELS: &str = "models";
const HEADERS: &str = "headers";

const BODY_FIELD: &str = "body";
const MODEL_FIELD: &str = "model";
const STATUS_FIELD: &str = "status";
const MESSAGE_FIELD: &str = "message";

const REGISTER: &str = "maki.provider.register";
const NO_NET_HOSTS: &str = "declare the hosts this provider talks to as `net_hosts` under \
     `[permissions]` in plugin.toml before registering";
const API_KEY_ENV_NEEDS_ENV: &str = "`api_key_env` reads the environment, which needs `env = true` \
     under `[permissions]` in plugin.toml";
const SECRET_MASK: char = '*';

/// One plugin-supplied provider callback, named the way the registry names it.
///
/// A slot is not the same thing as a Lua entry: `Auth` is one hook the registry
/// calls with a purpose, and three entries a plugin may write.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HookSlot {
    Auth,
    ListModels,
    BuildBody,
    MapError,
    FetchUsage,
    Login,
    Logout,
}

/// Everything that differs between hooks, one row each, so adding a hook is a
/// row and a call site rather than a branch anywhere in the bridge.
struct SlotSpec {
    /// Lua entries that can serve the slot, each with the serialized payload
    /// that selects it. Only auth has more than one, and `""` marks the entry
    /// every payload falls back to.
    entries: &'static [(&'static str, &'static str)],
    /// Payload fields handed over as positional arguments, in order. Whatever
    /// the payload holds beyond them arrives as a trailing table, so the Lua
    /// signature an author writes does not have to track the wire shape.
    positional: &'static [&'static str],
    /// Whether the call is handed a `ctx` that can talk to the terminal.
    ctx: bool,
    /// The payload field the answer is a *refinement* of, when the hook edits
    /// something maki handed it rather than producing something new.
    ///
    /// A JSON null crosses into Lua as a nil, and a nil key is an absent key,
    /// so a layer that never touched a null-valued field would otherwise
    /// delete it on the way back. Naming the original here is what lets
    /// [`lua_to_json_within`] put it back. Deleting a key holding a real value
    /// still works, which is the only thing a hook could have meant by it.
    refines: Option<&'static str>,
}

impl HookSlot {
    /// Every slot, so the Lua entry names live in exactly one table: the one
    /// [`Self::spec`] holds.
    const ALL: [Self; 7] = [
        Self::Auth,
        Self::ListModels,
        Self::BuildBody,
        Self::MapError,
        Self::FetchUsage,
        Self::Login,
        Self::Logout,
    ];

    const fn spec(self) -> SlotSpec {
        match self {
            Self::Auth => SlotSpec {
                entries: &[
                    ("resolve", RESOLVE_AUTH),
                    ("refresh", REFRESH_AUTH),
                    ("reload", RELOAD_AUTH),
                ],
                positional: &[],
                ctx: false,
                refines: None,
            },
            Self::ListModels => SlotSpec {
                entries: &[("", LIST_MODELS)],
                positional: &[],
                ctx: false,
                refines: None,
            },
            Self::BuildBody => SlotSpec {
                entries: &[("", BUILD_BODY)],
                positional: &[BODY_FIELD, MODEL_FIELD],
                ctx: false,
                refines: Some(BODY_FIELD),
            },
            Self::MapError => SlotSpec {
                entries: &[("", MAP_ERROR)],
                positional: &[STATUS_FIELD, MESSAGE_FIELD],
                ctx: false,
                refines: None,
            },
            Self::FetchUsage => SlotSpec {
                entries: &[("", FETCH_USAGE)],
                positional: &[],
                ctx: false,
                refines: None,
            },
            Self::Login => SlotSpec {
                entries: &[("", LOGIN)],
                positional: &[],
                ctx: true,
                refines: None,
            },
            Self::Logout => SlotSpec {
                entries: &[("", LOGOUT)],
                positional: &[],
                ctx: true,
                refines: None,
            },
        }
    }
}

/// The Lua functions one registration handed over, plus the way back to the
/// thread that may call them.
///
/// Held by `Arc` from every hook the registration produced, so a call that
/// started before an unload finishes against the functions it started with:
/// the plugin's environment goes away, these registry entries do not.
pub struct LuaHookKeys {
    plugin: Arc<str>,
    slug: String,
    keys: HashMap<&'static str, RegistryKey>,
    requests: flume::Sender<Request>,
    release: flume::Sender<DeferredCallback>,
}

impl LuaHookKeys {
    /// The entry that serves this call: the one the payload names, else the
    /// first the plugin supplied, in the order [`HookSlot::spec`] lists them.
    /// A plugin that writes only `resolve_auth` gets it for refreshes and
    /// reloads too, which is right for one that reads its credentials fresh
    /// every time, and one that writes only `refresh_auth` gets that for the
    /// initial resolve rather than no way to mint credentials at all.
    fn entry(&self, slot: HookSlot, payload: &Value) -> Option<&RegistryKey> {
        let entries = slot.spec().entries;
        let selector = payload.as_str().unwrap_or_default();
        entries
            .iter()
            .find(|(select, _)| *select == selector)
            .and_then(|(_, entry)| self.keys.get(entry))
            .or_else(|| entries.iter().find_map(|(_, entry)| self.keys.get(entry)))
    }

    fn serves(&self, slot: HookSlot) -> bool {
        slot.spec()
            .entries
            .iter()
            .any(|(_, entry)| self.keys.contains_key(entry))
    }
}

impl Drop for LuaHookKeys {
    /// A registry key may only be released on the Lua thread, and this drop
    /// runs wherever the last hook handle happened to die. The defer queue is
    /// the existing way back there; a callback handed over already cancelled is
    /// the dispatcher's path for releasing a key without running it.
    fn drop(&mut self) {
        for (_, func) in std::mem::take(&mut self.keys) {
            let _ = self.release.send(DeferredCallback {
                func,
                delay: Duration::ZERO,
                plugin: Arc::clone(&self.plugin),
                cancel: Arc::new(AtomicBool::new(true)),
            });
        }
    }
}

/// One hook, as the registry sees it: input in, output out, both checked by the
/// compiler against the slot's declared pair.
pub struct LuaHook<In, Out> {
    keys: Arc<LuaHookKeys>,
    slot: HookSlot,
    /// How long to wait for an answer, or `None` for the hooks that wait on a
    /// person rather than on a server.
    timeout: Option<Duration>,
    _types: PhantomData<fn(In) -> Out>,
}

impl<In, Out> Hook<In, Out> for LuaHook<In, Out>
where
    In: Serialize + Send + 'static,
    Out: DeserializeOwned + Send + 'static,
{
    fn call(&self, input: In) -> BoxFuture<'_, Result<Out, AgentError>> {
        Box::pin(async move {
            let payload = serde_json::to_value(input)?;
            let (reply, answer) = flume::bounded(1);
            // Fires when this future goes, answered or not: a caller that timed
            // out or was cancelled takes the hook down with it.
            let (_abandon, cancel) = CancelToken::new();
            self.keys
                .requests
                .send(Request::CallProviderHook {
                    hook: Arc::clone(&self.keys),
                    slot: self.slot,
                    payload,
                    cancel,
                    deadline: self.timeout.map(|limit| Instant::now() + limit),
                    answer: Box::new(move |lua, returned| {
                        let _ = reply.send(returned.and_then(|value| decode::<Out>(lua, value)));
                    }),
                })
                .map_err(|_| AgentError::Channel)?;
            let answered = async { answer.recv_async().await.map_err(|_| AgentError::Channel) };
            let value = match self.timeout {
                Some(limit) => {
                    futures_lite::future::or(answered, async {
                        smol::Timer::after(limit).await;
                        Err(self.timed_out(limit))
                    })
                    .await?
                }
                None => answered.await?,
            };
            value.map_err(|message| AgentError::Config {
                message: format!(
                    "provider '{}': {:?} hook failed: {message}",
                    self.keys.slug, self.slot
                ),
            })
        })
    }
}

/// What a hook call hands back, on the Lua thread: decodes the hook's return
/// value into the type the caller asked for and sends it off.
pub(crate) type HookAnswer = Box<dyn FnOnce(&Lua, Result<HookReturn, String>) + Send>;

/// A hook's return value, still a Lua value, plus the JSON it refines when the
/// slot hands a document to the hook and expects it back.
pub(crate) struct HookReturn {
    value: LuaValue,
    template: Option<Value>,
}

/// Decoded straight into `Out` while it is still a Lua value, because only the
/// target type knows whether an empty table is an empty list or an empty
/// object. A detour through JSON has to guess, and a guess of `{}` fails every
/// hook that returns a list that happens to be empty.
///
/// A returned document is the exception: it is refined against the one that
/// went in, which is what keeps the fields the hook never touched.
fn decode<Out: DeserializeOwned>(lua: &Lua, returned: HookReturn) -> Result<Out, String> {
    match &returned.template {
        Some(template) => lua_to_json_within(lua, &returned.value, template)
            .map_err(|e| e.to_string())
            .and_then(|json| serde_json::from_value(json).map_err(|e| e.to_string())),
        None => lua.from_value(returned.value).map_err(|e| e.to_string()),
    }
}

impl<In, Out> LuaHook<In, Out> {
    fn timed_out(&self, limit: Duration) -> AgentError {
        AgentError::Config {
            message: format!(
                "provider '{}': {:?} hook did not answer within {}s",
                self.keys.slug,
                self.slot,
                limit.as_secs()
            ),
        }
    }
}

/// The one place a hook handle is made. `In` and `Out` come from the field it
/// is assigned to, so the registry's idea of a hook's shape and the bridge's
/// are the same idea.
fn hook<In, Out>(
    keys: &Arc<LuaHookKeys>,
    slot: HookSlot,
    timeout: Option<Duration>,
) -> Option<Arc<dyn Hook<In, Out>>>
where
    In: Serialize + Send + 'static,
    Out: DeserializeOwned + Send + 'static,
{
    keys.serves(slot).then(|| {
        Arc::new(LuaHook {
            keys: Arc::clone(keys),
            slot,
            timeout,
            _types: PhantomData,
        }) as Arc<dyn Hook<In, Out>>
    })
}

/// Runs one hook call on the Lua thread, under the scope its caller set up:
/// JSON in, Lua arguments, the Lua return value back out for [`decode`].
pub(crate) async fn run_hook(
    lua: &Lua,
    keys: &LuaHookKeys,
    slot: HookSlot,
    payload: Value,
) -> Result<HookReturn, String> {
    let Some(key) = keys.entry(slot, &payload) else {
        return Err(format!("no lua function is registered for {slot:?}"));
    };
    let func: Function = lua.registry_value(key).map_err(|e| e.to_string())?;
    let spec = slot.spec();
    let template = spec.refines.and_then(|field| payload.get(field).cloned());
    let args = call_args(lua, spec, payload).map_err(|e| e.to_string())?;
    let call = async { lua.create_thread(func)?.into_async::<LuaValue>(args)?.await };
    let value = call.await.map_err(|e| e.to_string())?;
    Ok(HookReturn { value, template })
}

fn call_args(lua: &Lua, spec: SlotSpec, payload: Value) -> LuaResult<MultiValue> {
    let mut args = Vec::new();
    if spec.ctx {
        args.push(LuaValue::Table(stdio_ctx(lua)?));
    }
    match payload {
        Value::Null => {}
        Value::Object(mut fields) if !spec.positional.is_empty() => {
            for name in spec.positional {
                let field = fields.remove(*name).unwrap_or(Value::Null);
                args.push(json_to_lua(lua, &field)?);
            }
            args.push(json_to_lua(lua, &Value::Object(fields))?);
        }
        other => args.push(json_to_lua(lua, &other)?),
    }
    Ok(MultiValue::from_vec(args))
}

/// The `ctx` a login or logout gets. Stdio, because the one caller is the cli,
/// which is exactly what the retired subprocess providers got by inheriting it.
fn stdio_ctx(lua: &Lua) -> LuaResult<Table> {
    let ctx = lua.create_table()?;
    ctx.set(
        "print",
        lua.create_function(|_, text: String| {
            println!("{text}");
            Ok(())
        })?,
    )?;
    ctx.set(
        "prompt",
        lua.create_function(|_, opts: Table| {
            let label: String = opts.get("label").unwrap_or_default();
            let secret = opts.get::<Option<bool>>("secret")?.unwrap_or(false);
            Ok(read_answer(&label, secret))
        })?,
    )?;
    ctx.set(
        "open_url",
        lua.create_function(|_, url: String| -> LuaResult<Pair<bool>> {
            try_pair!(open::that(&url));
            Ok((Some(true), None))
        })?,
    )?;
    Ok(ctx)
}

/// Reads one line of an answer from the terminal.
///
/// A secret is read with the terminal in raw mode so the characters never reach
/// the scrollback, and echoed as mask characters so there is still feedback
/// that a key landed. Raw mode is given back whatever the read did.
fn read_answer(label: &str, secret: bool) -> Pair<String> {
    print!("{label}");
    if std::io::stdout().flush().is_err() {
        return (None, Some("cannot write to the terminal".to_owned()));
    }
    if !secret {
        let mut line = String::new();
        return match std::io::stdin().lock().read_line(&mut line) {
            Ok(_) => (Some(line.trim_end().to_owned()), None),
            Err(e) => (None, Some(e.to_string())),
        };
    }
    if let Err(e) = crossterm::terminal::enable_raw_mode() {
        return (None, Some(e.to_string()));
    }
    let answer = read_masked();
    let _ = crossterm::terminal::disable_raw_mode();
    println!();
    answer
}

/// Raw mode hands over bytes, so the answer is gathered as bytes and decoded
/// once at the end. Taking each one for a char would turn a typed accented
/// letter into two wrong ones.
fn read_masked() -> Pair<String> {
    const BACKSPACE: u8 = 0x7f;
    const CTRL_C: u8 = 0x03;
    let mut answer = Vec::new();
    for byte in std::io::stdin().lock().bytes() {
        match byte {
            Ok(b'\r' | b'\n') => break,
            Ok(CTRL_C) => return (None, Some("cancelled".to_owned())),
            Ok(BACKSPACE) => {
                if answer.pop().is_some() {
                    print!("\u{8} \u{8}");
                }
            }
            Ok(byte) => {
                answer.push(byte);
                print!("{SECRET_MASK}");
            }
            Err(e) => return (None, Some(e.to_string())),
        }
        let _ = std::io::stdout().flush();
    }
    (Some(String::from_utf8_lossy(&answer).into_owned()), None)
}

fn owned(slugs: &OwnedSlugs, slug: &str) -> LuaResult<()> {
    if slugs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|owned| owned == slug)
    {
        return Ok(());
    }
    Err(mlua::Error::runtime(format!(
        "maki.provider.auth: '{slug}' is not a provider this plugin registered"
    )))
}

/// Register a provider this plugin implements. maki then treats its models like
/// any other provider's: they appear in the model picker, in `/model`, and in
/// `providers.toml` overrides, addressed as `<slug>/<model>`.
///
/// The plugin must declare the hosts it talks to as `net_hosts` under
/// `[permissions]` in its `plugin.toml`. That list is what maki will send this
/// provider's credentials to, whatever a hook returns later. The one origin it
/// need not name is the one the user chose: a slug pointed at a gateway with
/// `<SLUG>_BASE_URL` or `providers.toml` is reachable from this provider's
/// hooks, since its requests already go there. Registering with no declared
/// host fails.
///
/// Give exactly one of `codec` (speak a wire protocol maki already knows) or
/// `base` (borrow a native provider whole, including its quirks and its model
/// table). `codec` is the supported choice for a new provider. `base` exists
/// for a provider that needs a specific vendor adapter, and what it inherits
/// changes whenever that provider does.
///
/// Every callback is optional, and a registration with none is a perfectly
/// good static provider. Callbacks run on maki's plugin host, so they may use
/// `maki.net`, `maki.fs` and the rest of the API.
///
/// An option the target cannot honour fails at registration rather than being
/// ignored at request time: `build_body` needs one of the `openai` codecs, and
/// `system_prefix` is refused by the `google` codec, which drops it.
///
/// {spec} fields:
///   `slug` (string) Required. How the provider is addressed: `<slug>/<model>`.
///           Letters, digits, `_` and `-`, starting with a letter or digit,
///           and neither a slug `providers.toml` already owns nor one of a
///           built-in provider. A built-in slug is reserved: claiming it
///           inherits that provider's `api_key_env`, which would hand a plugin
///           the key the user set for the built-in. Only the plugins maki
///           ships inside the binary may take one, and they inherit
///           `display_name`, `api_key_env`, the curated model table and its
///           pricing, so restating any of those is an error rather than an
///           override.
///   `display_name` (string) Required. Shown in the UI.
///   `codec` (string) `"openai"`, `"openai-responses"`, `"anthropic"` or
///           `"google"`. Mutually exclusive with `base`.
///   `base` (string) A native provider slug to build on, e.g. `"anthropic"`.
///   `base_url` (string) Fallback origin for requests. `<SLUG>_BASE_URL` and
///           `providers.toml` outrank it, and an origin an auth hook returns
///           outranks those. Its host must be one of the declared `net_hosts`,
///           and it must be `https` unless it points at loopback.
///   `api_key_env` (string) Environment variable holding an API key, sent as a
///           bearer token when set. Needs the `env` permission: resolving it
///           reads the environment, and the key saved for the slug.
///           Re-read every time maki builds the provider, so a key set or
///           replaced since is picked up.
///   `system_prefix` (string) Text prepended to the system prompt.
///   `max_tokens_field` (string) Body field carrying the output cap. Defaults
///           to `max_tokens`.
///   `include_stream_usage` (boolean) Whether to ask for usage on the stream.
///           Defaults to `true`.
///   `thinking_dialect` (string) Names the provider's effort dialect, one of
///           `"standard"`, `"codex"`, `"codex-5-1"`, `"coding-plan"`,
///           `"gpt-5-6"`, `"gpt-6"`, `"prefer-high"`, `"high-only"`, `"glm"`,
///           `"deepseek"`, `"anthropic-adaptive"`, `"tensorx"`, `"grok"` or
///           `"ollama"`. Omitting it sends no effort field.
///   `models` (table) List of model rows. Each row has `prefixes` (list): the
///            row answers for every model id starting with one of them,
///            longest prefix first, and `prefixes[1]` is the canonical id.
///            Also `tier` (`"weak"`, `"medium"`, `"strong"` or
///            `"compaction"`, default `"medium"`), `context_window`,
///            `max_output_tokens`, `supports_thinking`, `supports_vision`,
///            `supports_tool_examples`, `requires_thinking`, `pricing`
///            (`input`, `output`, `cache_write`, `cache_read`, in dollars per
///            million tokens) and `thinking_fields`. The three `supports_`
///            flags have three states: leaving one out asks the codec or base
///            provider, `false` turns the feature off for that model. Read
///            once, here, so this table must not depend on anything asked at
///            runtime.
///   `resolve_auth` (function) `function(purpose)` returning
///            `{ base_url = ..., headers = { ... } }`. Called once, lazily,
///            before the first request, so a provider whose credentials are
///            broken is still listed and fails when it is used. `purpose` is
///            `"resolve"`. Omitting `base_url` keeps the one in force.
///   `refresh_auth` (function) Same shape, called after a 401 with
///            `purpose = "refresh"`.
///   `reload_auth` (function) Same shape, called with `purpose = "reload"` to
///            re-read what a `login` wrote.
///            The three are one hook with three entry points: a purpose runs
///            the entry named for it, and falls back to the first of the three
///            the plugin supplied. Writing only `resolve_auth` therefore
///            serves all three, which is right for a plugin that reads its
///            credentials fresh every time.
///   `list_models` (function) `function()` returning a list of model rows,
///            for a provider whose catalogue is only known at runtime. Rows
///            carry `id`, `context_window`, `max_output_tokens`, `pricing`,
///            `supports_thinking`, `supports_vision` and `tier`.
///   `build_body` (function) `function(body, model, opts)` returning the
///            request body to send. `opts.thinking` is the effort level as
///            rendered. Only for the `openai` codecs.
///   `map_error` (function) `function(status, message)` returning
///            `{ status = ..., message = ... }`, or nil to keep the original.
///            Those two fields are all it may change: `retry_after` comes from
///            the response header, and whether an error is retryable is
///            derived from the status.
///   `fetch_usage` (function) `function()` returning a usage summary.
///   `login` (function) `function(ctx)`. Having one is what makes the provider
///            an auth target: it then shows up in `maki auth login`. There is
///            no separate flag for it. `ctx` is a table of functions, called
///            with a dot: `ctx.print(text)`,
///            `ctx.prompt({ label = ..., secret = ... })` and
///            `ctx.open_url(url)`.
///   `logout` (function) `function(ctx)`, the `maki auth logout` side.
///
/// @param spec table Provider specification (see above).
/// @return
/// @example
/// maki.provider.register({
///   slug = "acme",
///   display_name = "Acme",
///   codec = "openai",
///   base_url = "https://api.acme.com/v1",
///   api_key_env = "ACME_API_KEY",
///   models = {
///     { prefixes = { "acme-large" }, tier = "strong", context_window = 200000 },
///   },
///   resolve_auth = function()
///     local token = maki.provider.auth.get("acme")
///     return { headers = { Authorization = "Bearer " .. token.access } }
///   end,
/// })
#[lua_fn(guard = Net)]
fn register(
    lua: &Lua,
    #[ctx] plugin: Arc<str>,
    #[ctx] egress: NetEgress,
    #[ctx] authority: DeclAuthority,
    #[ctx] reads_env: bool,
    spec: Table,
) -> LuaResult<()> {
    let api_key_env = optional(&spec, API_KEY_ENV)?;
    if api_key_env.is_some() && !reads_env {
        return Err(register_error(API_KEY_ENV_NEEDS_ENV));
    }
    let net_hosts = egress
        .declared()
        .as_deref()
        .filter(|hosts| !hosts.is_empty())
        .ok_or_else(|| register_error(NO_NET_HOSTS))?
        .to_vec();
    let slug: String = field(&spec, SLUG)?;
    let (requests, release) = host_senders(lua)?;
    let keys = Arc::new(LuaHookKeys {
        plugin: Arc::clone(&plugin),
        slug: slug.clone(),
        keys: hook_keys(lua, &spec)?,
        requests,
        release,
    });

    let registered = plugin::register_plugin(
        plugin,
        Registration {
            decl: ProviderDecl {
                slug: slug.clone(),
                display_name: optional(&spec, DISPLAY_NAME)?,
                codec: codec(&spec)?,
                base: optional(&spec, BASE)?,
                base_url: optional(&spec, BASE_URL)?,
                api_key_env,
                system_prefix: optional(&spec, SYSTEM_PREFIX)?,
                max_tokens_field: optional(&spec, MAX_TOKENS_FIELD)?,
                include_stream_usage: optional_bool(&spec, INCLUDE_STREAM_USAGE)?,
                thinking_dialect: dialect(&spec, &slug)?,
                models: models(lua, &spec)?,
                net_hosts,
            },
            hooks: ProviderHooks {
                auth: hook(&keys, HookSlot::Auth, Some(HOOK_TIMEOUT)),
                list_models: hook(&keys, HookSlot::ListModels, Some(HOOK_TIMEOUT)),
                build_body: hook(&keys, HookSlot::BuildBody, Some(REQUEST_HOOK_TIMEOUT)),
                map_error: hook(&keys, HookSlot::MapError, Some(REQUEST_HOOK_TIMEOUT)),
                fetch_usage: hook(&keys, HookSlot::FetchUsage, Some(HOOK_TIMEOUT)),
                login: hook(&keys, HookSlot::Login, None),
                logout: hook(&keys, HookSlot::Logout, None),
            },
        },
        authority,
    )
    .map_err(|e| mlua::Error::runtime(e.to_string()))?;

    // Only the declaration that actually serves the slug gets what registering
    // grants: `maki.provider.auth` on it, and `maki.net` reach to the origin
    // its requests go to. A shadowed declaration is well formed but unused, and
    // handing it live credentials would let a plugin read the token of a
    // provider somebody else is serving.
    if registered == Registered::Serving {
        egress.owns(slug);
    } else {
        tracing::debug!(
            %slug,
            plugin = %keys.plugin,
            "another declaration serves this slug; the plugin gets no access to it"
        );
    }
    Ok(())
}

fn register_error(message: impl Display) -> mlua::Error {
    mlua::Error::runtime(format!("{REGISTER}: {message}"))
}

fn must_be(key: &str, kind: &str) -> mlua::Error {
    register_error(format!("'{key}' must be a {kind}"))
}

fn field(spec: &Table, key: &str) -> LuaResult<String> {
    optional(spec, key)?.ok_or_else(|| must_be(key, "string"))
}

fn optional(spec: &Table, key: &str) -> LuaResult<Option<String>> {
    spec.get::<Option<String>>(key)
        .map_err(|_| must_be(key, "string"))
}

fn optional_bool(spec: &Table, key: &str) -> LuaResult<Option<bool>> {
    spec.get::<Option<bool>>(key)
        .map_err(|_| must_be(key, "boolean"))
}

/// The registry holds protocols, not names, and has no parser: a codec nobody
/// implements is caught here, where the plugin that wrote it can be named.
fn codec(spec: &Table) -> LuaResult<Option<Protocol>> {
    optional(spec, CODEC)?
        .map(|name| {
            name.parse::<Protocol>().map_err(|_| {
                register_error(format!(
                    "unknown codec '{name}' (expected one of openai, openai-responses, \
                     anthropic, google)"
                ))
            })
        })
        .transpose()
}

/// Same story as [`codec`]: the dialect table lives in the registry, so a name
/// nobody implements is caught here, where the plugin that wrote it can be
/// named.
fn dialect(spec: &Table, slug: &str) -> LuaResult<Option<&'static EffortDialect<'static>>> {
    optional(spec, THINKING_DIALECT)?
        .map(|name| plugin::thinking_dialect(slug, &name).map_err(register_error))
        .transpose()
}

/// Model rows are static data, read here and never again, so a provider's
/// catalogue costs nothing per request.
fn models(lua: &Lua, spec: &Table) -> LuaResult<Vec<PluginModel>> {
    let Some(table) = spec
        .get::<Option<Table>>(MODELS)
        .map_err(|_| must_be(MODELS, "list of tables"))?
    else {
        return Ok(Vec::new());
    };
    let json = lua_to_json(lua, &LuaValue::Table(table))?;
    // An empty Lua table reads as an object, and a plugin that curates no models
    // writes one rather than leaving the key out.
    if json.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(Vec::new());
    }
    serde_json::from_value(json).map_err(|e| register_error(format!("invalid 'models' entry: {e}")))
}

fn hook_keys(lua: &Lua, spec: &Table) -> LuaResult<HashMap<&'static str, RegistryKey>> {
    let mut keys = HashMap::new();
    for entry in HookSlot::ALL
        .into_iter()
        .flat_map(|slot| slot.spec().entries)
        .map(|(_, entry)| *entry)
    {
        let Some(func) = spec
            .get::<Option<Function>>(entry)
            .map_err(|_| must_be(entry, "function"))?
        else {
            continue;
        };
        keys.insert(entry, lua.create_registry_value(func)?);
    }
    Ok(keys)
}

/// Read the credentials this plugin stored for one of its providers.
///
/// The value is whatever the plugin wrote. maki keeps the file, the plugin
/// keeps its shape. Nothing is returned when the provider has never stored
/// any, which is how a first `resolve_auth` tells a fresh install from a
/// logged-in one.
///
/// @param slug string A provider slug this plugin registered.
/// @return (table?, string?) The stored credentials, or nil plus an error.
/// @example
/// local creds = maki.provider.auth.get("acme")
/// if creds then print(creds.access_token) end
#[lua_fn]
fn get(lua: &Lua, #[ctx] slugs: OwnedSlugs, slug: String) -> LuaResult<Pair<Table>> {
    owned(&slugs, &slug)?;
    let dir = try_pair!(StateDir::resolve());
    let Some(data) = load_plugin_auth(&dir, &slug) else {
        return Ok((None, None));
    };
    match json_to_lua(lua, &Value::Object(data))? {
        LuaValue::Table(table) => Ok((Some(table), None)),
        _ => Ok((None, None)),
    }
}

/// Read the live credentials and effective origin of one of this plugin's
/// providers.
///
/// For a hook that has to reach an endpoint the codec knows nothing about, a
/// balance or a quota url, and so needs exactly what every request to the slug
/// already carries. `headers` holds whatever maki resolved for it: the bearer
/// token from the declared `api_key_env`, whatever `resolve_auth` returned, and
/// any `[<slug>.headers]` from `providers.toml`. `base_url` is the origin a
/// request would reach right now, resolved the way the codec resolves it: an
/// auth-supplied origin, then `<SLUG>_BASE_URL` or `providers.toml`, then the
/// declared `base_url`. Hard-coding an origin instead would send the call
/// somewhere else than the rest of the provider whenever a user points the slug
/// at a gateway.
///
/// This hands over live credentials, which is why it only answers for the
/// providers the calling plugin declared. A snapshot, like the one every
/// request takes, so a refresh landing mid-call cannot swap the headers a hook
/// is already building a request from.
///
/// @param slug string A provider slug this plugin registered.
/// @return (table?, string?) `{ base_url = ..., headers = { ... } }`, or
///   `(nil, err)` on failure.
/// @example
/// local auth, err = maki.provider.auth.resolved("acme")
/// if not auth then return end
/// local res = maki.net.request(auth.base_url .. "/usage", { headers = auth.headers })
#[lua_fn]
fn resolved(lua: &Lua, #[ctx] slugs: OwnedSlugs, slug: String) -> LuaResult<Pair<Table>> {
    owned(&slugs, &slug)?;
    let Some(auth) = plugin::resolved_auth(&slug) else {
        return Ok(err_pair(format!(
            "maki.provider.auth.resolved: no provider serves '{slug}'"
        )));
    };
    let headers = lua.create_table()?;
    for (name, value) in auth.headers {
        headers.set(name, value)?;
    }
    let resolved = lua.create_table()?;
    resolved.set(BASE_URL, plugin::effective_base_url(&slug))?;
    resolved.set(HEADERS, headers)?;
    Ok((Some(resolved), None))
}

/// Every change to the credential store, off the plugin host's thread.
///
/// [`lock_credentials`] waits on a file lock another maki process may hold, and
/// the host is single threaded: waiting for it here would stall every tool
/// call, timer and provider hook in this process behind one plugin's token
/// rotation. The common case takes no file lock at all, because the host
/// already holds this slug while its hook runs and the lock lets it back in,
/// so the write costs one hop off the thread and back.
async fn write_credentials(
    slug: String,
    write: impl FnOnce(&StateDir, &str) -> Result<(), String> + Send + 'static,
) -> Result<(), String> {
    smol::unblock(move || {
        let dir = StateDir::resolve().map_err(|e| e.to_string())?;
        let _lock = lock_credentials(&dir, &slug);
        write(&dir, &slug)
    })
    .await
}

/// Store credentials for one of this plugin's providers.
///
/// Written to `~/.local/state/maki/auth/plugins/<slug>.json` with mode 0600,
/// replaced in one atomic step, and serialised against other maki processes
/// touching the same provider's credentials. Any JSON-shaped table works, so a
/// plugin can keep a refresh token, an expiry and whatever else its flow needs.
///
/// @param slug string A provider slug this plugin registered.
/// @param credentials table Any table with string keys.
/// @return (boolean?, string?) True, or nil plus an error string.
/// @example
/// local ok, err = maki.provider.auth.set("acme", { access_token = token })
/// if not ok then maki.log.error(err) end
#[lua_fn]
async fn set(
    lua: Lua,
    #[ctx] slugs: OwnedSlugs,
    slug: String,
    credentials: Table,
) -> LuaResult<Pair<bool>> {
    owned(&slugs, &slug)?;
    let Value::Object(data) = lua_to_json(&lua, &LuaValue::Table(credentials))? else {
        return Err(mlua::Error::runtime(
            "maki.provider.auth.set: credentials must be a table with string keys",
        ));
    };
    try_pair!(
        write_credentials(slug, move |dir, slug| {
            save_plugin_auth(dir, slug, &data).map_err(|e| e.to_string())
        })
        .await
    );
    Ok((Some(true), None))
}

/// Forget the credentials stored for one of this plugin's providers.
///
/// @param slug string A provider slug this plugin registered.
/// @return (boolean?, string?) True, or nil plus an error string.
/// @example
/// maki.provider.auth.clear("acme")
#[lua_fn]
async fn clear(_lua: Lua, #[ctx] slugs: OwnedSlugs, slug: String) -> LuaResult<Pair<bool>> {
    owned(&slugs, &slug)?;
    try_pair!(
        write_credentials(slug, |dir, slug| {
            delete_plugin_auth(dir, slug)
                .map(drop)
                .map_err(|e| e.to_string())
        })
        .await
    );
    Ok((Some(true), None))
}

lua_table! {
    /// Credentials for the providers this plugin registered, kept by maki
    /// apart from its own: one file per slug at
    /// `~/.local/state/maki/auth/plugins/<slug>.json`, created with mode 0600,
    /// replaced in one atomic step, and locked against other maki processes
    /// while a refresh writes.
    ///
    /// The stored value is a free-form JSON object. maki owns where it lives
    /// and who may read it, the plugin owns what is in it.
    ///
    /// `resolved` is the other direction: not what the plugin wrote, but the
    /// credentials and origin maki resolved for the slug and sends on every
    /// request to it.
    ///
    /// A plugin can only reach slugs it registered itself.
    ///
    /// ```lua
    /// maki.provider.auth.set("acme", { access_token = tok, expires = when })
    /// local creds = maki.provider.auth.get("acme")
    /// local auth = maki.provider.auth.resolved("acme")
    /// maki.provider.auth.clear("acme")
    /// ```
    "maki.provider.auth" => pub(crate) fn create_auth_table(slugs: OwnedSlugs), AUTH_DOCS [
        get(slugs),
        set(slugs),
        clear(slugs),
        resolved(slugs),
    ]
}

lua_table! {
    /// Providers implemented in Lua.
    ///
    /// A registered provider is a first-class one: its models show up in the
    /// picker and in config, its requests go through maki's usual retry,
    /// pricing and usage accounting, and its credentials live where maki keeps
    /// every other provider's.
    ///
    /// Registration happens while the plugin loads, so it belongs at the top
    /// level of the plugin file rather than inside a callback.
    ///
    /// A plugin that registers a provider needs the `net` permission and a
    /// non-empty `net_hosts` list in its `plugin.toml`. Those hosts are the
    /// only origins maki sends the provider's credentials to.
    ///
    /// ```lua
    /// maki.provider.register({
    ///   slug = "acme",
    ///   display_name = "Acme",
    ///   codec = "openai",
    ///   base_url = "https://api.acme.com/v1",
    ///   api_key_env = "ACME_API_KEY",
    ///   models = { { prefixes = { "acme-large" }, tier = "strong" } },
    /// })
    /// ```
    "maki.provider" => pub(crate) fn create_provider_table(perms: &PluginPermissions, plugin: Arc<str>, egress: NetEgress, authority: DeclAuthority, reads_env: bool), DOCS [
        register(perms, plugin, egress, authority, reads_env),
    ]
}

/// `maki.provider`, with the credential store bound to the calling plugin.
///
/// `egress` is the same value `maki.net` holds, so a slug registered here is
/// a host reachable there without the two being kept in step by hand.
pub(crate) fn create_provider_namespace(
    lua: &Lua,
    permissions: &PluginPermissions,
    plugin: Arc<str>,
    egress: NetEgress,
    authority: DeclAuthority,
) -> LuaResult<Table> {
    let owned = egress.owned();
    let reads_env = permissions.is_allowed(Permission::Env);
    let provider = create_provider_table(lua, permissions, plugin, egress, authority, reads_env)?;
    provider.set("auth", create_auth_table(lua, owned)?)?;
    Ok(provider)
}

#[cfg(test)]
mod tests {
    use maki_providers::ProviderUsage;
    use maki_providers::plugin::AuthPurpose;
    use serde_json::json;
    use test_case::test_case;

    use super::*;
    use crate::plugin_permissions::NET_HOSTS_KEY;

    const PLUGIN: &str = "test";
    const SLUG_NAME: &str = "acme";
    const OTHER_SLUG: &str = "rival";
    const NOT_OWNED: &str = "is not a provider this plugin registered";
    const REGISTER_FN: &str = "register";
    const UNKNOWN_CODEC: &str = "grpc";
    const UNKNOWN_DIALECT: &str = "esperanto";

    fn keys_from(
        lua: &Lua,
        entries: impl IntoIterator<Item = (&'static str, Function)>,
    ) -> LuaHookKeys {
        LuaHookKeys {
            plugin: Arc::from(PLUGIN),
            slug: SLUG_NAME.to_owned(),
            keys: entries
                .into_iter()
                .map(|(name, func)| (name, lua.create_registry_value(func).unwrap()))
                .collect(),
            requests: flume::unbounded().0,
            release: flume::unbounded().0,
        }
    }

    fn keys_with(entries: &[&'static str]) -> LuaHookKeys {
        let lua = Lua::new();
        let noop = lua.create_function(|_, ()| Ok(())).unwrap();
        keys_from(&lua, entries.iter().map(|name| (*name, noop.clone())))
    }

    /// The selector on the wire is whatever serde renders [`AuthPurpose`] as,
    /// and the entry table here spells those three names again. A rename on
    /// either side would fall back to the first entry the plugin wrote, which
    /// is the wrong hook rather than an error, so the two tables are pinned to
    /// each other.
    #[test_case(AuthPurpose::Resolve, RESOLVE_AUTH ; "resolve")]
    #[test_case(AuthPurpose::Refresh, REFRESH_AUTH ; "refresh")]
    #[test_case(AuthPurpose::Reload, RELOAD_AUTH ; "reload")]
    fn a_purpose_picks_the_entry_named_after_it(purpose: AuthPurpose, expected: &'static str) {
        let keys = keys_with(&[RESOLVE_AUTH, REFRESH_AUTH, RELOAD_AUTH]);
        let payload = serde_json::to_value(purpose).unwrap();

        let chosen = keys.entry(HookSlot::Auth, &payload).unwrap();

        assert!(std::ptr::eq(chosen, keys.keys.get(expected).unwrap()));
    }

    /// One host-side hook, three Lua entries, and a plugin that reads its
    /// credentials fresh every time writes only `resolve_auth`.
    #[test_case(&[RESOLVE_AUTH], "refresh", Some(RESOLVE_AUTH) ; "refresh_falls_back_to_resolve")]
    #[test_case(&[RESOLVE_AUTH], "reload", Some(RESOLVE_AUTH) ; "reload_falls_back_to_resolve")]
    #[test_case(&[], "resolve", None ; "no_auth_entry_serves_nothing")]
    fn auth_purpose_selects_an_entry(
        entries: &[&'static str],
        purpose: &str,
        expected: Option<&'static str>,
    ) {
        let keys = keys_with(entries);
        let payload = json!(purpose);
        let chosen = keys.entry(HookSlot::Auth, &payload);
        assert_eq!(
            chosen.is_some(),
            expected.is_some(),
            "{entries:?} for {purpose}"
        );
        if let Some(expected) = expected {
            assert!(std::ptr::eq(
                chosen.unwrap(),
                keys.keys.get(expected).unwrap()
            ));
        }
    }

    #[test_case(HookSlot::Login, &[LOGIN], true ; "a_login_entry_serves_the_login_slot")]
    #[test_case(HookSlot::Login, &[LOGOUT], false ; "a_logout_entry_does_not")]
    #[test_case(HookSlot::Auth, &[RELOAD_AUTH], true ; "any_auth_entry_serves_the_auth_slot")]
    fn slots_are_served_by_their_own_entries(
        slot: HookSlot,
        entries: &[&'static str],
        expected: bool,
    ) {
        assert_eq!(keys_with(entries).serves(slot), expected);
    }

    /// `build_body(body, model, opts)`: the fields the signature names arrive
    /// positionally, the rest ride along in the trailing table.
    #[test]
    fn body_input_is_split_into_the_documented_arguments() {
        let lua = Lua::new();
        let payload = json!({ "body": { "stream": true }, "model": "m-1", "thinking": "high" });
        let args = call_args(&lua, HookSlot::BuildBody.spec(), payload).unwrap();
        let args: Vec<LuaValue> = args.into_iter().collect();

        assert_eq!(args.len(), 3);
        let body = args[0].as_table().unwrap();
        assert!(body.get::<bool>("stream").unwrap());
        assert_eq!(args[1].as_string().unwrap().to_string_lossy(), "m-1");
        let opts = args[2].as_table().unwrap();
        assert_eq!(opts.get::<String>("thinking").unwrap(), "high");
    }

    #[test]
    fn a_login_call_is_handed_a_terminal_ctx() {
        let lua = Lua::new();
        let args = call_args(&lua, HookSlot::Login.spec(), Value::Null).unwrap();
        let args: Vec<LuaValue> = args.into_iter().collect();

        assert_eq!(args.len(), 1);
        let ctx = args[0].as_table().unwrap();
        for name in ["print", "prompt", "open_url"] {
            assert!(ctx.get::<Function>(name).is_ok(), "ctx.{name} is missing");
        }
    }

    /// The hook every plugin writes without knowing it: `build_body` edits one
    /// key and hands the body back. A JSON null crosses into Lua as a nil and a
    /// nil key is an absent key, so without the template the fields the hook
    /// never looked at would come back deleted, silently, and only for the
    /// bodies that happen to carry a null.
    #[test]
    fn a_body_a_hook_handed_back_keeps_the_fields_it_never_touched() {
        const EDITED_FIELD: &str = "thinking";
        const KEPT_NULL_FIELD: &str = "tool_choice";

        let lua = Lua::new();
        let func = lua
            .create_function(|_, (body, _model, _opts): (Table, LuaValue, LuaValue)| {
                body.set(EDITED_FIELD, true)?;
                Ok(body)
            })
            .unwrap();
        let hooks = keys_from(&lua, [(BUILD_BODY, func)]);
        let payload = json!({
            BODY_FIELD: { KEPT_NULL_FIELD: Value::Null, "stream": true },
            MODEL_FIELD: "m-1",
        });

        let returned = smol::block_on(run_hook(&lua, &hooks, HookSlot::BuildBody, payload));
        let body: Value = decode(&lua, returned.unwrap()).unwrap();

        assert_eq!(
            body,
            json!({ KEPT_NULL_FIELD: Value::Null, "stream": true, EDITED_FIELD: true })
        );
    }

    /// An empty Lua table is an empty list and an empty object at once, and
    /// only the type the caller decodes into can say which. A `fetch_usage`
    /// that found no balances builds `limits` as exactly that.
    #[test]
    fn an_empty_table_decodes_as_the_list_the_caller_expects() {
        let lua = Lua::new();
        let value = lua.load("return { limits = {} }").eval().unwrap();

        let usage: ProviderUsage = decode(
            &lua,
            HookReturn {
                value,
                template: None,
            },
        )
        .unwrap();

        assert!(usage.limits.is_empty());
    }

    fn auth_table(lua: &Lua, owns: &[&str]) -> Table {
        let egress = NetEgress::default();
        for slug in owns {
            egress.owns((*slug).to_owned());
        }
        create_auth_table(lua, egress.owned()).unwrap()
    }

    /// The scoping this namespace exists to enforce: the plugin's own slugs are
    /// captured when its `maki` global is built, so naming someone else's is
    /// refused before anything reaches the credential store. The owned case is
    /// there so the refusal is about ownership and not about the call shape.
    #[test_case("get", OTHER_SLUG, true ; "get_refuses_a_foreign_slug")]
    #[test_case("set", OTHER_SLUG, true ; "set_refuses_a_foreign_slug")]
    #[test_case("clear", OTHER_SLUG, true ; "clear_refuses_a_foreign_slug")]
    #[test_case("get", SLUG_NAME, false ; "an_owned_slug_gets_through")]
    fn auth_calls_are_scoped_to_the_slugs_the_plugin_registered(
        call: &str,
        slug: &str,
        refused: bool,
    ) {
        let lua = Lua::new();
        lua.globals()
            .set("auth", auth_table(&lua, &[SLUG_NAME]))
            .unwrap();

        let error = lua
            .load(format!(r#"auth.{call}("{slug}", {{}})"#))
            .exec()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert_eq!(error.contains(NOT_OWNED), refused, "{error}");
    }

    #[test_case("openai" ; "openai")]
    #[test_case("openai-responses" ; "openai_responses")]
    #[test_case("anthropic" ; "anthropic")]
    #[test_case("google" ; "google")]
    fn every_documented_codec_parses(name: &str) {
        let lua = Lua::new();
        let spec = lua.create_table().unwrap();
        spec.set(CODEC, name).unwrap();
        assert!(codec(&spec).unwrap().is_some(), "{name}");
    }

    /// The registry owns the dialect names, and this doc comment is where a
    /// plugin author reads them. A name added there and not here is a dialect
    /// nobody can ask for, so the two lists are held together.
    #[test]
    fn every_dialect_name_is_documented_and_resolves() {
        let lua = Lua::new();
        let spec = lua.create_table().unwrap();
        let documented = DOCS
            .fns
            .iter()
            .find(|f| f.name == REGISTER_FN)
            .unwrap()
            .desc;

        for name in maki_providers::dialect::NAMES {
            assert!(documented.contains(&format!("`\"{name}\"`")), "{name}");
            spec.set(THINKING_DIALECT, *name).unwrap();
            assert!(dialect(&spec, SLUG_NAME).unwrap().is_some(), "{name}");
        }
    }

    /// Codec and dialect names both live in the registry, so a name nobody
    /// implements has to be caught here, where the plugin that wrote it can be
    /// named. The message needs both halves: which call refused, and the name
    /// it did not know. Asserting on those rather than on the whole sentence
    /// keeps this from breaking when the sentence is reworded.
    #[test_case(CODEC, UNKNOWN_CODEC ; "codec")]
    #[test_case(THINKING_DIALECT, UNKNOWN_DIALECT ; "thinking_dialect")]
    fn an_unknown_name_is_refused_where_the_plugin_can_be_named(key: &str, name: &str) {
        let lua = Lua::new();
        let spec = lua.create_table().unwrap();
        spec.set(key, name).unwrap();

        let refusal = codec(&spec)
            .err()
            .or_else(|| dialect(&spec, SLUG_NAME).err())
            .expect("an unknown name must be refused");

        let error = refusal.to_string();
        assert!(error.contains(REGISTER), "{error}");
        assert!(error.contains(name), "{error}");
    }

    /// Registers `extra` on top of a minimal spec, for a registration that has
    /// to be refused, and hands back why it was.
    fn register_refusal(reads_env: bool, extra: &str) -> String {
        let lua = Lua::new();
        let table = create_provider_table(
            &lua,
            &PluginPermissions::trusted(),
            Arc::from(PLUGIN),
            NetEgress::default(),
            DeclAuthority::ThirdParty,
            reads_env,
        )
        .unwrap();
        lua.globals().set("provider", table).unwrap();

        lua.load(format!(
            r#"provider.register({{ slug = "{SLUG_NAME}", display_name = "Acme", codec = "openai"{extra} }})"#
        ))
        .exec()
        .unwrap_err()
        .to_string()
    }

    /// A provider that names no host would have maki send its credentials
    /// wherever a hook later asks, so the manifest key is a hard requirement
    /// and the refusal points straight at it.
    #[test]
    fn registering_without_declared_hosts_is_refused() {
        let error = register_refusal(true, "");
        assert!(error.contains(NET_HOSTS_KEY), "{error}");
    }

    /// `api_key_env` names a variable for maki to read, so a plugin without
    /// `env` could otherwise read any secret in the environment by naming it
    /// and asking `maki.provider.auth.resolved` for the bearer it became.
    #[test]
    fn an_api_key_env_needs_the_env_permission() {
        let error = register_refusal(false, r#", api_key_env = "AWS_SECRET_ACCESS_KEY""#);
        assert!(error.contains(API_KEY_ENV_NEEDS_ENV), "{error}");
    }
}
