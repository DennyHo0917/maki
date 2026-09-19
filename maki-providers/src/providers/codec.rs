use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use flume::Sender;
use serde_json::Value;

use maki_config::providers::Protocol;
use maki_storage::id::SessionRef;

use super::ResolvedAuth;
use super::Timeouts;
use super::openai::responses;
use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use crate::model::{Model, ModelInfo};
use crate::provider::{BoxFuture, Provider};
use crate::spec::{ProviderRegistry, ProviderSpec};
use crate::types::{EffortDialect, ThinkingFallback};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse};

/// What a plain openai endpoint takes, and what a declaration that says
/// nothing gets.
const DEFAULT_MAX_TOKENS_FIELD: &str = "max_tokens";

/// Everything a provider declaration tells the codec about its wire, in one
/// struct: the six fields an [`OpenAiCompatConfig`] is made of, plus the three
/// a declaration may add on top of the protocol. Passed whole so a new option
/// is a field here rather than another positional argument at two call sites.
///
/// `Cow` throughout because a declaration authored in Rust names `&'static
/// str` and one authored in Lua owns its strings.
pub struct CodecOptions {
    pub protocol: Protocol,
    /// The real slug, never empty: it is what
    /// [`maki_config::providers::configured_base_url`] keys the user's
    /// `<SLUG>_BASE_URL` and `providers.toml` origin off.
    pub slug: Cow<'static, str>,
    pub api_key_env: Cow<'static, str>,
    /// The declaration's *static* origin, and the last resort of the three
    /// (see [`OpenAiCompatProvider::base_url`]): the user's env var and
    /// `providers.toml` outrank it, and credentials a hook resolved outrank
    /// those.
    pub base_url: Cow<'static, str>,
    /// `None` is [`DEFAULT_MAX_TOKENS_FIELD`].
    pub max_tokens_field: Option<Cow<'static, str>>,
    /// `None` asks for `stream_options.include_usage`, which every openai
    /// endpoint worth billing for supports.
    pub include_stream_usage: Option<bool>,
    /// A log label; defaults to the slug.
    pub provider_name: Cow<'static, str>,
    pub system_prefix: Option<String>,
    /// Set to spell effort the way this provider's API does, which replaces
    /// the generic thinking pass (see [`CompatProvider::stream_message`]).
    pub thinking_dialect: Option<&'static EffortDialect<'static>>,
    pub build_body: Option<Arc<dyn BodyHook>>,
}

impl CodecOptions {
    /// Everything but the protocol and the slug left as a plain openai
    /// endpoint behaves, so a caller states only what it differs on.
    pub fn new(protocol: Protocol, slug: impl Into<Cow<'static, str>>) -> Self {
        let slug = slug.into();
        Self {
            protocol,
            api_key_env: Cow::Borrowed(""),
            base_url: Cow::Borrowed(""),
            max_tokens_field: None,
            include_stream_usage: None,
            provider_name: slug.clone(),
            slug,
            system_prefix: None,
            thinking_dialect: None,
            build_body: None,
        }
    }
}

/// The native provider a custom or plugin slug borrows its codec and fallbacks
/// from. Resolved through [`ProviderRegistry::get`], never `for_slug`, so the
/// lookup cannot recurse back into here.
pub(crate) fn protocol_spec(protocol: Protocol) -> Option<&'static ProviderSpec> {
    ProviderRegistry::get(match protocol {
        Protocol::Openai | Protocol::OpenaiResponses => super::openai::SLUG,
        Protocol::Anthropic => super::anthropic::SLUG,
        Protocol::Google => super::google::SLUG,
    })
}

/// Applied to the final request body, after the codec built it and after the
/// thinking pass, so a hook sees exactly what goes on the wire.
pub trait BodyHook: Send + Sync {
    fn call<'a>(
        &'a self,
        body: Value,
        model: &'a Model,
        opts: RequestOptions,
    ) -> BoxFuture<'a, Result<Value, AgentError>>;
}

/// The one place a declaration's wire options become the compat layer's, so
/// the two defaults are written once.
fn compat_config(options: &CodecOptions) -> OpenAiCompatConfig {
    OpenAiCompatConfig {
        slug: options.slug.clone(),
        api_key_env: options.api_key_env.clone(),
        base_url: options.base_url.clone(),
        max_tokens_field: options
            .max_tokens_field
            .clone()
            .unwrap_or(Cow::Borrowed(DEFAULT_MAX_TOKENS_FIELD)),
        include_stream_usage: options.include_stream_usage.unwrap_or(true),
        provider_name: options.provider_name.clone(),
    }
}

/// The one place the protocol -> codec dispatch is spelled out. Every codec
/// here honours `system_prefix` except google, which drops it (see
/// [`super::google`]) and refuses it at registration instead.
pub fn build(
    options: CodecOptions,
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
) -> Box<dyn Provider> {
    match options.protocol {
        Protocol::Anthropic => Box::new(
            super::anthropic::Anthropic::with_auth(auth, timeouts)
                .with_system_prefix(options.system_prefix),
        ),
        Protocol::Openai | Protocol::OpenaiResponses => Box::new(CompatProvider {
            compat: OpenAiCompatProvider::new(compat_config(&options), timeouts),
            auth,
            protocol: options.protocol,
            system_prefix: options.system_prefix,
            thinking_dialect: options.thinking_dialect,
            build_body: options.build_body,
        }),
        Protocol::Google => Box::new(super::google::Google::with_auth(auth, timeouts)),
    }
}

pub(crate) struct CompatProvider {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    protocol: Protocol,
    system_prefix: Option<String>,
    thinking_dialect: Option<&'static EffortDialect<'static>>,
    build_body: Option<Arc<dyn BodyHook>>,
}

impl Provider for CompatProvider {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        opts: RequestOptions,
        _session_id: Option<&'a SessionRef>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            let mut auth = self.auth.lock().unwrap().clone();
            let mut buf = String::new();
            let system = super::with_prefix(&self.system_prefix, system, &mut buf);

            if self.protocol == Protocol::OpenaiResponses {
                // `responses::do_stream` reads the origin off the auth alone,
                // so the three-way precedence is resolved here instead. Left
                // untouched when nothing resolved, so a declaration with no
                // origin anywhere still fails there rather than posting to "".
                let resolved = self.compat.base_url(&auth);
                if !resolved.is_empty() {
                    auth.base_url = Some(resolved);
                }
                let mut body = responses::build_body(model, messages, system, tools);
                // TODO: wire thinking budget into responses API when llama.cpp supports it
                if let Some(hook) = &self.build_body {
                    body = hook.call(body, model, opts).await?;
                }
                return responses::do_stream(
                    self.compat.client(),
                    model,
                    &body,
                    event_tx,
                    &auth,
                    self.compat.stream_timeout(),
                )
                .await;
            }

            let mut body = self.compat.build_body(model, messages, system, tools);
            // A substitution, never both: a declared dialect is the provider
            // saying how its own API spells effort, which is a different
            // question from what the model declared about itself, and
            // `apply_reasoning_effort` deliberately skips both the
            // `thinking_fields` merge and the `supports_thinking` gate.
            match self.thinking_dialect {
                Some(dialect) => opts.thinking.apply_reasoning_effort(&mut body, dialect, model),
                None => opts
                    .thinking
                    .apply_thinking(&mut body, model, ThinkingFallback::None),
            }
            if let Some(hook) = &self.build_body {
                body = hook.call(body, model, opts).await?;
            }
            self.compat
                .do_stream(model, &[], &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        let auth = self.auth.lock().unwrap().clone();
        Box::pin(async move { self.compat.do_list_models(&auth).await })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::super::plugin::PluginAuth;
    use super::*;

    /// One slug per case: `<SLUG>_BASE_URL` is process-wide, and these run in
    /// one process under `cargo test`.
    const DECLARED_SLUG: &str = "codec-declared-origin";
    const OVERRIDDEN_SLUG: &str = "codec-overridden-origin";
    const OVERRIDDEN_ENV: &str = "CODEC_OVERRIDDEN_ORIGIN_BASE_URL";
    const HOOKED_SLUG: &str = "codec-hooked-origin";
    const HOOKED_ENV: &str = "CODEC_HOOKED_ORIGIN_BASE_URL";
    const DECLARED_URL: &str = "https://declared.example/v1";
    const USER_URL: &str = "https://user.example/v1";
    const HOOK_URL: &str = "https://hooked.example/v1";
    const HOOK_HOST: &str = "hooked.example";

    /// A declaration that states an origin and nothing else.
    fn compat(slug: &'static str) -> OpenAiCompatProvider {
        let options = CodecOptions {
            base_url: Cow::Borrowed(DECLARED_URL),
            ..CodecOptions::new(Protocol::Openai, slug)
        };
        OpenAiCompatProvider::new(compat_config(&options), Timeouts::default())
    }

    fn no_credentials(slug: &str) -> ResolvedAuth {
        ResolvedAuth::new(slug, Vec::new()).unwrap()
    }

    #[test]
    fn a_declared_origin_is_the_last_resort() {
        let compat = compat(DECLARED_SLUG);
        assert_eq!(compat.base_url(&no_credentials(DECLARED_SLUG)), DECLARED_URL);
    }

    /// What the empty `slug` on the old shared config hid: a plugin or
    /// `providers.toml` provider never consulted `<SLUG>_BASE_URL` at all, so
    /// a declaration outranked the user.
    #[test]
    fn a_user_origin_beats_a_declared_one() {
        unsafe { std::env::set_var(OVERRIDDEN_ENV, USER_URL) };
        let compat = compat(OVERRIDDEN_SLUG);
        assert_eq!(compat.base_url(&no_credentials(OVERRIDDEN_SLUG)), USER_URL);
    }

    /// Provenance is carried by the type. A hook's origin comes through
    /// [`PluginAuth::into_resolved`], the one door that vets it against the
    /// declared hosts, and lands in the auth cell; `resolved_base_url` is read
    /// off the user's env and `providers.toml` at construction and no hook can
    /// reach it. So a hook wins the request it answered for without becoming
    /// what the user configured.
    #[test]
    fn a_hook_origin_wins_without_replacing_the_user_s() {
        unsafe { std::env::set_var(HOOKED_ENV, USER_URL) };
        let compat = compat(HOOKED_SLUG);
        let hooked = PluginAuth {
            base_url: Some(HOOK_URL.to_string()),
            headers: HashMap::new(),
        }
        .into_resolved(HOOKED_SLUG, &[HOOK_HOST.to_string()])
        .unwrap();

        assert_eq!(compat.base_url(&hooked), HOOK_URL);
        assert_eq!(compat.base_url(&no_credentials(HOOKED_SLUG)), USER_URL);
    }
}
