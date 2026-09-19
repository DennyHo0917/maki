use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use flume::Sender;
use maki_storage::id::SessionRef;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::warn;

use maki_config::providers::Protocol;

use crate::model::{Model, ModelFamily};
use crate::pricing::{PricingSchedule, PricingWindow};
use crate::provider::{BoxFuture, Provider};
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{
    ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, Native, ProviderSpec,
};
use crate::types::{ProviderUsage, THINKING_OFF, UsageLimit};
use crate::{
    AgentError, Message, ProviderEvent, RequestOptions, StreamResponse, ThinkingConfig, dialect,
};

use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use super::plugin::{self, BodyInput, Hook, ProviderDecl, ProviderHooks};
use super::{KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts};

const PAD: &str = "";
const REASONER_ID: &str = "deepseek-reasoner";
const BALANCE_PATH: &str = "/user/balance";
const THINKING_FIELD: &str = "thinking";
const THINKING_ENABLED: &str = "enabled";
const THINKING_DISABLED: &str = "disabled";
const NET_HOST: &str = "api.deepseek.com";
const NOT_REGISTERED: &str = "deepseek is not registered";

const SLUG: &str = "deepseek";
const DISPLAY_NAME: &str = "DeepSeek";
const ENV_VAR: &str = "DEEPSEEK_API_KEY";
const BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_MODEL: &str = "deepseek/deepseek-flash";
const LOGIN_URL: &str = "https://platform.deepseek.com/api_keys";
const MAX_TOKENS_FIELD: &str = "max_tokens";
const FEATURES: &str = "Thinking mode toggle (on/off), open-weight models";

static CONFIG: OpenAiCompatConfig = OpenAiCompatConfig {
    slug: Cow::Borrowed(SLUG),
    api_key_env: Cow::Borrowed(ENV_VAR),
    base_url: Cow::Borrowed(BASE_URL),
    max_tokens_field: Cow::Borrowed(MAX_TOKENS_FIELD),
    include_stream_usage: true,
    provider_name: Cow::Borrowed(DISPLAY_NAME),
};

pub(crate) const SPEC: ProviderSpec = ProviderSpec {
    slug: SLUG,
    display_name: DISPLAY_NAME,
    api_key_env: ENV_VAR,
    family: ModelFamily::Generic,
    supports_thinking: true,
    accepts_arbitrary_models: false,
    fallback_max_output: Some(384_000),
    fallback_context_window: 1_000_000,
    models_toml: include_str!("../../models/deepseek.toml"),
    pricing_schedule: Some(&PEAK_HOURS),
    native: Some(Native {
        new: create,
        with_auth: create_with_auth,
    }),
    aperture: Some(ApertureRoute {
        path_prefix: DEFAULT_PATH_PREFIX,
    }),
    login: Some(LoginConfig {
        protocol: Protocol::Openai,
        default_base_url: BASE_URL,
        default_model: DEFAULT_MODEL,
        plans: None,
        login_url: Some(LOGIN_URL),
        needs_url: false,
    }),
    docs: GeneratedDocs {
        api_urls: &[BASE_URL],
        features: Some(FEATURES),
        auth: AuthDoc::EnvVar,
        catalog: CatalogDoc::Table,
        trailing_notes: &[],
    },
};

fn create(timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    Ok(Box::new(DeepSeek::new(timeouts)?))
}

fn create_with_auth(
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
    system_prefix: Option<String>,
) -> Box<dyn Provider> {
    Box::new(DeepSeek::with_auth(auth, timeouts).with_system_prefix(system_prefix))
}

inventory::submit!(SPEC.config_row());

/// This provider as a declaration plus the two things the openai codec cannot
/// spell: DeepSeek's thinking toggle and its balance endpoint.
///
/// Nothing the [`SPEC`] row already holds is restated -- claiming the built-in
/// slug inherits the display name, the key env var, the family, the fallback
/// limits, the curated model table and the peak-hour schedule. Neither are the
/// codec's own defaults: `max_tokens` and `include_stream_usage` are what
/// [`CONFIG`] already asks for, and a decl that states a default is a second
/// copy of it for every later port to keep in step.
///
/// Not in any startup list yet: the port that deletes the bespoke [`DeepSeek`]
/// below registers it, and until then the replay suite is what builds it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decl() -> ProviderDecl {
    ProviderDecl {
        slug: SLUG.to_owned(),
        display_name: None,
        codec: Some(Protocol::Openai),
        base: None,
        base_url: Some(BASE_URL.to_owned()),
        api_key_env: None,
        system_prefix: None,
        max_tokens_field: None,
        include_stream_usage: None,
        thinking_dialect: Some(&dialect::DEEPSEEK),
        models: Vec::new(),
        net_hosts: vec![NET_HOST.to_owned()],
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn hooks() -> ProviderHooks {
    ProviderHooks {
        build_body: Some(Arc::new(ThinkingToggle)),
        fetch_usage: Some(Arc::new(Balance)),
        ..ProviderHooks::default()
    }
}

/// The toggle DeepSeek wants on every request, and the padding a toggled-on
/// request needs.
///
/// `thinking` arrives rendered, and `off` is the one rendering that means
/// disabled: every effort level spells itself, `adaptive` spells itself and a
/// budget arrives as its bare token count.
///
/// The effort string has already been applied when this runs, which is the
/// reverse of the order the bespoke impl worked in. It makes no difference:
/// [`dialect::DEEPSEEK`] declares no `off` and no `adaptive` string, so the two
/// modes the bespoke impl skipped the effort call for are exactly the two that
/// emit nothing anyway. The replay goldens are what settle that, not this note.
struct ThinkingToggle;

impl Hook<BodyInput, Value> for ThinkingToggle {
    fn call(&self, input: BodyInput) -> BoxFuture<'_, Result<Value, AgentError>> {
        Box::pin(async move {
            let BodyInput {
                mut body,
                model,
                thinking,
            } = input;
            let enabled = thinking != THINKING_OFF;
            set_thinking(&mut body, enabled);
            if enabled {
                pad_reasoning_content(&model, &mut body);
            }
            Ok(body)
        })
    }
}

/// DeepSeek's balance endpoint, which is not on any codec's request path.
///
/// The hook is handed no credentials, so it reads the ones the registration
/// resolved for this slug, and the origin the same way every request to this
/// provider resolves it: an auth-supplied base url, then the user's
/// `DEEPSEEK_BASE_URL` / `providers.toml`, then the declared default. The
/// bespoke impl below posts this one request to a hard-coded
/// `https://api.deepseek.com`, so a user who points the slug at a gateway has
/// their balance read straight from DeepSeek instead.
struct Balance;

impl Hook<(), Option<ProviderUsage>> for Balance {
    fn call(&self, (): ()) -> BoxFuture<'_, Result<Option<ProviderUsage>, AgentError>> {
        Box::pin(async move {
            let auth = plugin::resolved_auth(SLUG).ok_or_else(|| AgentError::Config {
                message: NOT_REGISTERED.to_owned(),
            })?;
            let compat = OpenAiCompatProvider::new(&CONFIG, Timeouts::default());
            let url = format!("{}{BALANCE_PATH}", compat.base_url(&auth));
            let parsed: BalanceResponse =
                serde_json::from_str(&compat.get_text(&auth, &url).await?)?;
            Ok(Some(parsed.into()))
        })
    }
}

fn set_thinking(body: &mut Value, enabled: bool) {
    let mode = if enabled {
        THINKING_ENABLED
    } else {
        THINKING_DISABLED
    };
    body[THINKING_FIELD] = json!({"type": mode});
}

/// Peak hours double every rate, and `models/deepseek.toml` quotes the off-peak
/// ones. The weekend stays off-peak around the clock.
/// <https://api-docs.deepseek.com/quick_start/pricing/>
pub(crate) const PEAK_HOURS: PricingSchedule =
    PricingSchedule::new(PEAK_WINDOWS, PEAK_MULTIPLIER).weekdays_only();

const PEAK_WINDOWS: &[PricingWindow] = &[PricingWindow::hours(1, 4), PricingWindow::hours(6, 10)];
const PEAK_MULTIPLIER: f64 = 2.0;

#[derive(Deserialize)]
struct BalanceResponse {
    balance_infos: Vec<BalanceInfo>,
}

#[derive(Deserialize)]
struct BalanceInfo {
    currency: String,
    total_balance: String,
    granted_balance: String,
    topped_up_balance: String,
}

impl From<BalanceResponse> for ProviderUsage {
    fn from(resp: BalanceResponse) -> Self {
        let limits = resp
            .balance_infos
            .into_iter()
            .map(|b| {
                let symbol = match b.currency.as_str() {
                    "USD" => "$",
                    "CNY" => "¥",
                    _ => "",
                };

                UsageLimit {
                    label: "Balance".into(),
                    percentage: None,
                    reset_at: None,
                    detail: Some(format!(
                        "total: {}{}, topped-up: {}{}, granted: {}{}",
                        symbol,
                        b.total_balance,
                        symbol,
                        b.topped_up_balance,
                        symbol,
                        b.granted_balance
                    )),
                }
            })
            .collect();
        ProviderUsage {
            plan: None,
            limits,
            by_model_today: vec![],
        }
    }
}

pub struct DeepSeek {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    key_pool: Option<KeyPool>,
    system_prefix: Option<String>,
}

impl DeepSeek {
    pub fn new(timeouts: super::Timeouts) -> Result<Self, AgentError> {
        let pool = KeyPool::resolve(&CONFIG.slug, &CONFIG.api_key_env)?;
        Ok(Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth: Arc::new(Mutex::new(ResolvedAuth::bearer(
                &CONFIG.slug,
                pool.current(),
            )?)),
            key_pool: Some(pool),
            system_prefix: None,
        })
    }

    pub(crate) fn with_auth(auth: Arc<Mutex<ResolvedAuth>>, timeouts: super::Timeouts) -> Self {
        Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth,
            key_pool: None,
            system_prefix: None,
        }
    }

    pub(crate) fn with_system_prefix(mut self, prefix: Option<String>) -> Self {
        self.system_prefix = prefix;
        self
    }
}

impl Provider for DeepSeek {
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
            let auth = self.auth.lock().unwrap().clone();
            let mut buf = String::new();
            let system = super::with_prefix(&self.system_prefix, system, &mut buf);
            let mut body = self.compat.build_body(model, messages, system, tools);

            let enabled = opts.thinking.is_enabled();
            set_thinking(&mut body, enabled);
            if enabled {
                opts.thinking
                    .apply_reasoning_effort(&mut body, &dialect::DEEPSEEK, model);
                if matches!(opts.thinking, ThinkingConfig::Budget(_)) {
                    warn!("DeepSeek reasoning does not support token budgets");
                }
                pad_reasoning_content(&model.id, &mut body);
            }

            self.compat
                .do_stream(model, &[], &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<crate::model::ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            self.compat.do_list_models(&auth).await
        })
    }

    fn fetch_usage(&self) -> BoxFuture<'_, Result<Option<ProviderUsage>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            let url = format!("{BASE_URL}{BALANCE_PATH}");
            let body = self.compat.get_text(&auth, &url).await?;
            let parsed: BalanceResponse = serde_json::from_str(&body)?;
            Ok(Some(parsed.into()))
        })
    }

    fn keys(&self) -> Option<KeyRotation<'_>> {
        Some(KeyRotation::new(
            self.key_pool.as_ref()?,
            &self.auth,
            KeyHeader::Bearer,
        ))
    }
}

/// Whether a model speaks the thinking protocol DeepSeek introduced with V4:
/// an explicit toggle, and `reasoning_content` echoed back on input. Only
/// `deepseek-reasoner` (R1) sits outside it, reasoning unconditionally and
/// refusing the field as input, so we name that one id rather than match a
/// version marker the next rename would break. Providers that resell DeepSeek
/// share the gate, after stripping their vendor prefix.
///
/// Ref: <https://api-docs.deepseek.com/guides/thinking_mode>
pub(crate) fn uses_v4_thinking_protocol(model_id: &str) -> bool {
    !model_id.starts_with(REASONER_ID)
}

/// V4 and later want `reasoning_content` on every assistant turn of a request
/// carrying `tools` (missing = 400), so we back-fill the turns that have none:
/// plain replies and tool-only turns. The API only checks the field exists, so
/// `""` is enough. Requests without tools are left alone, since nothing asks
/// for the field there and this runs for any id a DeepSeek-based custom
/// provider is pointed at.
fn pad_reasoning_content(model_id: &str, body: &mut Value) {
    if !uses_v4_thinking_protocol(model_id) || body.get("tools").is_none() {
        return;
    }
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for msg in messages {
        if msg.get("role").and_then(Value::as_str) != Some("assistant")
            || msg
                .get("reasoning_content")
                .and_then(Value::as_str)
                .is_some()
        {
            continue;
        }
        msg["reasoning_content"] = Value::String(PAD.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ProviderRegistry;
    use serde_json::json;
    use test_case::test_case;

    /// No version marker in the id, which is what the old substring gate missed.
    const FLASH: &str = "deepseek-flash";
    /// The hours, days and surcharge as the pricing page states them.
    const PUBLISHED_PEAK_HOURS: &str = "2x during 01:00-04:00, 06:00-10:00 UTC, Mon-Fri";

    /// `PEAK_HOURS` only reaches a bill through the spec, and a schedule
    /// that never got hooked up looks exactly like off-peak all day. The
    /// published hours are pinned here too, since either drifting bills every
    /// DeepSeek turn at the wrong rate.
    #[test]
    fn the_manifest_bills_the_published_peak_hours() {
        let schedule = ProviderRegistry::get(&CONFIG.slug)
            .expect("deepseek is a builtin")
            .pricing_schedule
            .expect("deepseek bills by the clock");
        assert_eq!(schedule.to_string(), PEAK_HOURS.to_string());
        assert_eq!(PEAK_HOURS.to_string(), PUBLISHED_PEAK_HOURS);
    }

    fn tool_call_body() -> Value {
        json!({
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "messages": [
                {"role": "system",    "content": "sys"},
                {"role": "user",      "content": "hi"},
                {"role": "assistant", "content": "ok", "reasoning_content": "kept"},
                {"role": "assistant", "content": "",   "tool_calls": [{"id": "c1"}]},
                {"role": "tool",      "tool_call_id": "c1", "content": "out"},
            ],
        })
    }

    #[test]
    fn pads_only_assistant_turns_without_reasoning() {
        let mut body = tool_call_body();
        pad_reasoning_content(FLASH, &mut body);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[2]["reasoning_content"], "kept");
        assert_eq!(msgs[3]["reasoning_content"], PAD);
        for i in [0, 1, 4] {
            assert!(msgs[i].get("reasoning_content").is_none());
        }
    }

    /// Padding exists for one 400, raised on requests that carry `tools`, by
    /// models that echo the field. Everything else has to come back byte for
    /// byte, and the tool-less case matters because the gate now lets through
    /// any id a DeepSeek-based custom provider is pointed at.
    #[test_case(REASONER_ID, true; "the one model that refuses the field")]
    #[test_case(FLASH, false; "a request that never carried tools")]
    fn bodies_outside_the_workaround_are_untouched(model_id: &str, tools: bool) {
        let mut input = tool_call_body();
        if !tools {
            input.as_object_mut().unwrap().remove("tools");
        }
        let mut body = input.clone();
        pad_reasoning_content(model_id, &mut body);
        assert_eq!(body, input);
    }

    /// The gate the rename broke once already: it has to key off the one id that
    /// refuses the field, never off a version marker in the others.
    #[test_case(FLASH, true; "current flash")]
    #[test_case("deepseek-v9-turbo", true; "a release the table has never seen")]
    #[test_case(REASONER_ID, false; "the one model that refuses it")]
    fn only_the_legacy_reasoner_is_outside_the_v4_protocol(model_id: &str, expected: bool) {
        assert_eq!(uses_v4_thinking_protocol(model_id), expected);
    }
}

/// The port of DeepSeek onto [`decl`] plus [`hooks`], proved one recorded
/// exchange at a time: every fixture runs through the declaration and through
/// the bespoke [`DeepSeek`] above, and both are held against the same golden
/// artifact.
///
/// The two implementations reach the thinking toggle from opposite ends. The
/// bespoke one writes `thinking` first and only then asks for an effort string,
/// skipping that call outright when thinking is off; the declared one applies
/// the effort string in the codec, before the hook that writes `thinking` has
/// run. The fixtures cover every mode so the goldens answer whether that is
/// observable, rather than an argument about [`dialect::DEEPSEEK`] doing so.
///
/// The goldens outlive the comparison. When the bespoke impl goes, the second
/// half of each test goes with it and every case is still pinned to what this
/// provider put on the wire on the day it was ported.
#[cfg(test)]
mod replay_tests {
    use serde_json::Value;
    use tempfile::TempDir;
    use test_case::test_case;

    use crate::model::Model;
    use crate::providers::plugin::{self, DeclSource, Registration};
    use crate::providers::replay::{self, Fixture};
    use crate::test_support::Canned;
    use crate::types::ContentBlock;
    use crate::{Effort, Message, Role, ThinkingConfig};

    use super::{DeepSeek, ENV_VAR, SLUG, Timeouts};

    const FLASH_SPEC: &str = "deepseek/deepseek-flash";
    /// The one id outside the V4 thinking protocol, and not in the curated
    /// table, so it also stands for any id a DeepSeek-based custom provider is
    /// pointed at.
    const REASONER_SPEC: &str = "deepseek/deepseek-reasoner";
    const BASE_URL_ENV: &str = "DEEPSEEK_BASE_URL";
    const API_KEY: &str = "sk-replay";
    /// DeepSeek accepts `max` and nothing else, so a level below it is what
    /// proves the dialect snaps rather than passes through.
    const EFFORT: Effort = Effort::High;
    const BUDGET_TOKENS: u32 = 2048;

    const TEMPDIR_FAILED: &str = "no temporary state directory";
    const UNKNOWN_MODEL: &str = "the model spec did not resolve";
    const REGISTER_FAILED: &str = "the declaration was rejected";
    const CREATE_FAILED: &str = "the provider could not be built";

    const FIRST_ASK: &str = "read a.txt";
    const KEPT_REASONING: &str = "a.txt first";
    const REPLY: &str = "on it";
    const TOOL_ID: &str = "call_1";
    const TOOL_INPUT_PATH: &str = "a.txt";
    const TOOL_OUTPUT: &str = "contents of a.txt";
    const FOLLOW_UP: &str = "now read b.txt";

    const UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"invalid api key"}}"#;
    const RATE_LIMITED_BODY: &str = r#"{"error":{"message":"too many requests"}}"#;
    const SERVER_ERROR_BODY: &str = r#"{"error":{"message":"internal error"}}"#;
    const RETRY_AFTER_HEADERS: &[(&str, &str)] =
        &[("content-type", "application/json"), ("retry-after", "7")];
    const BALANCE_BODY: &str = r#"{"is_available":true,"balance_infos":[{"currency":"USD","total_balance":"12.34","granted_balance":"2.00","topped_up_balance":"10.34"},{"currency":"CNY","total_balance":"88.00","granted_balance":"0.00","topped_up_balance":"88.00"}]}"#;

    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"reasoning_content":"weighing the options"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"b.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"prompt_cache_hit_tokens":4}}

data: [DONE]

"#;

    /// One unparseable frame between two good ones: the bad frame is skipped
    /// and the turn still ends, rather than the whole stream failing.
    const MALFORMED_TRANSCRIPT: &str = r#"data: {"choices": [ this is not json

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: [DONE]

"#;

    /// An error frame on a 200, carrying a tag but no message. The substituted
    /// message is a real bug fix (`EMPTY_SSE_ERROR_MESSAGE`): without it the
    /// turn ended with an empty assistant message and no retry.
    const EMPTY_ERROR_TRANSCRIPT: &str = r#"data: {"error":{"type":"server_error","message":""}}

"#;

    /// Ends mid-frame, with no `finish_reason` and no `[DONE]`.
    const TRUNCATED_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"con"#;

    const SUCCESS_SCRIPT: &[Canned] = &[Canned::sse(SUCCESS_TRANSCRIPT)];
    const MALFORMED_SCRIPT: &[Canned] = &[Canned::sse(MALFORMED_TRANSCRIPT)];
    const EMPTY_ERROR_SCRIPT: &[Canned] = &[Canned::sse(EMPTY_ERROR_TRANSCRIPT)];
    const TRUNCATED_SCRIPT: &[Canned] = &[Canned::sse(TRUNCATED_TRANSCRIPT)];
    /// A second answer neither side is expected to ask for: a run that replays
    /// the rejected key is recorded as a second request rather than parking on
    /// an `accept` that never returns.
    const UNAUTHORIZED_SCRIPT: &[Canned] = &[
        Canned::json(401, UNAUTHORIZED_BODY),
        Canned::json(401, UNAUTHORIZED_BODY),
    ];
    const RATE_LIMITED_SCRIPT: &[Canned] = &[Canned::json(429, RATE_LIMITED_BODY)];
    const SLOW_DOWN_SCRIPT: &[Canned] = &[Canned {
        status: 429,
        headers: RETRY_AFTER_HEADERS,
        body: RATE_LIMITED_BODY,
    }];
    const SERVER_ERROR_SCRIPT: &[Canned] = &[Canned::json(500, SERVER_ERROR_BODY)];
    const BALANCE_SCRIPT: &[Canned] = &[Canned::json(200, BALANCE_BODY)];

    const SUCCESS: Fixture = Fixture {
        name: "success",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
    };
    const UNAUTHORIZED: Fixture = Fixture {
        name: "unauthorized",
        script: UNAUTHORIZED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const SLOW_DOWN: Fixture = Fixture {
        name: "rate_limited_with_retry_after",
        script: SLOW_DOWN_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const RATE_LIMITED: Fixture = Fixture {
        name: "rate_limited",
        script: RATE_LIMITED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const SERVER_ERROR: Fixture = Fixture {
        name: "server_error",
        script: SERVER_ERROR_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const MALFORMED: Fixture = Fixture {
        name: "malformed_sse",
        script: MALFORMED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const EMPTY_ERROR: Fixture = Fixture {
        name: "empty_sse_error_frame",
        script: EMPTY_ERROR_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const TRUNCATED: Fixture = Fixture {
        name: "truncated_stream",
        script: TRUNCATED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    /// The mode the bespoke impl never calls `apply_reasoning_effort` for at
    /// all, against a declaration that always does.
    const THINKING_OFF: Fixture = Fixture {
        name: "thinking_off",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    /// The other silent mode: [`dialect::DEEPSEEK`] declares no adaptive
    /// string, so the API's own default depth is what an empty
    /// `reasoning_effort` asks for.
    const THINKING_ADAPTIVE: Fixture = Fixture {
        name: "thinking_adaptive",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Adaptive,
    };
    /// DeepSeek has no token budget to spend, so a budget resolves to a level
    /// like any other and the count reaches the wire nowhere.
    const THINKING_BUDGET: Fixture = Fixture {
        name: "thinking_budget",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Budget(BUDGET_TOKENS),
    };
    const PADDED: Fixture = Fixture {
        name: "padding_with_tools",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
    };
    const PADDED_WITHOUT_TOOLS: Fixture = Fixture {
        name: "padding_without_tools",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
    };
    const REASONER_PADDED: Fixture = Fixture {
        name: "reasoner_with_tools",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
    };
    const REASONER_WITHOUT_TOOLS: Fixture = Fixture {
        name: "reasoner_without_tools",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
    };
    const BALANCE: Fixture = Fixture {
        name: "user_balance",
        script: BALANCE_SCRIPT,
        thinking: ThinkingConfig::Off,
    };

    /// Points every base directory at a throwaway tree and publishes the key
    /// both sides resolve, so neither reads this machine's credentials,
    /// `providers.toml` or saved origins.
    fn isolated() -> TempDir {
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
        unsafe { std::env::set_var(ENV_VAR, API_KEY) };
        dir
    }

    /// The one door onto the recorded server that both sides go through.
    ///
    /// The base-url precedence is `auth.base_url` > `<SLUG>_BASE_URL` /
    /// `providers.toml` > the static default, and only the middle rung is open
    /// to both: the declaration's static `base_url` is the codec's *last*
    /// resort, so writing loopback there would mean registering a declaration
    /// that is not the one being ported, and `auth.base_url` is only ever
    /// written by an auth hook, which this declaration has none of.
    fn point_at(base_url: &str) {
        unsafe { std::env::set_var(BASE_URL_ENV, base_url) };
    }

    /// The real construction path rather than a hand-built
    /// [`crate::providers::codec::CodecOptions`]: `create` resolves the
    /// *inherited* `api_key_env` into a key pool eagerly, so the claim on the
    /// built-in slug is exercised instead of assumed.
    fn register_decl() {
        plugin::begin_load();
        plugin::register(
            Registration {
                decl: super::decl(),
                hooks: super::hooks(),
            },
            DeclSource::Rust,
        )
        .expect(REGISTER_FAILED);
        plugin::commit_load();
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
            ..Default::default()
        }
    }

    /// A history with one of each turn the padding has an opinion about: an
    /// assistant reply that already carries reasoning, an assistant turn that
    /// is nothing but a tool call, and the tool result and user turns that must
    /// come back untouched.
    fn history() -> Vec<Message> {
        vec![
            Message::user(FIRST_ASK.to_owned()),
            assistant(vec![
                ContentBlock::Thinking {
                    thinking: KEPT_REASONING.to_owned(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: REPLY.to_owned(),
                },
            ]),
            assistant(vec![ContentBlock::ToolUse {
                id: TOOL_ID.to_owned(),
                name: "read".to_owned(),
                input: serde_json::json!({"path": TOOL_INPUT_PATH}),
                thought_signature: None,
            }]),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: TOOL_ID.to_owned(),
                    content: TOOL_OUTPUT.to_owned(),
                    is_error: false,
                }],
                ..Default::default()
            },
            Message::user(FOLLOW_UP.to_owned()),
        ]
    }

    fn model(spec: &str) -> Model {
        Model::from_spec(spec).expect(UNKNOWN_MODEL)
    }

    /// The registry, the environment and the credential store are all
    /// process-global; `cargo nextest` gives each case its own process, which
    /// is what keeps one fixture's origin out of the next one's.
    #[test_case(&SUCCESS ; "success")]
    #[test_case(&UNAUTHORIZED ; "unauthorized")]
    #[test_case(&SLOW_DOWN ; "rate_limited_with_retry_after")]
    #[test_case(&RATE_LIMITED ; "rate_limited")]
    #[test_case(&SERVER_ERROR ; "server_error")]
    #[test_case(&MALFORMED ; "malformed_sse")]
    #[test_case(&EMPTY_ERROR ; "empty_sse_error_frame")]
    #[test_case(&TRUNCATED ; "truncated_stream")]
    #[test_case(&THINKING_OFF ; "thinking_off")]
    #[test_case(&THINKING_ADAPTIVE ; "thinking_adaptive")]
    #[test_case(&THINKING_BUDGET ; "thinking_budget")]
    fn the_declaration_replays_the_bespoke_impl(fixture: &Fixture) {
        let _isolated = isolated();
        let model = model(FLASH_SPEC);
        register_decl();

        let declared = replay::run(fixture, &model, |base_url| {
            point_at(base_url);
            plugin::create(SLUG, Timeouts::default()).expect(CREATE_FAILED)
        });
        replay::assert_golden(SLUG, fixture, &declared);

        let bespoke = replay::run(fixture, &model, |base_url| {
            point_at(base_url);
            Box::new(DeepSeek::new(Timeouts::default()).expect(CREATE_FAILED))
        });
        replay::assert_ported(fixture, &declared, &bespoke);
    }

    /// The `reasoning_content` back-fill, which needs a history to act on and a
    /// tool list to be allowed to: the plain fixtures above send one user turn
    /// and could not tell padding from its absence.
    #[test_case(&PADDED, FLASH_SPEC, true ; "padding_with_tools")]
    #[test_case(&PADDED_WITHOUT_TOOLS, FLASH_SPEC, false ; "padding_without_tools")]
    #[test_case(&REASONER_PADDED, REASONER_SPEC, true ; "reasoner_with_tools")]
    #[test_case(&REASONER_WITHOUT_TOOLS, REASONER_SPEC, false ; "reasoner_without_tools")]
    fn the_declaration_pads_the_same_turns(fixture: &Fixture, spec: &str, with_tools: bool) {
        let _isolated = isolated();
        let model = model(spec);
        let messages = history();
        let tools = if with_tools {
            replay::tools()
        } else {
            Value::Array(Vec::new())
        };
        register_decl();

        let declared = replay::run_with(fixture, &model, &messages, &tools, |base_url| {
            point_at(base_url);
            plugin::create(SLUG, Timeouts::default()).expect(CREATE_FAILED)
        });
        replay::assert_golden(SLUG, fixture, &declared);

        let bespoke = replay::run_with(fixture, &model, &messages, &tools, |base_url| {
            point_at(base_url);
            Box::new(DeepSeek::new(Timeouts::default()).expect(CREATE_FAILED))
        });
        replay::assert_ported(fixture, &declared, &bespoke);
    }

    /// The balance endpoint, which only the declaration can be pointed at a
    /// recorded server: the bespoke impl below sends this one request to a
    /// hard-coded `https://api.deepseek.com`, ignoring the `DEEPSEEK_BASE_URL`
    /// every other request to the slug honours. So the golden pins the port,
    /// and there is no second side to compare it against -- the difference is
    /// the bespoke impl's, and it is real: a user who points the slug at a
    /// gateway has their balance read straight from DeepSeek with the key the
    /// gateway was supposed to hold.
    #[test]
    fn the_declaration_reads_the_balance_endpoint() {
        let _isolated = isolated();
        register_decl();

        let declared = replay::run_usage(&BALANCE, |base_url| {
            point_at(base_url);
            plugin::create(SLUG, Timeouts::default()).expect(CREATE_FAILED)
        });
        replay::assert_golden(SLUG, &BALANCE, &declared);
    }
}
