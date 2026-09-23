use std::borrow::Cow;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use maki_config::providers::Protocol;

use crate::model::ModelFamily;
use crate::pricing::{PricingSchedule, PricingWindow};
use crate::provider::BoxFuture;
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, ProviderSpec};
use crate::types::{ProviderUsage, THINKING_OFF, UsageLimit};
use crate::{AgentError, dialect};

use super::Timeouts;
use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use super::plugin::{
    self, BodyInput, EffortField, Hook, OpenAiWire, ProviderDecl, ProviderHooks, ThinkingWire,
};

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
    native: None,
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

inventory::submit!(SPEC.config_row());

/// DeepSeek as a declaration, plus the two things the openai codec cannot
/// spell, see [`hooks`].
///
/// Only what the codec cannot guess is stated here. Claiming a built-in slug
/// inherits the whole [`SPEC`] row, and `max_tokens` and streamed usage are
/// already the codec's defaults, so restating either would leave a second copy
/// for every later port to keep in step.
///
/// The bundled `deepseek` Lua plugin says all of this again on the surface a
/// third-party plugin uses, and outranks this at every real startup.
pub(crate) fn decl() -> ProviderDecl {
    ProviderDecl {
        slug: SLUG.to_owned(),
        display_name: None,
        codec: Some(Protocol::Openai),
        base: None,
        base_url: Some(BASE_URL.to_owned()),
        api_key_env: None,
        system_prefix: None,
        models: Vec::new(),
        openai: Some(OpenAiWire {
            thinking: Some(ThinkingWire {
                dialect: &dialect::DEEPSEEK,
                field: EffortField::default(),
                requires_support: false,
            }),
            ..OpenAiWire::default()
        }),
        net_hosts: vec![NET_HOST.to_owned()],
    }
}

/// The two callbacks [`decl`] cannot spell, registered alongside it.
pub(crate) fn hooks() -> ProviderHooks {
    ProviderHooks {
        build_body: Some(Arc::new(ThinkingToggle)),
        fetch_usage: Some(Arc::new(Balance)),
        ..ProviderHooks::default()
    }
}

/// DeepSeek reasons unless it is told not to, so every request carries the
/// toggle, and a toggled-on request carries the padding below.
///
/// `thinking` arrives already rendered by the dialect, and `off` is the one
/// rendering that means disabled: every effort level spells itself, `adaptive`
/// spells itself and a budget arrives as its bare token count.
struct ThinkingToggle;

impl Hook<BodyInput, Value> for ThinkingToggle {
    fn call(&self, input: BodyInput) -> BoxFuture<'_, Result<Value, AgentError>> {
        Box::pin(async move {
            let BodyInput {
                mut body,
                model,
                thinking,
                model_info: _,
            } = input;
            let enabled = thinking != THINKING_OFF;
            let mode = if enabled {
                THINKING_ENABLED
            } else {
                THINKING_DISABLED
            };
            body[THINKING_FIELD] = json!({ "type": mode });
            if enabled {
                pad_reasoning_content(&model, &mut body);
            }
            Ok(body)
        })
    }
}

/// DeepSeek's balance endpoint, which sits off every codec's request path.
///
/// The hook is handed no credentials, so it reads the ones the registration
/// resolved for this slug, and the origin the way every other request to the
/// slug resolves it: an auth-supplied base url, then the user's
/// `DEEPSEEK_BASE_URL` / `providers.toml`, then the declared default. The
/// retired bespoke impl posted to a hard-coded `https://api.deepseek.com`, so
/// a user who pointed the slug at a gateway had their balance read straight
/// from DeepSeek with the key the gateway was meant to hold.
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
        msg["reasoning_content"] = PAD.into();
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
    /// The assistant turn of [`tool_call_body`] that carries no reasoning.
    const TOOL_ONLY_TURN: usize = 3;
    /// The hours, days and surcharge as the pricing page states them.
    const PUBLISHED_PEAK_HOURS: &str = "2x during 01:00-04:00, 06:00-10:00 UTC, Mon-Fri";

    /// A schedule that never got hooked up to the spec looks exactly like
    /// off-peak all day, so the assert goes through the registry the biller
    /// reads. Drift on either side bills every DeepSeek turn at the wrong rate.
    #[test]
    fn the_manifest_bills_the_published_peak_hours() {
        let schedule = ProviderRegistry::get(SLUG)
            .expect("deepseek is a builtin")
            .pricing_schedule
            .expect("deepseek bills by the clock");
        assert_eq!(schedule.to_string(), PUBLISHED_PEAK_HOURS);
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

    /// Padding exists for one 400, which DeepSeek raises when a request carries
    /// `tools` and a model that echoes the field gets a turn without it. Only
    /// the tool-only turn gains anything, and whole bodies are compared so the
    /// turns that must come back untouched stay covered too.
    #[test_case(FLASH, true, Some(TOOL_ONLY_TURN); "a turn with no reasoning gets some")]
    #[test_case(REASONER_ID, true, None; "the one model that refuses the field")]
    #[test_case(FLASH, false, None; "a request that never carried tools")]
    fn pads_only_assistant_turns_without_reasoning(
        model_id: &str,
        tools: bool,
        padded: Option<usize>,
    ) {
        let mut expected = tool_call_body();
        if !tools {
            expected.as_object_mut().unwrap().remove("tools");
        }
        let mut body = expected.clone();
        pad_reasoning_content(model_id, &mut body);
        if let Some(turn) = padded {
            expected["messages"][turn]["reasoning_content"] = PAD.into();
        }
        assert_eq!(body, expected);
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

/// The recorded cases, kept out of the test module so both authorings replay
/// the same list: [`decl`] plus [`hooks`], and the bundled `deepseek` Lua
/// plugin that outranks them at every real startup.
///
/// Every case was recorded while the bespoke `impl Provider` this module used
/// to hold was still here, with both sides run against the same artifact. That
/// impl wrote `thinking` first and asked for an effort string second, where the
/// declared path applies the effort in the codec before the hook that writes
/// `thinking` runs. Whether that swap shows on the wire is a question for the
/// goldens, which is why every thinking mode has one. The impl is gone and the
/// artifacts are not.
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures {
    use serde_json::json;

    use crate::model::Model;
    use crate::providers::replay::Fixture;
    use crate::test_support::Canned;
    use crate::types::ContentBlock;
    use crate::{Effort, Message, Role, ThinkingConfig};

    pub const FLASH_SPEC: &str = "deepseek/deepseek-flash";
    /// The one id outside the V4 thinking protocol, and not in the curated
    /// table, so it also stands for any id a DeepSeek-based custom provider is
    /// pointed at.
    pub const REASONER_SPEC: &str = "deepseek/deepseek-reasoner";
    /// DeepSeek accepts `max` and nothing else, so a level below it is what
    /// proves the dialect snaps rather than passes through.
    pub const EFFORT: Effort = Effort::High;

    const UNKNOWN_MODEL: &str = "the model spec did not resolve";

    const FIRST_ASK: &str = "read a.txt";
    const KEPT_REASONING: &str = "a.txt first";
    const REPLY: &str = "on it";
    const TOOL_ID: &str = "call_1";
    const TOOL_NAME: &str = "read";
    const TOOL_INPUT_PATH: &str = "a.txt";
    const TOOL_OUTPUT: &str = "contents of a.txt";
    const FOLLOW_UP: &str = "now read b.txt";

    const BALANCE_UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"Authentication Fails, Your api key: ****abcd is invalid","type":"authentication_error","param":null,"code":"invalid_request_error"}}"#;
    const BALANCE_BODY: &str = r#"{"is_available":true,"balance_infos":[{"currency":"USD","total_balance":"12.34","granted_balance":"2.00","topped_up_balance":"10.34"},{"currency":"CNY","total_balance":"88.00","granted_balance":"0.00","topped_up_balance":"88.00"}]}"#;

    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"reasoning_content":"weighing the options"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"b.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"prompt_cache_hit_tokens":4}}

data: [DONE]

"#;

    pub const SUCCESS_SCRIPT: &[Canned] = &[Canned::sse(SUCCESS_TRANSCRIPT)];

    pub const SUCCESS: Fixture = Fixture {
        name: "success",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
        session: None,
    };
    /// The mode the toggle has to spell out, since DeepSeek reasons unless it
    /// is told not to.
    pub const THINKING_OFF: Fixture = Fixture {
        name: "thinking_off",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// The quiet one: [`crate::dialect::DEEPSEEK`] declares no adaptive
    /// string, so thinking is switched on while `reasoning_effort` stays off
    /// the wire and the API picks its own depth.
    pub const THINKING_ADAPTIVE: Fixture = Fixture {
        name: "thinking_adaptive",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Adaptive,
        session: None,
    };
    pub const BALANCE: Fixture = Fixture {
        name: "user_balance",
        script: &[Canned::json(200, BALANCE_BODY)],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// A refused balance request. Both authorings must fail with the same `Api`
    /// error, so the plugin cannot pass by calling it a broken hook.
    pub const USER_BALANCE_UNAUTHORIZED: Fixture = Fixture {
        name: "user_balance_unauthorized",
        script: &[Canned::json(401, BALANCE_UNAUTHORIZED_BODY)],
        thinking: ThinkingConfig::Off,
        session: None,
    };

    /// The padding cases, which every authoring replays against the same
    /// recording. Only the request in front of it changes, so the name is what
    /// tells the goldens apart.
    pub fn padding(name: &'static str) -> Fixture {
        Fixture {
            name,
            script: SUCCESS_SCRIPT,
            thinking: ThinkingConfig::Effort(EFFORT),
            session: None,
        }
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
    pub fn history() -> Vec<Message> {
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
                name: TOOL_NAME.to_owned(),
                input: json!({"path": TOOL_INPUT_PATH}),
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

    pub fn model(spec: &str) -> Model {
        Model::from_spec(spec).expect(UNKNOWN_MODEL)
    }
}

/// DeepSeek as [`decl`] plus [`hooks`] put it on the wire, one recorded
/// exchange at a time.
#[cfg(test)]
mod replay_tests {
    use serde_json::Value;
    use test_case::test_case;

    use crate::providers::replay::{self, Fixture};

    use super::SLUG;
    use super::fixtures::*;

    #[test_case(&SUCCESS ; "success")]
    #[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
    #[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
    #[test_case(&replay::RATE_LIMITED ; "rate_limited")]
    #[test_case(&replay::SERVER_ERROR ; "server_error")]
    #[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
    #[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
    #[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
    #[test_case(&THINKING_OFF ; "thinking_off")]
    #[test_case(&THINKING_ADAPTIVE ; "thinking_adaptive")]
    fn the_declaration_replays_the_recorded_exchange(fixture: &Fixture) {
        replay::declared(replay::rust_authoring, SLUG, fixture, &model(FLASH_SPEC));
    }

    /// The `reasoning_content` back-fill on the wire, which needs a history to
    /// act on and a tool list to be allowed to: the fixtures above send one
    /// user turn and could not tell padding from its absence.
    #[test_case("padding_with_tools", FLASH_SPEC, true ; "a turn with no reasoning gets some")]
    #[test_case("padding_without_tools", FLASH_SPEC, false ; "no tools, nothing added")]
    #[test_case("reasoner_with_tools", REASONER_SPEC, true ; "the model that refuses the field")]
    fn the_declaration_pads_the_same_turns(name: &'static str, spec: &str, with_tools: bool) {
        let tools = if with_tools {
            replay::tools()
        } else {
            Value::Array(Vec::new())
        };
        replay::declared_with(
            replay::rust_authoring,
            SLUG,
            &padding(name),
            &model(spec),
            &history(),
            &tools,
        );
    }

    /// The golden pins the origin the balance request goes to, which is the one
    /// thing the port changed about it, see [`super::Balance`], and the error a
    /// refused request fails with.
    #[test_case(&BALANCE ; "balance")]
    #[test_case(&USER_BALANCE_UNAUTHORIZED ; "unauthorized")]
    fn the_declaration_reads_the_balance_endpoint(fixture: &Fixture) {
        replay::declared_usage(replay::rust_authoring, SLUG, fixture);
    }
}
