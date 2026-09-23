use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use isahc::http::HeaderName;
use serde_json::{Value, json};

use maki_config::providers::{Protocol, ProviderPlan};

use crate::model::{ModelFamily, ModelInfo, ThinkingSupport};
use crate::provider::BoxFuture;
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, ProviderSpec};
use crate::{AgentError, dialect};

use super::Timeouts;
use super::openai_compat::{MODELS_PATH, OpenAiCompatConfig, OpenAiCompatProvider};
use super::plugin::{
    self, BodyInput, EffortField, Hook, OpenAiWire, ProviderDecl, ProviderHooks, SessionCarrier,
    ThinkingWire,
};

const SLUG: &str = "mistral";
const DISPLAY_NAME: &str = "Mistral";
const ENV_VAR: &str = "MISTRAL_API_KEY";
const BASE_URL: &str = "https://api.mistral.ai/v1";
const DEFAULT_MODEL: &str = "mistral/mistral-medium-latest";
const CODING_MODEL: &str = "mistral/mistral-vibe-cli-latest";
const LOGIN_URL: &str = "https://admin.mistral.ai/organization/api-keys";
const MAX_TOKENS_FIELD: &str = "max_tokens";
const NET_HOST: &str = "api.mistral.ai";
const AFFINITY_HEADER: &str = "x-affinity";
/// Mistral's small models, which the API refuses reasoning for whatever the
/// model table says.
const NO_THINKING_PREFIX: &str = "ministral-";

const MESSAGES_FIELD: &str = "messages";
const ROLE_FIELD: &str = "role";
const ASSISTANT_ROLE: &str = "assistant";
const REASONING_FIELD: &str = "reasoning_content";
const CONTENT_FIELD: &str = "content";
const ID_FIELD: &str = "id";
const CAPABILITIES_FIELD: &str = "capabilities";
const COMPLETION_CHAT_CAPABILITY: &str = "completion_chat";
const REASONING_CAPABILITY: &str = "reasoning";
const VISION_CAPABILITY: &str = "vision";
const CONTEXT_WINDOW_FIELD: &str = "max_context_length";

static CONFIG: OpenAiCompatConfig = OpenAiCompatConfig {
    slug: Cow::Borrowed(SLUG),
    api_key_env: Cow::Borrowed(ENV_VAR),
    base_url: Cow::Borrowed(BASE_URL),
    max_tokens_field: Cow::Borrowed(MAX_TOKENS_FIELD),
    include_stream_usage: true,
    provider_name: Cow::Borrowed(DISPLAY_NAME),
};

const PLANS: &[(&str, ProviderPlan)] = &[
    (
        "standard",
        ProviderPlan {
            display_name: "Standard",
            base_url: BASE_URL,
            default_model: Some(DEFAULT_MODEL),
            login_url: None,
        },
    ),
    (
        "coding",
        ProviderPlan {
            display_name: "Vibe / Coding",
            base_url: BASE_URL,
            default_model: Some(CODING_MODEL),
            login_url: Some("https://console.mistral.ai/codestral/cli"),
        },
    ),
];

pub(crate) const SPEC: ProviderSpec = ProviderSpec {
    slug: SLUG,
    display_name: DISPLAY_NAME,
    api_key_env: ENV_VAR,
    family: ModelFamily::Generic,
    supports_thinking: true,
    accepts_arbitrary_models: true,
    fallback_max_output: None,
    fallback_context_window: 128_000,
    models_toml: include_str!("../../models/mistral.toml"),
    pricing_schedule: None,
    native: None,
    aperture: Some(ApertureRoute {
        path_prefix: DEFAULT_PATH_PREFIX,
    }),
    login: Some(LoginConfig {
        protocol: Protocol::Openai,
        default_base_url: BASE_URL,
        default_model: DEFAULT_MODEL,
        plans: Some(PLANS),
        login_url: Some(LOGIN_URL),
        needs_url: false,
    }),
    docs: GeneratedDocs {
        api_urls: &[BASE_URL],
        features: None,
        auth: AuthDoc::EnvVar,
        catalog: CatalogDoc::Table,
        trailing_notes: &[],
    },
};

inventory::submit!(SPEC.config_row());

/// Mistral as a declaration, plus the two things the openai codec cannot
/// spell, see [`hooks`].
///
/// Only what the codec cannot guess is stated here. Claiming a built-in slug
/// inherits the whole [`SPEC`] row, and `max_tokens` and streamed usage are
/// already the codec's defaults. The small models' refusal to reason is data
/// rather than a hook: `adjust_model` is synchronous, and the codec applies
/// it on aperture's route onto this slug as well.
///
/// The bundled `mistral` Lua plugin says all of this again on the surface a
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
                dialect: &dialect::HIGH_ONLY,
                field: EffortField::default(),
                requires_support: false,
            }),
            session_id: Some(SessionCarrier::Header(HeaderName::from_static(
                AFFINITY_HEADER,
            ))),
            thinking_overrides: BTreeMap::from([(
                NO_THINKING_PREFIX.to_owned(),
                ThinkingSupport::No,
            )]),
            ..OpenAiWire::default()
        }),
        net_hosts: vec![NET_HOST.to_owned()],
    }
}

/// The two callbacks [`decl`] cannot spell, registered alongside it.
pub(crate) fn hooks() -> ProviderHooks {
    ProviderHooks {
        build_body: Some(Arc::new(AssistantThinking)),
        list_models: Some(Arc::new(Catalogue)),
        ..ProviderHooks::default()
    }
}

/// Mistral takes reasoning back as a `thinking` part of the assistant turn's
/// `content`, not as the `reasoning_content` the openai codec writes.
struct AssistantThinking;

impl Hook<BodyInput, Value> for AssistantThinking {
    fn call(&self, input: BodyInput) -> BoxFuture<'_, Result<Value, AgentError>> {
        Box::pin(async move {
            let mut body = input.body;
            if let Some(messages) = body.get_mut(MESSAGES_FIELD) {
                convert_assistant_messages_in_place(messages);
            }
            Ok(body)
        })
    }
}

/// Mistral's `/models`, which lists embedding, OCR and moderation models too
/// and names its fields its own way.
struct Catalogue;

impl Hook<(), Vec<ModelInfo>> for Catalogue {
    fn call(&self, (): ()) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = plugin::registered_auth(SLUG)?;
            let compat = OpenAiCompatProvider::new(&CONFIG, Timeouts::default());
            compat
                .fetch_and_parse_models(&auth, MODELS_PATH, parse_model)
                .await
        })
    }
}

/// Only chat-capable rows survive. `vision` defaults to off, where an unstated
/// `reasoning` stays unstated.
fn parse_model(m: &Value) -> Option<ModelInfo> {
    let capabilities = m.get(CAPABILITIES_FIELD).and_then(Value::as_object)?;
    let capability = |name: &str| capabilities.get(name).and_then(Value::as_bool);
    if capability(COMPLETION_CHAT_CAPABILITY) != Some(true) {
        return None;
    }
    Some(ModelInfo {
        id: m[ID_FIELD].as_str()?.to_owned(),
        context_window: m[CONTEXT_WINDOW_FIELD]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok()),
        supports_thinking: capability(REASONING_CAPABILITY),
        supports_vision: Some(capability(VISION_CAPABILITY).unwrap_or(false)),
        ..ModelInfo::default()
    })
}

/// Moves each assistant turn's string `reasoning_content` to the front of its
/// `content`, as a `thinking` part. A non-string one is dropped.
fn convert_assistant_messages_in_place(messages: &mut Value) {
    let Some(messages) = messages.as_array_mut() else {
        return;
    };
    for msg in messages {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        if obj.get(ROLE_FIELD).and_then(Value::as_str) != Some(ASSISTANT_ROLE) {
            continue;
        }
        let Some(Value::String(reasoning)) = obj.remove(REASONING_FIELD) else {
            continue;
        };
        let thinking = json!({
            "type": "thinking",
            "thinking": [{"type": "text", "text": reasoning}]
        });
        let content = match obj.remove(CONTENT_FIELD) {
            Some(Value::String(text)) if !text.is_empty() => {
                json!([thinking, {"type": "text", "text": text}])
            }
            Some(Value::Array(mut parts)) => {
                parts.insert(0, thinking);
                Value::Array(parts)
            }
            _ => json!([thinking]),
        };
        obj.insert(CONTENT_FIELD.to_owned(), content);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use test_case::test_case;

    use super::*;
    use crate::model::Model;
    use crate::provider::Provider;
    use crate::providers::ResolvedAuth;
    use crate::providers::aperture::Aperture;

    const API_KEY: &str = "sk-test";
    const APERTURE: &str = "aperture";
    const MINISTRAL: &str = "ministral-14b-latest";
    const MEDIUM: &str = "mistral-medium-latest";
    const UNKNOWN_MODEL: &str = "the model spec did not resolve";
    const NOT_BUILT: &str = "mistral did not build from its declaration";

    #[test_case(
        json!([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "text", "reasoning_content": "thinking"}
        ]),
        json!([
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": "thinking"}]},
                    {"type": "text", "text": "text"}
                ]
            }
        ])
        ; "assistant_text_and_thinking"
    )]
    #[test_case(
        json!([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "reasoning_content": "thinking"}
        ]),
        json!([
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": [{"type": "text", "text": "thinking"}]}]
            }
        ])
        ; "assistant_empty_content_with_thinking"
    )]
    #[test_case(
        json!([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "reasoning_content": "thinking"}
        ]),
        json!([
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": [{"type": "text", "text": "thinking"}]}]
            }
        ])
        ; "assistant_no_content_with_thinking"
    )]
    #[test_case(
        json!([
            {"role": "assistant", "content": [{"type": "text", "text": "text"}], "reasoning_content": "thinking"}
        ]),
        json!([
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": "thinking"}]},
                    {"type": "text", "text": "text"}
                ]
            }
        ])
        ; "assistant_array_content_with_thinking"
    )]
    #[test_case(
        json!([
            {"role": "assistant", "content": "text", "reasoning_content": null}
        ]),
        json!([
            {"role": "assistant", "content": "text"}
        ])
        ; "assistant_null_reasoning_dropped"
    )]
    #[test_case(
        json!([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "text"}
        ]),
        json!([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "text"}
        ])
        ; "assistant_text_only_no_thinking"
    )]
    fn convert_assistant_messages_in_place_test(mut messages: Value, expected: Value) {
        convert_assistant_messages_in_place(&mut messages);
        assert_eq!(messages, expected);
    }

    fn declared(model_id: &str) -> Model {
        let mut model = Model::from_spec(&format!("{SLUG}/{model_id}")).expect(UNKNOWN_MODEL);
        plugin::create(SLUG, Timeouts::default())
            .expect(NOT_BUILT)
            .adjust_model(&mut model);
        model
    }

    fn routed(model_id: &str) -> Model {
        let mut model =
            Model::from_spec(&format!("{APERTURE}/{SLUG}/{model_id}")).expect(UNKNOWN_MODEL);
        let auth = Arc::new(Mutex::new(ResolvedAuth::for_test(None, Vec::new())));
        Aperture::with_auth(auth, Timeouts::default()).adjust_model(&mut model);
        model
    }

    /// The override is declared data the codec applies, so aperture's route
    /// onto the slug, which builds that same codec, has to honour it too.
    #[test_case(declared, MINISTRAL, false ; "declared_ministral")]
    #[test_case(routed, MINISTRAL, false ; "aperture_ministral")]
    #[test_case(declared, MEDIUM, true ; "declared_medium")]
    #[test_case(routed, MEDIUM, true ; "aperture_medium")]
    fn only_ministral_is_denied_thinking(
        adjusted: fn(&str) -> Model,
        model_id: &str,
        thinks: bool,
    ) {
        unsafe { std::env::set_var(ENV_VAR, API_KEY) };
        plugin::begin_load();
        plugin::commit_load();

        let model = adjusted(model_id);
        assert_eq!(
            model.thinking_override == Some(ThinkingSupport::No),
            !thinks
        );
        assert_eq!(model.supports_thinking(), thinks);
    }
}

/// The recorded cases, kept out of the test module so both authorings replay
/// the same list: [`decl`] plus [`hooks`], and the bundled `mistral` Lua
/// plugin.
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures {
    use serde_json::json;

    use crate::model::Model;
    use crate::providers::replay::Fixture;
    use crate::test_support::Canned;
    use crate::types::ContentBlock;
    use crate::{Effort, Message, Role, ThinkingConfig};

    const MODEL_SPEC: &str = "mistral/mistral-medium-latest";
    /// The one level [`crate::dialect::HIGH_ONLY`] sends.
    const EFFORT: Effort = Effort::High;
    /// Travels as `x-affinity`, verbatim.
    const SESSION: &str = "0192f0c4-6b1e-7c3a-9d2e-5f8a1b3c4d5e";

    const UNKNOWN_MODEL: &str = "the curated table has no such model";

    const FIRST_ASK: &str = "read a.txt";
    const KEPT_REASONING: &str = "a.txt first";
    const REPLY: &str = "on it";
    const LONE_REASONING: &str = "nothing to say yet";
    const TOOL_NAME: &str = "read";
    const BARE_TOOL_ID: &str = "call_1";
    const BARE_TOOL_PATH: &str = "a.txt";
    const BARE_TOOL_OUTPUT: &str = "contents of a.txt";
    const REASONED_TOOL_ID: &str = "call_2";
    const REASONED_TOOL_PATH: &str = "b.txt";
    const REASONED_TOOL_REASONING: &str = "b.txt next";
    const REASONED_TOOL_OUTPUT: &str = "contents of b.txt";
    const PLAIN_REPLY: &str = "both read";
    const FOLLOW_UP: &str = "now summarise them";

    /// Mistral's own shape for reasoning: `content` as an array of `thinking`
    /// and `text` parts, a thinking part given as a bare string as well as a
    /// block, then a plain string delta and a tool call sent whole.
    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"weighing the options"}]}]}}]}

data: {"choices":[{"delta":{"content":[{"type":"thinking","thinking":[" and a plan"]},{"type":"text","text":"Hello"}]}}]}

data: {"choices":[{"delta":{"content":" there"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"read","arguments":"{\"path\":\"c.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17,"prompt_tokens_details":{"cached_tokens":4}}}

data: [DONE]

"#;

    /// Only `completion_chat` rows survive, sorted by id, and every field the
    /// parser reads is given each shape it must reject: a float, a negative,
    /// a null, a string and one past `u32::MAX` for `max_context_length`
    /// (with `u32::MAX` itself kept), non-bool capability flags, and a
    /// missing or non-string id. The two `mistral-large-latest` rows pin the
    /// stable sort.
    const MODELS_BODY: &str = r#"{"object":"list","data":[
{"id":"mistral-medium-latest","object":"model","capabilities":{"completion_chat":true,"function_calling":true,"reasoning":true,"vision":true},"max_context_length":262144},
{"id":"mistral-large-latest","object":"model","capabilities":{"completion_chat":true,"vision":true},"max_context_length":131072},
{"id":"codestral-latest","object":"model","capabilities":{"completion_chat":true,"reasoning":false,"vision":false},"max_context_length":256000.0},
{"id":"mistral-embed","object":"model","capabilities":{"completion_chat":false},"max_context_length":8192},
{"id":"ministral-14b-latest","object":"model","capabilities":{"completion_chat":true},"max_context_length":-1},
{"id":"mistral-ocr-latest","object":"model","capabilities":{},"max_context_length":32768},
{"id":"magistral-medium-latest","object":"model","capabilities":{"completion_chat":true,"reasoning":true,"vision":"yes"},"max_context_length":4294967296},
{"id":"pixtral-large-latest","object":"model","capabilities":{"completion_chat":"true","vision":true},"max_context_length":131072},
{"id":"mistral-small-latest","object":"model","capabilities":{"completion_chat":true,"reasoning":null,"vision":null},"max_context_length":null},
{"id":"devstral-medium-latest","object":"model","capabilities":{"completion_chat":true,"vision":false},"max_context_length":4294967295},
{"id":"open-mistral-nemo","object":"model","capabilities":{"completion_chat":true,"reasoning":"false"},"max_context_length":"131072"},
{"id":"mistral-moderation-latest","object":"model","max_context_length":8192},
{"object":"model","capabilities":{"completion_chat":true},"max_context_length":32768},
{"id":42,"object":"model","capabilities":{"completion_chat":true},"max_context_length":32768},
{"id":"mistral-large-latest","object":"model","capabilities":{"completion_chat":true,"reasoning":true},"max_context_length":128000}
]}"#;

    const UNAUTHORIZED_BODY: &str = r#"{"message":"Unauthorized","request_id":"req_replay"}"#;

    const SUCCESS_SCRIPT: &[Canned] = &[Canned::sse(SUCCESS_TRANSCRIPT)];

    pub const SUCCESS: Fixture = Fixture {
        name: "success",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
        session: None,
    };
    /// [`crate::dialect::HIGH_ONLY`] has no off string, so nothing is sent.
    pub const THINKING_OFF: Fixture = Fixture {
        name: "thinking_off",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Off,
        session: None,
    };
    pub const IN_SESSION: Fixture = Fixture {
        name: "in_session",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
        session: Some(SESSION),
    };
    /// Replayed with [`history`], so the assistant-turn rewrite has turns to
    /// act on.
    pub const HISTORY: Fixture = Fixture {
        name: "history",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
        session: None,
    };
    pub const MODELS: Fixture = Fixture {
        name: "models",
        script: &[Canned::json(200, MODELS_BODY)],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// The second answer records a retry of the rejected key instead of
    /// parking on it.
    pub const MODELS_UNAUTHORIZED: Fixture = Fixture {
        name: "models_unauthorized",
        script: &[
            Canned::json(401, UNAUTHORIZED_BODY),
            Canned::json(401, UNAUTHORIZED_BODY),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };

    pub fn model() -> Model {
        Model::from_spec(MODEL_SPEC).expect(UNKNOWN_MODEL)
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
            ..Default::default()
        }
    }

    fn tool_use(id: &str, path: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_owned(),
            name: TOOL_NAME.to_owned(),
            input: json!({ "path": path }),
            thought_signature: None,
        }
    }

    fn thinking(text: &str) -> ContentBlock {
        ContentBlock::Thinking {
            thinking: text.to_owned(),
            signature: None,
        }
    }

    fn tool_result(id: &str, output: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.to_owned(),
                content: output.to_owned(),
                is_error: false,
            }],
            ..Default::default()
        }
    }

    /// One of each assistant turn the rewrite has an opinion about: text with
    /// reasoning, reasoning with empty text, a bare tool call, a tool call
    /// with reasoning, and plain text that must come back untouched. Missing
    /// and array `content` never leave the openai codec, so only the unit
    /// tests reach them.
    pub fn history() -> Vec<Message> {
        vec![
            Message::user(FIRST_ASK.to_owned()),
            assistant(vec![
                thinking(KEPT_REASONING),
                ContentBlock::Text {
                    text: REPLY.to_owned(),
                },
            ]),
            assistant(vec![thinking(LONE_REASONING)]),
            assistant(vec![tool_use(BARE_TOOL_ID, BARE_TOOL_PATH)]),
            tool_result(BARE_TOOL_ID, BARE_TOOL_OUTPUT),
            assistant(vec![
                thinking(REASONED_TOOL_REASONING),
                tool_use(REASONED_TOOL_ID, REASONED_TOOL_PATH),
            ]),
            tool_result(REASONED_TOOL_ID, REASONED_TOOL_OUTPUT),
            assistant(vec![ContentBlock::Text {
                text: PLAIN_REPLY.to_owned(),
            }]),
            Message::user(FOLLOW_UP.to_owned()),
        ]
    }
}

/// Mistral as [`decl`] plus [`hooks`] put it on the wire, one recorded
/// exchange at a time.
#[cfg(test)]
mod replay_tests {
    use test_case::test_case;

    use crate::providers::replay::{self, Fixture};

    use super::SLUG;
    use super::fixtures::{
        HISTORY, IN_SESSION, MODELS, MODELS_UNAUTHORIZED, SUCCESS, THINKING_OFF, history, model,
    };

    #[test_case(&SUCCESS ; "success")]
    #[test_case(&THINKING_OFF ; "thinking_off")]
    #[test_case(&IN_SESSION ; "in_session")]
    #[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
    #[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
    #[test_case(&replay::RATE_LIMITED ; "rate_limited")]
    #[test_case(&replay::SERVER_ERROR ; "server_error")]
    #[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
    #[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
    #[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
    fn the_declaration_replays_the_recorded_exchange(fixture: &Fixture) {
        replay::declared(replay::rust_authoring, SLUG, fixture, &model());
    }

    /// The assistant-turn rewrite on the wire, which needs a history to act
    /// on: the fixtures above send one user turn.
    #[test]
    fn the_declaration_rewrites_the_same_turns() {
        replay::declared_with(
            replay::rust_authoring,
            SLUG,
            &HISTORY,
            &model(),
            &history(),
            &replay::tools(),
        );
    }

    #[test_case(&MODELS ; "models")]
    #[test_case(&MODELS_UNAUTHORIZED ; "models_unauthorized")]
    fn the_declaration_lists_the_recorded_models(fixture: &Fixture) {
        replay::declared_models(replay::rust_authoring, SLUG, fixture);
    }
}
