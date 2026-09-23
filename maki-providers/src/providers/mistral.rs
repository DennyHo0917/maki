use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use flume::Sender;
use maki_storage::id::SessionRef;
use serde_json::{Value, json};

use maki_config::providers::{Protocol, ProviderPlan};

use crate::model::{Model, ModelFamily, ThinkingSupport};
use crate::provider::{BoxFuture, Provider};
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{
    ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, Native, ProviderSpec,
};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse, dialect};

use super::openai_compat::{MODELS_PATH, OpenAiCompatConfig, OpenAiCompatProvider};
use super::{KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts};

const SLUG: &str = "mistral";
const DISPLAY_NAME: &str = "Mistral";
const ENV_VAR: &str = "MISTRAL_API_KEY";
const BASE_URL: &str = "https://api.mistral.ai/v1";
const DEFAULT_MODEL: &str = "mistral/mistral-medium-latest";
const CODING_MODEL: &str = "mistral/mistral-vibe-cli-latest";
const LOGIN_URL: &str = "https://admin.mistral.ai/organization/api-keys";
const MAX_TOKENS_FIELD: &str = "max_tokens";

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

fn create(timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    Ok(Box::new(Mistral::new(timeouts)?))
}

fn create_with_auth(
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
    system_prefix: Option<String>,
) -> Box<dyn Provider> {
    Box::new(Mistral::with_auth(auth, timeouts).with_system_prefix(system_prefix))
}

inventory::submit!(SPEC.config_row());

pub struct Mistral {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    key_pool: Option<KeyPool>,
    system_prefix: Option<String>,
}

fn convert_assistant_messages_in_place(messages: &mut Value) {
    if let Some(msgs) = messages.as_array_mut() {
        for msg in msgs {
            if let Some(obj) = msg.as_object_mut()
                && obj.get("role").and_then(Value::as_str) == Some("assistant")
            {
                let Some(reasoning_val) = obj.remove("reasoning_content") else {
                    continue;
                };
                let Some(reasoning_text) = reasoning_val.as_str() else {
                    continue;
                };

                let thinking_block = json!({
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": reasoning_text}]
                });

                if let Some(content) = obj.get_mut("content") {
                    if let Some(content_str) = content.as_str()
                        && !content_str.is_empty()
                    {
                        // Has text content, create array with both
                        let text_content = json!({"type": "text", "text": content_str});
                        *content = json!([thinking_block, text_content]);
                    } else if content.is_string() {
                        // Empty string content, just use thinking
                        *content = json!([thinking_block]);
                    } else if let Some(arr) = content.as_array_mut() {
                        // Already an array, prepend thinking
                        arr.insert(0, thinking_block);
                    } else {
                        *content = json!([thinking_block]);
                    }
                } else {
                    obj.insert("content".to_string(), json!([thinking_block]));
                }
            }
        }
    }
}

impl Mistral {
    pub fn new(timeouts: super::Timeouts) -> Result<Self, AgentError> {
        let pool = KeyPool::resolve("mistral", &CONFIG.api_key_env)?;
        Ok(Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth: Arc::new(Mutex::new(ResolvedAuth::bearer("mistral", pool.current())?)),
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

impl Provider for Mistral {
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
            let auth = self.auth.lock().unwrap().clone();
            let mut buf = String::new();
            let system = super::with_prefix(&self.system_prefix, system, &mut buf);
            let mut body = self.compat.build_body(model, messages, system, tools);
            opts.thinking
                .apply_reasoning_effort(&mut body, &dialect::HIGH_ONLY, model);
            // Convert assistant messages to Mistral's expected format with thinking content
            convert_assistant_messages_in_place(body.get_mut("messages").unwrap());

            let mut extra_headers = vec![];
            if let Some(session_id) = session_id {
                extra_headers.push(("x-affinity", session_id.as_str()));
            }
            self.compat
                .do_stream(model, &extra_headers, &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<crate::model::ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            self.compat
                .fetch_and_parse_models(&auth, MODELS_PATH, |m| {
                    // Filter: only completion_chat capable models
                    let has_completion_chat = m
                        .get("capabilities")
                        .and_then(Value::as_object)
                        .and_then(|c| c.get("completion_chat"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    if !has_completion_chat {
                        return None;
                    }

                    // Parse with Mistral-specific field names
                    let id = m["id"].as_str()?;
                    let context_window = m["max_context_length"]
                        .as_u64()
                        .and_then(|v| u32::try_from(v).ok());
                    let supports_thinking = m
                        .get("capabilities")
                        .and_then(Value::as_object)
                        .and_then(|c| c.get("reasoning"))
                        .and_then(Value::as_bool);
                    let supports_vision = m
                        .get("capabilities")
                        .and_then(Value::as_object)
                        .and_then(|c| c.get("vision"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    Some(crate::model::ModelInfo {
                        id: id.to_string(),
                        context_window,
                        max_output_tokens: None,
                        pricing: None,
                        supports_thinking,
                        supports_vision: Some(supports_vision),
                        tier: None,
                        provider_info: None,
                    })
                })
                .await
        })
    }

    fn keys(&self) -> Option<KeyRotation<'_>> {
        Some(KeyRotation::new(
            self.key_pool.as_ref()?,
            &self.auth,
            KeyHeader::Bearer,
        ))
    }

    fn adjust_model(&self, model: &mut Model) {
        adjust_model(model);
    }
}

fn adjust_model(model: &mut Model) {
    if model.id.starts_with("ministral-") {
        model.thinking_override = Some(ThinkingSupport::No);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use test_case::test_case;

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
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "text"}
        ]),
        json!([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "text"}
        ])
        ; "assistant_text_only_no_thinking"
    )]
    fn convert_assistant_messages_in_place_test(input: Value, expected: Value) {
        let mut input_clone = input.clone();
        convert_assistant_messages_in_place(&mut input_clone);
        assert_eq!(input_clone, expected);
    }

    #[test_case("mistral/ministral-14b-latest", false ; "ministral_no_thinking")]
    #[test_case("mistral/mistral-medium-latest", true ; "mistral_medium_supports_thinking")]
    fn adjust_model_sets_thinking_support(spec: &str, expected: bool) {
        let mut model = Model::from_spec(spec).unwrap();
        adjust_model(&mut model);
        assert_eq!(model.supports_thinking(), expected);
    }
}

/// The recorded cases, kept out of the test module so both authorings replay
/// the same list once the port lands. Every golden is recorded against the
/// bespoke [`Mistral`] impl above.
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

/// Mistral as the bespoke impl puts it on the wire, recorded for the port.
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
    fn the_bespoke_impl_records_the_exchange(fixture: &Fixture) {
        replay::bespoke(SLUG).stream(fixture, &model());
    }

    #[test]
    fn the_bespoke_impl_records_the_history() {
        replay::bespoke(SLUG).with(&HISTORY, &model(), &history(), &replay::tools());
    }

    #[test_case(&MODELS ; "models")]
    #[test_case(&MODELS_UNAUTHORIZED ; "models_unauthorized")]
    fn the_bespoke_impl_records_the_listing(fixture: &Fixture) {
        replay::bespoke(SLUG).models(fixture);
    }
}
