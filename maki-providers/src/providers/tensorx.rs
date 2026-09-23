use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use flume::Sender;
use maki_storage::id::SessionRef;
use serde_json::{Value, json};

use maki_config::providers::Protocol;

use crate::model::{Model, ModelFamily, ModelInfo, ModelPricing};
use crate::provider::{BoxFuture, Provider};
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{
    ApertureRoute, AuthDoc, CatalogDoc, GENERIC_DISCOVERY_NOTE, GeneratedDocs, LoginConfig,
    NO_CURATED_MODELS, Native, ProviderSpec,
};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse, dialect};

use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use super::{KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts, deepseek};

/// TensorX namespaces resold models by vendor, so DeepSeek ids arrive as
/// `deepseek/deepseek-flash`.
const DEEPSEEK_VENDOR_PREFIX: &str = "deepseek/";

const SLUG: &str = "tensorx";
const DISPLAY_NAME: &str = "TensorX";
const ENV_VAR: &str = "TENSORX_API_KEY";
const BASE_URL: &str = "https://api.tensorx.ai/v1";
const DEFAULT_MODEL: &str = "tensorx/z-ai/glm-5.2";
const LOGIN_URL: &str = "https://tensorx.ai";
const MAX_TOKENS_FIELD: &str = "max_tokens";
const FEATURES: &str = "Open-weight models, zero data retention, prompt caching";

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
    accepts_arbitrary_models: true,
    fallback_max_output: None,
    fallback_context_window: 200_000,
    models_toml: NO_CURATED_MODELS,
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
        plans: None,
        login_url: Some(LOGIN_URL),
        needs_url: false,
    }),
    docs: GeneratedDocs {
        api_urls: &[BASE_URL],
        features: Some(FEATURES),
        auth: AuthDoc::EnvVar,
        catalog: CatalogDoc::Discovered(GENERIC_DISCOVERY_NOTE),
        trailing_notes: &[],
    },
};

fn create(timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    Ok(Box::new(TensorX::new(timeouts)?))
}

fn create_with_auth(
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
    system_prefix: Option<String>,
) -> Box<dyn Provider> {
    Box::new(TensorX::with_auth(auth, timeouts).with_system_prefix(system_prefix))
}

inventory::submit!(SPEC.config_row());

#[derive(Debug)]
struct TensorXModelInfo {
    has_thinking: bool,
    has_reasoning_effort: bool,
}

pub struct TensorX {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    key_pool: Option<KeyPool>,
    system_prefix: Option<String>,
}

impl TensorX {
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

impl Provider for TensorX {
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

            let (has_thinking, has_reasoning_effort) =
                crate::model_registry::provider_info::<TensorXModelInfo>("tensorx", &model.id)
                    .map_or((false, false), |info| {
                        (info.has_thinking, info.has_reasoning_effort)
                    });

            if has_thinking {
                body["thinking"] = json!(opts.thinking.is_enabled());
            }
            if has_reasoning_effort {
                opts.thinking
                    .apply_reasoning_effort(&mut body, &dialect::TENSORX, model);
            }
            // DeepSeek takes the toggle through the chat template and TensorX
            // advertises neither knob for it. Sharing DeepSeek's own predicate
            // means a rename upstream cannot quietly turn thinking off here.
            else if !has_thinking
                && opts.thinking.is_enabled()
                && model
                    .id
                    .strip_prefix(DEEPSEEK_VENDOR_PREFIX)
                    .is_some_and(deepseek::uses_v4_thinking_protocol)
            {
                body["chat_template_kwargs"] = json!({"thinking": true});
            }

            self.compat
                .do_stream(model, &[], &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            let url = format!("{}/model/info", self.compat.base_url(&auth));
            let text = self.compat.get_text(&auth, &url).await?;
            let body: Value = serde_json::from_str(&text)?;

            let mut models: Vec<ModelInfo> = body["data"]
                .as_array()
                .map(|arr| arr.iter().filter_map(model_info).collect())
                .unwrap_or_default();
            models.sort_by(|a, b| a.id.cmp(&b.id));
            Ok(models)
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

/// One entry of `/model/info`. `None` for anything that is not a chat model.
fn model_info(entry: &Value) -> Option<ModelInfo> {
    let id = entry["model_name"].as_str()?;
    let info = entry.get("model_info")?;

    let mode_ok = info
        .get("mode")
        .and_then(|v| v.as_str())
        .is_none_or(|m| m == "chat");
    if !mode_ok {
        return None;
    }

    let context_window = info["max_tokens"]
        .as_u64()
        .or_else(|| info["max_input_tokens"].as_u64())
        .and_then(|v| u32::try_from(v).ok());

    // This endpoint enforces `input + max_output <= context_window`, which used
    // to make reporting the real cap fatal: the agent asked for whatever the
    // window had left, so any undercount of the prompt put the sum over. The
    // ask is a flat turn budget now, clamped down to this number, so the cap is
    // safe to report again.
    let max_output_tokens = info["max_output_tokens"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok());

    let input_cost = info["input_cost_per_token"].as_f64();
    let output_cost = info["output_cost_per_token"].as_f64();
    let pricing = if input_cost.is_some() || output_cost.is_some() {
        let per_million = 1_000_000.0;
        Some(ModelPricing::per_million(
            input_cost.unwrap_or(0.0) * per_million,
            output_cost.unwrap_or(0.0) * per_million,
            info["cache_creation_input_token_cost"]
                .as_f64()
                .unwrap_or(0.0)
                * per_million,
            info["cache_read_input_token_cost"].as_f64().unwrap_or(0.0) * per_million,
        ))
    } else {
        None
    };

    let supports_vision = info
        .get("supports_vision")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let supports_thinking = info.get("supports_reasoning").and_then(Value::as_bool);

    let supported_params = info
        .get("supported_openai_params")
        .and_then(Value::as_array)
        .map(|params| TensorXModelInfo {
            has_thinking: params.iter().any(|v| v.as_str() == Some("thinking")),
            has_reasoning_effort: params
                .iter()
                .any(|v| v.as_str() == Some("reasoning_effort")),
        });

    Some(ModelInfo {
        id: id.to_string(),
        context_window,
        max_output_tokens,
        pricing,
        supports_thinking,
        supports_vision: Some(supports_vision),
        tier: None,
        provider_info: supported_params
            .map(|p| Arc::new(p) as Arc<dyn std::any::Any + Send + Sync>),
    })
}

/// The recorded cases, kept out of the test modules so every authoring replays
/// the same list. They are recorded against the bespoke [`TensorX`] impl, and
/// the artifacts stay the spec once it is gone.
///
/// What a turn puts on the wire depends on what `/model/info` said about the
/// model, so every stream fixture past the shared failures is a discovery
/// run: the listing answer first, the stream answer second, from one script.
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures {
    use crate::model::Model;
    use crate::providers::replay::Fixture;
    use crate::test_support::Canned;
    use crate::{Effort, ThinkingConfig};

    /// Advertises `thinking` alone, so the turn carries the bool.
    pub const THINKING_PARAM_SPEC: &str = "tensorx/z-ai/glm-5.2";
    /// Advertises `reasoning_effort` alone, so the turn carries the effort.
    pub const REASONING_EFFORT_SPEC: &str = "tensorx/openai/gpt-oss-120b";
    pub const BOTH_KNOBS_SPEC: &str = "tensorx/qwen/qwen3.5-397b";
    /// Advertises neither knob, so thinking goes through the chat template.
    pub const DEEPSEEK_V4_SPEC: &str = "tensorx/deepseek/deepseek-flash";
    /// The one DeepSeek id outside the V4 protocol, which gets no template
    /// toggle either.
    pub const DEEPSEEK_REASONER_SPEC: &str = "tensorx/deepseek/deepseek-reasoner";
    /// Absent from every listing below.
    pub const UNLISTED_SPEC: &str = "tensorx/moonshotai/kimi-k3";
    /// Absent from every listing too, and still a V4 id, which is all the
    /// template toggle keys off.
    pub const UNLISTED_DEEPSEEK_V4_SPEC: &str = "tensorx/deepseek/deepseek-v9-turbo";
    /// Above what [`crate::dialect::TENSORX`] accepts, so the effort on the
    /// wire proves it snaps rather than passes through.
    const EFFORT: Effort = Effort::Max;

    const UNKNOWN_MODEL: &str = "the model spec did not resolve";

    /// Every kind of model the stream path tells apart, then the non-chat
    /// entries the parser drops, then one entry per edge of each field it
    /// reads: a float, a negative, a null and an out-of-u32 count, string and
    /// wrong-typed values, a price that lands on a rounding tie, and two rows
    /// sharing an id so the sort has to be stable. Entries arrive unsorted.
    const MODELS_BODY: &str = r#"{"data":[
{"model_name":"openai/gpt-oss-120b","model_info":{"mode":"chat","max_input_tokens":131072,"max_output_tokens":32768,"input_cost_per_token":1.5e-7,"output_cost_per_token":6e-7,"supports_reasoning":true,"supported_openai_params":["max_tokens","tools","reasoning_effort"]}},
{"model_name":"z-ai/glm-5.2","model_info":{"mode":"chat","max_tokens":202752,"max_input_tokens":200000,"max_output_tokens":131072,"input_cost_per_token":6e-7,"output_cost_per_token":2.2e-6,"cache_read_input_token_cost":1.1e-7,"supports_vision":false,"supports_reasoning":true,"supported_openai_params":["max_tokens","tools","thinking"]}},
{"model_name":"edge/duplicate","model_info":{"mode":"chat","max_tokens":1000}},
{"model_name":"qwen/qwen3.5-397b","model_info":{"mode":"chat","max_tokens":262144,"supports_vision":true,"supports_reasoning":true,"supported_openai_params":["thinking","reasoning_effort"]}},
{"model_name":"deepseek/deepseek-flash","model_info":{"mode":"chat","max_tokens":1000000,"max_output_tokens":384000,"input_cost_per_token":2.8e-7,"output_cost_per_token":4.2e-7,"cache_creation_input_token_cost":0,"cache_read_input_token_cost":2.8e-8,"supports_reasoning":true,"supported_openai_params":["max_tokens","tools"]}},
{"model_name":"deepseek/deepseek-reasoner","model_info":{"mode":"chat","max_tokens":131072,"supports_reasoning":true,"supported_openai_params":["tools"]}},
{"model_name":"moonshotai/kimi-k3","model_info":{"max_tokens":262144,"supports_vision":true}},
{"model_name":"qwen/qwen3-embedding","model_info":{"mode":"embedding","max_tokens":8192}},
{"model_name":"black-forest-labs/flux","model_info":{"mode":"image_generation"}},
{"model_name":"edge/float-counts","model_info":{"mode":"chat","max_tokens":131072.0,"max_input_tokens":65536,"max_output_tokens":8192.5}},
{"model_name":"edge/negative","model_info":{"mode":"chat","max_tokens":-1,"max_input_tokens":-1,"max_output_tokens":-5,"input_cost_per_token":-1e-6}},
{"model_name":"edge/null","model_info":{"mode":null,"max_tokens":null,"max_input_tokens":32768,"max_output_tokens":null,"input_cost_per_token":null,"output_cost_per_token":1e-6,"supports_vision":null,"supports_reasoning":null,"supported_openai_params":null}},
{"model_name":"edge/beyond-u32","model_info":{"mode":"chat","max_tokens":5000000000,"max_input_tokens":131072,"max_output_tokens":4294967296}},
{"model_name":"edge/wrong-types","model_info":{"mode":1,"input_cost_per_token":"0.000001","output_cost_per_token":"abc","supports_vision":"true","supports_reasoning":"yes","supported_openai_params":"thinking"}},
{"model_name":"edge/mixed-params","model_info":{"mode":"chat","supported_openai_params":["thinking",1,null,{"name":"reasoning_effort"}]}},
{"model_name":"edge/rounding-tie","model_info":{"mode":"chat","input_cost_per_token":1.25e-7,"output_cost_per_token":3.75e-7,"cache_read_input_token_cost":1.5e-8}},
{"model_name":"edge/null-info","model_info":null},
{"model_name":"edge/duplicate","model_info":{"mode":"chat","max_tokens":2000}},
{"model_name":42,"model_info":{"mode":"chat"}},
{"model_info":{"mode":"chat"}},
{"model_name":"edge/no-info"},
null,
"stray"
]}"#;

    /// Only the models the discovery fixtures stream against, each carrying
    /// the one field the stream path reads.
    const DISCOVERY_BODY: &str = r#"{"data":[
{"model_name":"z-ai/glm-5.2","model_info":{"mode":"chat","supported_openai_params":["max_tokens","tools","thinking"]}},
{"model_name":"openai/gpt-oss-120b","model_info":{"mode":"chat","supported_openai_params":["max_tokens","tools","reasoning_effort"]}},
{"model_name":"qwen/qwen3.5-397b","model_info":{"mode":"chat","supported_openai_params":["thinking","reasoning_effort"]}},
{"model_name":"deepseek/deepseek-flash","model_info":{"mode":"chat","supported_openai_params":["max_tokens","tools"]}},
{"model_name":"deepseek/deepseek-reasoner","model_info":{"mode":"chat","supported_openai_params":["tools"]}}
]}"#;

    /// No `data` array at all, which lists nothing rather than failing.
    const NO_DATA_BODY: &str = r#"{"object":"list"}"#;

    const UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"Authentication Error, Invalid proxy server token passed.","type":"auth_error","param":"None","code":"401"}}"#;

    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"reasoning_content":"weighing the options"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}

data: [DONE]

"#;

    const DISCOVERED_SCRIPT: &[Canned] = &[
        Canned::json(200, DISCOVERY_BODY),
        Canned::sse(SUCCESS_TRANSCRIPT),
    ];

    pub const MODELS: Fixture = Fixture {
        name: "models",
        script: &[Canned::json(200, MODELS_BODY)],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    pub const MODELS_WITHOUT_DATA: Fixture = Fixture {
        name: "models_without_data",
        script: &[Canned::json(200, NO_DATA_BODY)],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    pub const MODELS_UNAUTHORIZED: Fixture = Fixture {
        name: "models_unauthorized",
        script: &[Canned::json(401, UNAUTHORIZED_BODY)],
        thinking: ThinkingConfig::Off,
        session: None,
    };

    const fn discovered(name: &'static str, thinking: ThinkingConfig) -> Fixture {
        Fixture {
            name,
            script: DISCOVERED_SCRIPT,
            thinking,
            session: None,
        }
    }

    pub const THINKING_PARAM: Fixture =
        discovered("thinking_param", ThinkingConfig::Effort(EFFORT));
    pub const THINKING_PARAM_OFF: Fixture = discovered("thinking_param_off", ThinkingConfig::Off);
    pub const REASONING_EFFORT: Fixture =
        discovered("reasoning_effort", ThinkingConfig::Effort(EFFORT));
    /// [`crate::dialect::TENSORX`] spells off out loud, as `none`.
    pub const REASONING_EFFORT_OFF: Fixture =
        discovered("reasoning_effort_off", ThinkingConfig::Off);
    pub const BOTH_KNOBS: Fixture = discovered("both_knobs", ThinkingConfig::Effort(EFFORT));
    pub const DEEPSEEK_V4: Fixture = discovered("deepseek_v4", ThinkingConfig::Effort(EFFORT));
    pub const DEEPSEEK_V4_OFF: Fixture = discovered("deepseek_v4_off", ThinkingConfig::Off);
    pub const DEEPSEEK_REASONER: Fixture =
        discovered("deepseek_reasoner", ThinkingConfig::Effort(EFFORT));
    pub const UNDISCOVERED: Fixture = discovered("undiscovered", ThinkingConfig::Effort(EFFORT));
    pub const UNDISCOVERED_DEEPSEEK_V4: Fixture =
        discovered("undiscovered_deepseek_v4", ThinkingConfig::Effort(EFFORT));

    pub fn model(spec: &str) -> Model {
        Model::from_spec(spec).expect(UNKNOWN_MODEL)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use test_case::test_case;

    use super::*;

    const WINDOW: u32 = 1_048_576;
    const INPUT_WINDOW: u32 = 131_072;
    const OUTPUT_CAP: u32 = 262_144;

    fn entry(info: Value) -> Value {
        json!({ "model_name": "kimi-k3", "model_info": info })
    }

    /// The output cap was hard-coded to `None` while reporting it was fatal,
    /// which left the model picker and the thinking math with no number at all.
    #[test_case(
        json!({ "mode": "chat", "max_tokens": WINDOW, "max_input_tokens": INPUT_WINDOW, "max_output_tokens": OUTPUT_CAP }),
        (Some(WINDOW), Some(OUTPUT_CAP))
        ; "max_tokens_wins_over_max_input_tokens"
    )]
    #[test_case(
        json!({ "max_input_tokens": INPUT_WINDOW }),
        (Some(INPUT_WINDOW), None)
        ; "an_unstated_window_falls_back_to_the_input_window"
    )]
    #[test_case(json!({}), (None, None) ; "nothing_stated_leaves_the_choice_to_the_provider")]
    fn the_window_and_the_cap_are_read_off_the_entry(
        info: Value,
        expected: (Option<u32>, Option<u32>),
    ) {
        let model = model_info(&entry(info)).expect("a chat model is listed");
        assert_eq!((model.context_window, model.max_output_tokens), expected);
    }

    #[test]
    fn models_that_do_not_chat_are_skipped() {
        assert!(model_info(&entry(json!({ "mode": "embedding" }))).is_none());
    }
}

/// The bespoke [`TensorX`] on the wire, one recorded exchange at a time.
#[cfg(test)]
mod replay_tests {
    use test_case::test_case;

    use crate::providers::replay::{self, Fixture};

    use super::SLUG;
    use super::fixtures;

    #[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
    #[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
    #[test_case(&replay::RATE_LIMITED ; "rate_limited")]
    #[test_case(&replay::SERVER_ERROR ; "server_error")]
    #[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
    #[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
    #[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
    fn the_bespoke_impl_replays_the_recorded_exchange(fixture: &Fixture) {
        replay::bespoke(SLUG).stream(fixture, &fixtures::model(fixtures::UNLISTED_SPEC));
    }

    #[test_case(&fixtures::MODELS ; "models")]
    #[test_case(&fixtures::MODELS_WITHOUT_DATA ; "models_without_data")]
    #[test_case(&fixtures::MODELS_UNAUTHORIZED ; "models_unauthorized")]
    fn the_bespoke_impl_lists_the_recorded_catalogue(fixture: &Fixture) {
        replay::bespoke(SLUG).models(fixture);
    }

    #[test_case(&fixtures::THINKING_PARAM, fixtures::THINKING_PARAM_SPEC ; "thinking_param")]
    #[test_case(&fixtures::THINKING_PARAM_OFF, fixtures::THINKING_PARAM_SPEC ; "thinking_param_off")]
    #[test_case(&fixtures::REASONING_EFFORT, fixtures::REASONING_EFFORT_SPEC ; "reasoning_effort")]
    #[test_case(&fixtures::REASONING_EFFORT_OFF, fixtures::REASONING_EFFORT_SPEC ; "reasoning_effort_off")]
    #[test_case(&fixtures::BOTH_KNOBS, fixtures::BOTH_KNOBS_SPEC ; "both_knobs")]
    #[test_case(&fixtures::DEEPSEEK_V4, fixtures::DEEPSEEK_V4_SPEC ; "deepseek_v4")]
    #[test_case(&fixtures::DEEPSEEK_V4_OFF, fixtures::DEEPSEEK_V4_SPEC ; "deepseek_v4_off")]
    #[test_case(&fixtures::DEEPSEEK_REASONER, fixtures::DEEPSEEK_REASONER_SPEC ; "deepseek_reasoner")]
    #[test_case(&fixtures::UNDISCOVERED, fixtures::UNLISTED_SPEC ; "undiscovered")]
    #[test_case(&fixtures::UNDISCOVERED_DEEPSEEK_V4, fixtures::UNLISTED_DEEPSEEK_V4_SPEC ; "undiscovered_deepseek_v4")]
    fn the_bespoke_impl_shapes_the_turn_by_what_discovery_found(fixture: &Fixture, spec: &str) {
        replay::bespoke(SLUG).discovered(fixture, &fixtures::model(spec));
    }
}
