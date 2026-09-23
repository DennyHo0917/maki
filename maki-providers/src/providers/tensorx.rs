use std::borrow::Cow;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use maki_config::providers::Protocol;

use crate::model::{ModelFamily, ModelInfo, ModelPricing};
use crate::provider::BoxFuture;
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{
    ApertureRoute, AuthDoc, CatalogDoc, GENERIC_DISCOVERY_NOTE, GeneratedDocs, LoginConfig,
    NO_CURATED_MODELS, ProviderSpec,
};
use crate::types::THINKING_OFF;
use crate::{AgentError, dialect};

use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use super::plugin::{
    self, BodyInput, EffortField, Hook, OpenAiWire, ProviderDecl, ProviderHooks, ThinkingWire,
};
use super::{Timeouts, deepseek};

/// TensorX namespaces resold models by vendor, so DeepSeek ids arrive as
/// `deepseek/deepseek-flash`.
const DEEPSEEK_VENDOR_PREFIX: &str = "deepseek/";
const MODEL_INFO_PATH: &str = "/model/info";
/// The name of the knob both in `supported_openai_params` and on the wire.
const THINKING: &str = "thinking";
/// Also where the declared dialect writes the effort, the codec's default.
const REASONING_EFFORT: &str = "reasoning_effort";
const TEMPLATE_KWARGS: &str = "chat_template_kwargs";
const CHAT_MODE: &str = "chat";
const PER_MILLION: f64 = 1_000_000.0;
const NET_HOST: &str = "api.tensorx.ai";

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
        catalog: CatalogDoc::Discovered(GENERIC_DISCOVERY_NOTE),
        trailing_notes: &[],
    },
};

inventory::submit!(SPEC.config_row());

/// TensorX as a declaration, plus the two things the openai codec cannot
/// spell, see [`hooks`].
///
/// The declared dialect writes `reasoning_effort` on every request, which is
/// right for a model that advertises the knob and wrong for every other one,
/// so the body hook takes it back off where discovery did not vouch for it.
///
/// The bundled `tensorx` Lua plugin says all of this again on the surface a
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
                dialect: &dialect::TENSORX,
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
        list_models: Some(Arc::new(ModelCatalog)),
        build_body: Some(Arc::new(ThinkingKnobs)),
        ..ProviderHooks::default()
    }
}

/// Which of the two thinking knobs a model lists in `supported_openai_params`,
/// carried from the listing to the turn as [`ModelInfo::extra`].
#[derive(Default, Serialize, Deserialize)]
struct AdvertisedKnobs {
    has_thinking: bool,
    has_reasoning_effort: bool,
}

/// `/model/info`, which sits off the codec's `/models` path.
struct ModelCatalog;

impl Hook<(), Vec<ModelInfo>> for ModelCatalog {
    fn call(&self, (): ()) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = plugin::registered_auth(SLUG)?;
            let compat = OpenAiCompatProvider::new(&CONFIG, Timeouts::default());
            let url = format!("{}{MODEL_INFO_PATH}", compat.base_url(&auth));
            let body: Value = serde_json::from_str(&compat.get_text(&auth, &url).await?)?;

            let mut models: Vec<ModelInfo> = body["data"]
                .as_array()
                .map(|arr| arr.iter().filter_map(model_info).collect())
                .unwrap_or_default();
            models.sort_by(|a, b| a.id.cmp(&b.id));
            Ok(models)
        })
    }
}

/// Each knob goes on the wire only for a model that advertised it: `thinking`
/// as a bool, `reasoning_effort` as the dialect rendered it. A DeepSeek model
/// that advertises neither takes the toggle through its chat template instead.
struct ThinkingKnobs;

impl Hook<BodyInput, Value> for ThinkingKnobs {
    fn call(&self, input: BodyInput) -> BoxFuture<'_, Result<Value, AgentError>> {
        Box::pin(async move {
            let BodyInput {
                mut body,
                model,
                thinking,
                model_info,
            } = input;
            let knobs: AdvertisedKnobs = model_info
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            let enabled = thinking != THINKING_OFF;

            if knobs.has_thinking {
                body[THINKING] = json!(enabled);
            }
            if knobs.has_reasoning_effort {
                return Ok(body);
            }
            if let Some(object) = body.as_object_mut() {
                object.remove(REASONING_EFFORT);
            }
            // Sharing DeepSeek's own predicate means a rename upstream cannot
            // quietly turn thinking off here.
            if !knobs.has_thinking
                && enabled
                && model
                    .strip_prefix(DEEPSEEK_VENDOR_PREFIX)
                    .is_some_and(deepseek::uses_v4_thinking_protocol)
            {
                body[TEMPLATE_KWARGS] = json!({ THINKING: true });
            }
            Ok(body)
        })
    }
}

/// One entry of `/model/info`. `None` for anything that is not a chat model.
fn model_info(entry: &Value) -> Option<ModelInfo> {
    let id = entry["model_name"].as_str()?;
    let info = entry.get("model_info")?;

    let mode_ok = info
        .get("mode")
        .and_then(|v| v.as_str())
        .is_none_or(|m| m == CHAT_MODE);
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
        Some(ModelPricing::per_million(
            input_cost.unwrap_or(0.0) * PER_MILLION,
            output_cost.unwrap_or(0.0) * PER_MILLION,
            info["cache_creation_input_token_cost"]
                .as_f64()
                .unwrap_or(0.0)
                * PER_MILLION,
            info["cache_read_input_token_cost"].as_f64().unwrap_or(0.0) * PER_MILLION,
        ))
    } else {
        None
    };

    let supports_vision = info
        .get("supports_vision")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let supports_thinking = info.get("supports_reasoning").and_then(Value::as_bool);

    let knobs = info
        .get("supported_openai_params")
        .and_then(Value::as_array)
        .map(|params| AdvertisedKnobs {
            has_thinking: params.iter().any(|v| v.as_str() == Some(THINKING)),
            has_reasoning_effort: params.iter().any(|v| v.as_str() == Some(REASONING_EFFORT)),
        });

    Some(ModelInfo {
        id: id.to_string(),
        context_window,
        max_output_tokens,
        pricing,
        supports_thinking,
        supports_vision: Some(supports_vision),
        tier: None,
        provider_info: None,
        extra: knobs.map(|knobs| json!(knobs)),
        effort: None,
    })
}

/// The recorded cases, kept out of the test modules so both authorings replay
/// the same list: [`decl`] plus [`hooks`], and the bundled `tensorx` Lua
/// plugin.
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

/// TensorX as [`decl`] plus [`hooks`] put it on the wire, one recorded
/// exchange at a time.
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
    fn the_declaration_replays_the_recorded_exchange(fixture: &Fixture) {
        replay::declared(
            replay::rust_authoring,
            SLUG,
            fixture,
            &fixtures::model(fixtures::UNLISTED_SPEC),
        );
    }

    #[test_case(&fixtures::MODELS ; "models")]
    #[test_case(&fixtures::MODELS_WITHOUT_DATA ; "models_without_data")]
    #[test_case(&fixtures::MODELS_UNAUTHORIZED ; "models_unauthorized")]
    fn the_declaration_lists_the_recorded_catalogue(fixture: &Fixture) {
        replay::declared_models(replay::rust_authoring, SLUG, fixture);
    }

    /// The knobs travel from the listing to the turn through
    /// [`crate::model::ModelInfo::extra`], which no listing golden records, so
    /// these bodies are the only thing that pins it.
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
    fn the_declaration_shapes_the_turn_by_what_discovery_found(fixture: &Fixture, spec: &str) {
        replay::declared_discovered(
            replay::rust_authoring,
            SLUG,
            fixture,
            &fixtures::model(spec),
        );
    }
}
