use std::borrow::Cow;
use std::sync::Arc;

use isahc::http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value, json};

use maki_config::providers::Protocol;

use crate::model::{ModelEffort, ModelFamily, ModelInfo, ModelPricing};
use crate::provider::BoxFuture;
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{
    ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, NO_CURATED_MODELS, ProviderSpec,
};
use crate::{AgentError, dialect};

use super::Timeouts;
use super::openai_compat::{MODELS_PATH, OpenAiCompatConfig, OpenAiCompatProvider};
use super::plugin::{
    self, EffortField, Hook, OpenAiWire, ProviderDecl, ProviderHooks, SessionCarrier, ThinkingWire,
};

const REFERER_HEADER: &str = "http-referer";
const REFERER: &str = "https://maki.sh";
const TITLE_HEADER: &str = "x-openrouter-title";
const APP_TITLE: &str = "maki";
const EFFORT_FIELD: &str = "reasoning.effort";
const INVALID_EFFORT_FIELD: &str = "openrouter's effort field is a valid dotted path";
/// Marks the whole prompt as cacheable, for the upstreams that only cache
/// when asked to.
const CACHE_FIELD: &str = "cache_control";
const SESSION_FIELD: &str = "session_id";
const PER_MILLION: f64 = 1_000_000.0;
const TEXT_MODALITY: &str = "text";
const IMAGE_MODALITY: &str = "image";
const REASONING_PARAMETER: &str = "reasoning";
const NET_HOST: &str = "openrouter.ai";

const SLUG: &str = "openrouter";
const DISPLAY_NAME: &str = "OpenRouter";
const ENV_VAR: &str = "OPENROUTER_API_KEY";
const BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_MODEL: &str = "openrouter/openai/gpt-5.5";
const LOGIN_URL: &str = "https://openrouter.ai/keys";
const MAX_TOKENS_FIELD: &str = "max_tokens";
const FEATURES: &str = "300+ models from all providers, prompt caching, provider routing";

const DISCOVERY_NOTE: &str = "OpenRouter aggregates models from many providers behind a single API key. \
     Browse available models at [openrouter.ai/models](https://openrouter.ai/models). \
     Use any model ID directly (e.g. `openrouter/anthropic/claude-sonnet-4`).";

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
    fallback_max_output: Some(128_000),
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
        catalog: CatalogDoc::Discovered(DISCOVERY_NOTE),
        trailing_notes: &[],
    },
};

inventory::submit!(SPEC.config_row());

/// OpenRouter as a declaration, plus the listing the openai codec cannot
/// spell, see [`hooks`].
///
/// Everything static about its wire is data here: the attribution headers,
/// the cache marker in every body, the session id in the body, and effort
/// under `reasoning.effort` in the `prefer-high` dialect, sent only to a model
/// that reasons. That dialect is only the fallback for a model the listing
/// never described: a listed one narrows it through [`ModelInfo::effort`].
/// Claiming a built-in slug inherits the whole [`SPEC`] row, and `max_tokens`
/// and streamed usage are already the codec's defaults.
///
/// The bundled `openrouter` Lua plugin says all of this again on the surface
/// a third-party plugin uses, and outranks this at every real startup.
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
                dialect: &dialect::PREFER_HIGH,
                field: EffortField::parse(EFFORT_FIELD).expect(INVALID_EFFORT_FIELD),
                requires_support: true,
            }),
            headers: HeaderMap::from_iter([
                (
                    HeaderName::from_static(REFERER_HEADER),
                    HeaderValue::from_static(REFERER),
                ),
                (
                    HeaderName::from_static(TITLE_HEADER),
                    HeaderValue::from_static(APP_TITLE),
                ),
            ]),
            extra_body: Some(Map::from_iter([(
                CACHE_FIELD.to_owned(),
                json!({ "type": "ephemeral" }),
            )])),
            session_id: Some(SessionCarrier::BodyField(SESSION_FIELD.to_owned())),
            ..OpenAiWire::default()
        }),
        net_hosts: vec![NET_HOST.to_owned()],
    }
}

/// The one callback [`decl`] cannot spell, registered alongside it.
pub(crate) fn hooks() -> ProviderHooks {
    ProviderHooks {
        list_models: Some(Arc::new(Catalog)),
        ..ProviderHooks::default()
    }
}

/// `/models`, read with OpenRouter's own field names.
struct Catalog;

impl Hook<(), Vec<ModelInfo>> for Catalog {
    fn call(&self, (): ()) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = plugin::registered_auth(SLUG)?;
            OpenAiCompatProvider::new(&CONFIG, Timeouts::default())
                .fetch_and_parse_models(&auth, MODELS_PATH, parse_model)
                .await
        })
    }
}

fn lists(modalities: &[Value], wanted: &str) -> bool {
    modalities.iter().any(|m| m.as_str() == Some(wanted))
}

/// Only text-in, text-out models are listed. Prices arrive per token as
/// strings and are scaled to $/M. One we cannot read leaves the pricing
/// unknown rather than free.
///
/// A `reasoning` block comes in three states, which become the model's
/// [`ModelEffort`]: mandatory (always on, Off sends nothing), default enabled
/// (Off sends `none`) and default off (Off sends nothing, any effort turns it
/// on). Its effort names go through as listed, for [`ModelEffort`] to vet.
fn parse_model(m: &Value) -> Option<ModelInfo> {
    let architecture = m["architecture"].as_object()?;
    let input_modalities = architecture.get("input_modalities")?.as_array()?;
    let output_modalities = architecture.get("output_modalities")?.as_array()?;
    if !lists(input_modalities, TEXT_MODALITY) || !lists(output_modalities, TEXT_MODALITY) {
        return None;
    }

    let id = m["id"].as_str()?;
    let context_window = m["context_length"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok());
    let per_token =
        |p: &Value| -> Option<f64> { Some(p.as_str()?.parse::<f64>().ok()? * PER_MILLION) };
    let pricing = m["pricing"].as_object().and_then(|p| {
        Some(ModelPricing::per_million(
            per_token(p.get("prompt")?)?,
            per_token(p.get("completion")?)?,
            p.get("input_cache_write")
                .and_then(per_token)
                .unwrap_or(0.0),
            p.get("input_cache_read").and_then(per_token).unwrap_or(0.0),
        ))
    });

    let effort = m["reasoning"].as_object().map(|reasoning| {
        let flag = |name: &str| reasoning.get(name).and_then(Value::as_bool) == Some(true);
        ModelEffort {
            supported: reasoning
                .get("supported_efforts")
                .and_then(Value::as_array)
                .map(|listed| ModelEffort::known_levels(listed))
                .unwrap_or_default(),
            send_off: Some(flag("default_enabled") && !flag("mandatory")),
        }
    });
    let supports_thinking = effort.is_some()
        || m["supported_parameters"]
            .as_array()
            .is_some_and(|params| lists(params, REASONING_PARAMETER));

    Some(ModelInfo {
        id: id.to_string(),
        context_window,
        pricing,
        supports_thinking: Some(supports_thinking),
        supports_vision: Some(lists(input_modalities, IMAGE_MODALITY)),
        effort,
        ..ModelInfo::default()
    })
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::*;
    use crate::Effort;

    const KIMI_ID: &str = "moonshotai/kimi-k3";
    const UNKNOWN_PRICE_STAYS_UNKNOWN: &str = "a price we cannot read must not become a zero price";

    fn kimi_k3_json() -> Value {
        json!({
            "id": KIMI_ID,
            "context_length": 1_048_576,
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "input_cache_read": "0.0000003",
            },
            "supported_parameters": ["reasoning"],
        })
    }

    #[test]
    fn parse_model_scales_pricing_to_per_million() {
        let info = parse_model(&kimi_k3_json()).expect("model should parse");

        assert_eq!(info.id, KIMI_ID);
        assert_eq!(info.context_window, Some(1_048_576));
        assert_eq!(info.supports_vision, Some(true));
        assert_eq!(info.supports_thinking, Some(true));
        assert_eq!(info.effort, None);
        let pricing = info.pricing.expect("pricing should be parsed");
        assert_eq!(pricing.input, 3.0);
        assert_eq!(pricing.output, 15.0);
        assert_eq!(pricing.cache_read, 0.3);
        assert_eq!(pricing.cache_write, 0.0);
    }

    #[test]
    fn parse_model_scales_cache_write() {
        let mut m = kimi_k3_json();
        m["pricing"]["input_cache_write"] = json!("0.00000375");

        let pricing = parse_model(&m)
            .expect("model should parse")
            .pricing
            .expect("pricing should be parsed");
        assert_eq!(pricing.cache_write, 3.75);
    }

    /// A price we cannot read used to collapse to an all-zero `ModelPricing`,
    /// which downstream reads as "free". Unknown has to stay unknown.
    #[test_case(json!(null)                                       ; "no_pricing_object")]
    #[test_case(json!({"prompt": "0.000003"})                     ; "no_completion")]
    #[test_case(json!({"prompt": "n/a", "completion": "0.000015"}) ; "unparsable_prompt")]
    fn parse_model_keeps_unusable_pricing_unknown(pricing: Value) {
        let mut m = kimi_k3_json();
        m["pricing"] = pricing;

        let info = parse_model(&m).expect("model should parse");
        assert!(info.pricing.is_none(), "{UNKNOWN_PRICE_STAYS_UNKNOWN}");
    }

    #[test_case(false, false, Some(false) ; "default_off_sends_nothing")]
    #[test_case(true,  false, Some(true)  ; "default_enabled_disables_with_none")]
    #[test_case(true,  true,  Some(false) ; "mandatory_cannot_be_disabled")]
    fn parse_model_reads_the_reasoning_block_into_effort(
        default_enabled: bool,
        mandatory: bool,
        send_off: Option<bool>,
    ) {
        let mut m = kimi_k3_json();
        m["reasoning"] = json!({
            "mandatory": mandatory,
            "default_enabled": default_enabled,
            "supported_efforts": ["high", "bogus", "low", "none"],
        });

        let info = parse_model(&m).expect("model should parse");
        assert_eq!(
            info.effort,
            Some(ModelEffort {
                supported: vec![Effort::Low, Effort::High],
                send_off,
            })
        );
    }

    #[test_case(json!(["image"]), json!(["image"]); "image_only")]
    #[test_case(json!(["image"]), json!(["text"]); "image_input_only")]
    #[test_case(json!(["text"]), json!(["image"]); "image_output_only")]
    fn parse_model_skips_non_text_models(input: Value, output: Value) {
        let mut m = kimi_k3_json();
        m["architecture"]["input_modalities"] = input;
        m["architecture"]["output_modalities"] = output;

        assert!(parse_model(&m).is_none());
    }
}

/// The recorded cases, kept out of the test module so both authorings replay
/// the same list: [`decl`] plus [`hooks`], and the bundled `openrouter` Lua
/// plugin.
///
/// [`ModelInfo::effort`] never reaches a golden, so the `models` listing alone
/// shows only what the `reasoning` block did to `supports_thinking`. What it
/// did to the effort dialect is pinned by the discovered turns, each of which
/// lists the one catalog below and then asks one of its models for a thinking
/// mode.
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures {
    use crate::model::Model;
    use crate::providers::replay::{self, Fixture};
    use crate::test_support::Canned;
    use crate::{Effort, ThinkingConfig};

    /// Lists `xhigh` down to `minimal` plus two names maki has no level for.
    pub const XHIGH_SPEC: &str = "openrouter/openai/gpt-5.5";
    /// Lists nothing above `medium`, so a high ask snaps below the static
    /// dialect's ceiling.
    pub const MEDIUM_SPEC: &str = "openrouter/google/gemini-3-flash";
    /// Reasons by default and may be switched off.
    pub const DEFAULT_ENABLED_SPEC: &str = "openrouter/deepseek/deepseek-v4";
    /// Reasons by default and may not be switched off.
    pub const MANDATORY_SPEC: &str = "openrouter/anthropic/claude-opus-5";
    /// Has a `reasoning` block whose every effort name is one maki drops.
    pub const UNKNOWN_EFFORTS_SPEC: &str = "openrouter/moonshotai/kimi-k3";
    /// Has no `reasoning` block, only `reasoning` in `supported_parameters`.
    pub const PARAMETER_ONLY_SPEC: &str = "openrouter/qwen/qwen3-coder";
    /// Listed with no reasoning signal at all.
    pub const NO_REASONING_SPEC: &str = "openrouter/mistralai/codestral";
    /// Absent from [`CATALOG`].
    pub const UNLISTED_SPEC: &str = "openrouter/meta-llama/llama-5";

    /// 62% of the 64k thinking budget the fallback 128k output window allows,
    /// which converts to `xhigh`: past the static dialect's `high` ceiling, so
    /// only a discovered effort list lets it through.
    const BUDGET_TOKENS: u32 = 40_000;
    const SESSION: &str = "01965087-4c71-7f00-8000-000000000001";

    const UNKNOWN_MODEL: &str = "the model spec did not resolve";

    /// One `/models` answer for every discovered turn and for the listing
    /// itself. Besides the reasoning variants it carries each number the
    /// parser has to refuse or scale. `context_length` comes as a float, a
    /// negative, a null and past u32. Prices come as a bare number, an
    /// unparsable string, a negative and a sub-cent `0.125`. Some rows are for the modality filter
    /// to drop, and two share an id, which a stable sort keeps in arrival
    /// order.
    const CATALOG: &str = r#"{"data":[
{"id":"z-ai/glm-5","context_length":128000,"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"pricing":{"prompt":"0.0000006","completion":"0.0000022"},"reasoning":{"supported_efforts":"high"}},
{"id":"openai/gpt-5.5","context_length":400000,"architecture":{"input_modalities":["text","image","file"],"output_modalities":["text"]},"pricing":{"prompt":"0.00000125","completion":"0.00001","input_cache_read":"0.000000125"},"supported_parameters":["reasoning","tools"],"reasoning":{"supported_efforts":["xhigh","high","medium","low","minimal","none","bogus"],"default_enabled":false,"mandatory":false}},
{"id":"black-forest-labs/flux-2","architecture":{"input_modalities":["text"],"output_modalities":["image"]},"pricing":{"prompt":"0","completion":"0"}},
{"id":"deepseek/deepseek-v4","context_length":-1,"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"pricing":{"prompt":"n/a","completion":"0.0000011"},"reasoning":{"default_enabled":true,"mandatory":false}},
{"id":"anthropic/claude-opus-5","context_length":1000000.0,"architecture":{"input_modalities":["image","text"],"output_modalities":["text"]},"pricing":{"prompt":"0.000005","completion":"0.000025","input_cache_read":"0.0000005","input_cache_write":"0.00000625"},"reasoning":{"supported_efforts":[],"default_enabled":true,"mandatory":true}},
{"id":"openai/whisper-2","architecture":{"input_modalities":["audio"],"output_modalities":["text"]}},
{"id":"moonshotai/kimi-k3","context_length":null,"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"pricing":{"prompt":0.000003,"completion":"0.000015"},"reasoning":{"supported_efforts":["none","bogus"],"default_enabled":"true","mandatory":null}},
{"id":"qwen/qwen3-coder","context_length":5000000000,"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"pricing":{"prompt":"-1","completion":"-1","input_cache_read":"garbage"},"supported_parameters":["tools","reasoning"]},
{"id":"google/gemini-3-flash","context_length":1048576,"architecture":{"input_modalities":["text","image"],"output_modalities":["text","image"]},"pricing":{"prompt":"0.0000005","completion":"0.000003"},"reasoning":{"supported_efforts":[1,null,"LOW","medium","minimal","low"]}},
{"id":"mistralai/codestral","context_length":256000,"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"pricing":{"prompt":"0.0000003","completion":"0.0000009"},"supported_parameters":["tools"],"reasoning":null},
{"id":"x-ai/grok-5","context_length":2000000,"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"reasoning":true},
{"architecture":{"input_modalities":["text"],"output_modalities":["text"]}},
{"id":"openrouter/auto","architecture":{"input_modalities":["text"]}},
{"id":"meta-llama/llama-guard-5","context_length":131072},
{"id":"z-ai/glm-5","context_length":200000,"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"pricing":{"prompt":"0.0000006","completion":"0.0000022"}}
]}"#;

    const SUCCESS_TRANSCRIPT: &str = r#": OPENROUTER PROCESSING

data: {"id":"gen-1","choices":[{"delta":{"reasoning":"weighing the options"}}]}

data: {"id":"gen-1","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"gen-1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"id":"gen-1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}

data: {"id":"gen-1","choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4},"cost":0.00042}}

data: [DONE]

"#;

    const DISCOVERY_SCRIPT: &[Canned] =
        &[Canned::json(200, CATALOG), Canned::sse(SUCCESS_TRANSCRIPT)];

    const fn discovered(name: &'static str, thinking: ThinkingConfig) -> Fixture {
        Fixture {
            name,
            script: DISCOVERY_SCRIPT,
            thinking,
            session: None,
        }
    }

    pub const MODELS: Fixture = Fixture {
        name: "models",
        script: &[Canned::json(200, CATALOG)],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    pub const MODELS_UNAUTHORIZED: Fixture = Fixture {
        name: "models_unauthorized",
        script: replay::UNAUTHORIZED.script,
        thinking: ThinkingConfig::Off,
        session: None,
    };

    /// Replayed on [`XHIGH_SPEC`].
    pub const MAX_SNAPS_TO_XHIGH: Fixture =
        discovered("max_snaps_to_xhigh", ThinkingConfig::Effort(Effort::Max));
    /// Replayed on [`MEDIUM_SPEC`].
    pub const MAX_SNAPS_TO_MEDIUM: Fixture =
        discovered("max_snaps_to_medium", ThinkingConfig::Effort(Effort::Max));
    /// Replayed on [`UNKNOWN_EFFORTS_SPEC`]: an emptied list inherits the
    /// declared levels.
    pub const MAX_WITH_UNKNOWN_EFFORTS_ONLY: Fixture = discovered(
        "max_with_unknown_efforts_only",
        ThinkingConfig::Effort(Effort::Max),
    );
    /// Replayed on [`PARAMETER_ONLY_SPEC`].
    pub const MAX_WITH_PARAMETER_ONLY_REASONING: Fixture = discovered(
        "max_with_parameter_only_reasoning",
        ThinkingConfig::Effort(Effort::Max),
    );
    /// Replayed on [`UNLISTED_SPEC`]: plain `prefer-high`.
    pub const UNDISCOVERED: Fixture =
        discovered("undiscovered", ThinkingConfig::Effort(Effort::Max));
    /// Replayed on [`DEFAULT_ENABLED_SPEC`], which is told `none`.
    pub const OFF_DEFAULT_ENABLED: Fixture = discovered("off_default_enabled", ThinkingConfig::Off);
    /// Replayed on [`MANDATORY_SPEC`], which is told nothing.
    pub const OFF_MANDATORY: Fixture = discovered("off_mandatory", ThinkingConfig::Off);
    /// Replayed on [`XHIGH_SPEC`].
    pub const BUDGET: Fixture = discovered("budget", ThinkingConfig::Budget(BUDGET_TOKENS));
    /// Replayed on [`NO_REASONING_SPEC`]: no support, so no effort at all.
    pub const NOT_A_REASONING_MODEL: Fixture = discovered(
        "not_a_reasoning_model",
        ThinkingConfig::Effort(Effort::High),
    );
    /// Replayed on [`XHIGH_SPEC`]. The session rides in the body.
    pub const IN_SESSION: Fixture = Fixture {
        session: Some(SESSION),
        ..discovered("in_session", ThinkingConfig::Effort(Effort::High))
    };

    pub fn model(spec: &str) -> Model {
        Model::from_spec(spec).expect(UNKNOWN_MODEL)
    }
}

/// OpenRouter as [`decl`] plus [`hooks`] put it on the wire, one recorded
/// exchange at a time.
#[cfg(test)]
mod replay_tests {
    use test_case::test_case;

    use crate::providers::replay::{self, Fixture};

    use super::SLUG;
    use super::fixtures::{
        BUDGET, DEFAULT_ENABLED_SPEC, IN_SESSION, MANDATORY_SPEC, MAX_SNAPS_TO_MEDIUM,
        MAX_SNAPS_TO_XHIGH, MAX_WITH_PARAMETER_ONLY_REASONING, MAX_WITH_UNKNOWN_EFFORTS_ONLY,
        MEDIUM_SPEC, MODELS, MODELS_UNAUTHORIZED, NO_REASONING_SPEC, NOT_A_REASONING_MODEL,
        OFF_DEFAULT_ENABLED, OFF_MANDATORY, PARAMETER_ONLY_SPEC, UNDISCOVERED,
        UNKNOWN_EFFORTS_SPEC, UNLISTED_SPEC, XHIGH_SPEC, model,
    };

    #[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
    #[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
    #[test_case(&replay::RATE_LIMITED ; "rate_limited")]
    #[test_case(&replay::SERVER_ERROR ; "server_error")]
    #[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
    #[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
    #[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
    fn the_declaration_replays_the_recorded_exchange(fixture: &Fixture) {
        replay::declared(replay::rust_authoring, SLUG, fixture, &model(UNLISTED_SPEC));
    }

    #[test_case(&MODELS ; "models")]
    #[test_case(&MODELS_UNAUTHORIZED ; "models_unauthorized")]
    fn the_declaration_lists_the_recorded_catalog(fixture: &Fixture) {
        replay::declared_models(replay::rust_authoring, SLUG, fixture);
    }

    #[test_case(&MAX_SNAPS_TO_XHIGH, XHIGH_SPEC ; "max_snaps_to_xhigh")]
    #[test_case(&MAX_SNAPS_TO_MEDIUM, MEDIUM_SPEC ; "max_snaps_to_medium")]
    #[test_case(&MAX_WITH_UNKNOWN_EFFORTS_ONLY, UNKNOWN_EFFORTS_SPEC ; "max_with_unknown_efforts_only")]
    #[test_case(&MAX_WITH_PARAMETER_ONLY_REASONING, PARAMETER_ONLY_SPEC ; "max_with_parameter_only_reasoning")]
    #[test_case(&UNDISCOVERED, UNLISTED_SPEC ; "undiscovered")]
    #[test_case(&OFF_DEFAULT_ENABLED, DEFAULT_ENABLED_SPEC ; "off_default_enabled")]
    #[test_case(&OFF_MANDATORY, MANDATORY_SPEC ; "off_mandatory")]
    #[test_case(&BUDGET, XHIGH_SPEC ; "budget")]
    #[test_case(&NOT_A_REASONING_MODEL, NO_REASONING_SPEC ; "not_a_reasoning_model")]
    #[test_case(&IN_SESSION, XHIGH_SPEC ; "in_session")]
    fn the_declaration_replays_the_discovered_turn(fixture: &Fixture, spec: &str) {
        replay::declared_discovered(replay::rust_authoring, SLUG, fixture, &model(spec));
    }
}
