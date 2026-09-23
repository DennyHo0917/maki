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
    ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, NO_CURATED_MODELS, Native,
    ProviderSpec,
};
use crate::{
    AgentError, Effort, EffortDialect, Message, ProviderEvent, RequestOptions, StreamResponse,
    dialect,
};

use super::openai_compat::{MODELS_PATH, OpenAiCompatConfig, OpenAiCompatProvider};
use super::{KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts};

const REFERER: &str = "https://maki.sh";
const APP_TITLE: &str = "maki";
const PER_MILLION: f64 = 1_000_000.0;

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
        catalog: CatalogDoc::Discovered(DISCOVERY_NOTE),
        trailing_notes: &[],
    },
};

fn create(timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    Ok(Box::new(OpenRouter::new(timeouts)?))
}

fn create_with_auth(
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
    system_prefix: Option<String>,
) -> Box<dyn Provider> {
    Box::new(OpenRouter::with_auth(auth, timeouts).with_system_prefix(system_prefix))
}

inventory::submit!(SPEC.config_row());

#[derive(Debug)]
struct OpenRouterModelInfo {
    reasoning_mandatory: bool,
    reasoning_default_enabled: bool,
    reasoning_efforts: Vec<Effort>,
}

pub struct OpenRouter {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    key_pool: Option<KeyPool>,
    system_prefix: Option<String>,
}

impl OpenRouter {
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

/// OpenRouter models come in three reasoning states, encoded here as a
/// dialect so `effort_str` can resolve them like any other provider:
/// 1. mandatory - always on; Off sends nothing (can't disable).
/// 2. default_enabled - on by default; Off sends effort "none".
/// 3. default off - Off sends nothing; any effort string turns it on.
fn effort_dialect(info: Option<&OpenRouterModelInfo>) -> EffortDialect<'_> {
    let Some(info) = info else {
        return dialect::PREFER_HIGH;
    };
    EffortDialect {
        supported: match info.reasoning_efforts.as_slice() {
            [] => dialect::PREFER_HIGH.supported,
            declared => declared,
        },
        off: (info.reasoning_default_enabled && !info.reasoning_mandatory).then_some(dialect::OFF),
        ..dialect::PREFER_HIGH
    }
}

fn parse_model(m: &Value) -> Option<ModelInfo> {
    // Filter: only text input/output models
    let architecture = m["architecture"].as_object()?;
    let input_modalities = architecture.get("input_modalities")?.as_array()?;
    let output_modalities = architecture.get("output_modalities")?.as_array()?;

    let has_text_input = input_modalities.iter().any(|m| m.as_str() == Some("text"));
    let has_text_output = output_modalities.iter().any(|m| m.as_str() == Some("text"));
    if !has_text_input || !has_text_output {
        return None;
    }

    let supports_vision = input_modalities.iter().any(|m| m.as_str() == Some("image"));

    // Parse with OpenRouter-specific pricing field names. OpenRouter reports
    // per-token prices; scale to $/M as `ModelPricing` expects. A missing or
    // unparsable price stays `None` so it never reads as free.
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

    let reasoning = m
        .get("reasoning")
        .and_then(|v| v.as_object())
        .map(|v| OpenRouterModelInfo {
            reasoning_mandatory: v.get("mandatory").and_then(Value::as_bool) == Some(true),
            reasoning_default_enabled: v.get("default_enabled").and_then(Value::as_bool)
                == Some(true),
            reasoning_efforts: v
                .get("supported_efforts")
                .and_then(Value::as_array)
                .map(|arr| {
                    let mut efforts: Vec<Effort> = arr
                        .iter()
                        .filter_map(|v| v.as_str()?.parse().ok())
                        .collect();
                    efforts.sort_unstable();
                    efforts
                })
                .unwrap_or_default(),
        });

    let supports_thinking = reasoning.is_some()
        || m.get("supported_parameters")
            .and_then(|v| v.as_array())
            .is_some_and(|v| v.iter().any(|v| v.as_str() == Some("reasoning")));

    Some(ModelInfo {
        id: id.to_string(),
        context_window,
        max_output_tokens: None,
        pricing,
        supports_thinking: Some(supports_thinking),
        supports_vision: Some(supports_vision),
        tier: None,
        provider_info: reasoning.map(|r| Arc::new(r) as Arc<dyn std::any::Any + Send + Sync>),
    })
}

impl Provider for OpenRouter {
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

            body["cache_control"] = json!({"type": "ephemeral"});

            let reasoning_info = crate::model_registry::provider_info::<OpenRouterModelInfo>(
                &CONFIG.slug,
                &model.id,
            );

            let effort_dialect = effort_dialect(reasoning_info.as_deref());
            if model.supports_thinking()
                && let Some(effort) = opts.thinking.effort_str(&effort_dialect, model)
            {
                body["reasoning"] = json!({"effort": effort});
            }

            if let Some(sid) = session_id {
                body["session_id"] = json!(sid.to_string());
            }

            let extra_headers = [("HTTP-Referer", REFERER), ("X-OpenRouter-Title", APP_TITLE)];
            self.compat
                .do_stream(model, &extra_headers, &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            self.compat
                .fetch_and_parse_models(&auth, MODELS_PATH, parse_model)
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
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::*;
    use crate::ThinkingConfig;

    const UNKNOWN_PRICE_STAYS_UNKNOWN: &str = "a price we cannot read must not become a zero price";

    fn kimi_k3_json() -> Value {
        json!({
            "id": "moonshotai/kimi-k3",
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

        assert_eq!(info.id, "moonshotai/kimi-k3");
        assert_eq!(info.context_window, Some(1_048_576));
        assert_eq!(info.supports_vision, Some(true));
        assert_eq!(info.supports_thinking, Some(true));
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

    #[test]
    fn parse_model_reasoning_efforts_skips_unknown_and_sorts() {
        let mut m = kimi_k3_json();
        m["reasoning"] = json!({
            "mandatory": false,
            "default_enabled": true,
            "supported_efforts": ["high", "bogus", "low", "none"],
        });

        let info = parse_model(&m).expect("model should parse");
        let provider_info = info.provider_info.expect("reasoning info should be set");
        let reasoning = provider_info
            .downcast_ref::<OpenRouterModelInfo>()
            .expect("wrong provider info type");
        assert!(reasoning.reasoning_default_enabled);
        assert!(!reasoning.reasoning_mandatory);
        assert_eq!(reasoning.reasoning_efforts, vec![Effort::Low, Effort::High]);
    }

    fn openrouter_model(info: Option<&OpenRouterModelInfo>) -> (EffortDialect<'_>, Model) {
        let model = Model {
            id: "test-model".into(),
            provider: "openrouter".into(),
            tier: crate::model::ModelTier::Medium,
            family: crate::model::ModelFamily::Generic,
            supports_tool_examples_override: None,
            thinking_override: None,
            supports_vision_override: None,
            supports_fast_override: None,
            pricing: ModelPricing::default(),
            subsidised_by: None,
            discovered_free: false,
            max_output_tokens: Some(8192),
            turn_output_tokens: None,
            context_window: 200_000,
            thinking_fields: None,
        };
        (effort_dialect(info), model)
    }

    fn reasoning_info(efforts: &[Effort]) -> OpenRouterModelInfo {
        OpenRouterModelInfo {
            reasoning_mandatory: false,
            reasoning_default_enabled: false,
            reasoning_efforts: efforts.to_vec(),
        }
    }

    #[test_case(&[Effort::High, Effort::XHigh], ThinkingConfig::Effort(Effort::XHigh), "xhigh" ; "declared_xhigh_passes_through")]
    #[test_case(&[Effort::High, Effort::XHigh], ThinkingConfig::Effort(Effort::Max),   "xhigh" ; "max_snaps_to_declared_xhigh")]
    #[test_case(&[Effort::Minimal, Effort::Low], ThinkingConfig::Adaptive,             "low"   ; "adaptive_snaps_into_declared")]
    #[test_case(&[], ThinkingConfig::Effort(Effort::XHigh), "high" ; "no_declared_falls_back_to_static")]
    fn effort_dialect_snaps_once_against_declared_levels(
        efforts: &[Effort],
        config: ThinkingConfig,
        expected: &str,
    ) {
        let info = reasoning_info(efforts);
        let (dialect, model) = openrouter_model(Some(&info));
        assert_eq!(config.effort_str(&dialect, &model), Some(expected));
    }

    #[test]
    fn no_reasoning_info_still_requests_high_effort() {
        let (dialect, model) = openrouter_model(None);
        assert_eq!(
            ThinkingConfig::Adaptive.effort_str(&dialect, &model),
            Some("high")
        );
    }

    #[test_case(false, false, None         ; "default_off_sends_nothing")]
    #[test_case(true,  false, Some("none") ; "default_enabled_disables_with_none")]
    #[test_case(true,  true,  None         ; "mandatory_cannot_be_disabled")]
    fn off_resolves_per_reasoning_flags(
        default_enabled: bool,
        mandatory: bool,
        expected: Option<&str>,
    ) {
        let info = OpenRouterModelInfo {
            reasoning_mandatory: mandatory,
            reasoning_default_enabled: default_enabled,
            reasoning_efforts: vec![],
        };
        let (dialect, model) = openrouter_model(Some(&info));
        assert_eq!(ThinkingConfig::Off.effort_str(&dialect, &model), expected);
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

/// The recorded cases, kept out of the test module so every authoring replays
/// the same list: the bespoke [`OpenRouter`] records them, and the declaration
/// that replaces it, Rust or bundled Lua, must replay them byte for byte.
///
/// `provider_info` never reaches a golden, so the `models` listing alone shows
/// only what the `reasoning` block did to `supports_thinking`. What it did to
/// the effort dialect is pinned by the discovered turns, each of which lists
/// the one catalog below and then asks one of its models for a thinking mode.
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

/// The bespoke [`OpenRouter`] recording the exchanges its replacement has to
/// reproduce.
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
    fn the_bespoke_impl_records_the_exchange(fixture: &Fixture) {
        replay::bespoke(SLUG).stream(fixture, &model(UNLISTED_SPEC));
    }

    #[test_case(&MODELS ; "models")]
    #[test_case(&MODELS_UNAUTHORIZED ; "models_unauthorized")]
    fn the_bespoke_impl_records_the_listing(fixture: &Fixture) {
        replay::bespoke(SLUG).models(fixture);
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
    fn the_bespoke_impl_records_the_discovered_turn(fixture: &Fixture, spec: &str) {
        replay::bespoke(SLUG).discovered(fixture, &model(spec));
    }
}
