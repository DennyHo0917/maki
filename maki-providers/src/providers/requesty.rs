use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use flume::Sender;
use maki_storage::id::SessionRef;
use serde_json::{Value, json};
use tracing::warn;

use maki_config::providers::Protocol;

use crate::model::{Model, ModelFamily, ModelInfo, ModelPricing};
use crate::provider::{BoxFuture, Provider};
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{
    ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, NO_CURATED_MODELS, Native,
    ProviderSpec,
};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse, dialect};

use super::openai_compat::{MODELS_PATH, OpenAiCompatConfig, OpenAiCompatProvider};
use super::{KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts};

const REFERER: &str = "https://maki.sh";
const APP_TITLE: &str = "maki";
const PER_MILLION: f64 = 1_000_000.0;
/// Requesty's own curated routing policies, with short stable ids like
/// `claude-sonnet-4-5` that spread across several upstream providers. Listed
/// before the raw `<vendor>/<model>` catalog at [`MODELS_PATH`].
const MANAGED_MODELS_PATH: &str = "/models/managed";
const CHAT_API: &str = "chat";

const SLUG: &str = "requesty";
const DISPLAY_NAME: &str = "Requesty";
const ENV_VAR: &str = "REQUESTY_API_KEY";
const BASE_URL: &str = "https://router.requesty.ai/v1";
const DEFAULT_MODEL: &str = "requesty/openai/gpt-5.5";
const LOGIN_URL: &str = "https://app.requesty.ai/api-keys";
const MAX_TOKENS_FIELD: &str = "max_tokens";
const FEATURES: &str = "700+ models behind one key, curated managed routing policies, EU region via `REQUESTY_BASE_URL`";

const DISCOVERY_NOTE: &str = "Requesty routes 700+ models from many providers behind a single API key. \
     Models are listed live from the API: curated managed policies first \
     (short ids such as `requesty/claude-sonnet-4-5` or `requesty/gpt-5.4-mini`, \
     `@eu` variants route only through EU providers), then the full \
     `<vendor>/<model>` catalog (e.g. `requesty/openai/gpt-4o-mini`). \
     Get a key at [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys). \
     Set `REQUESTY_BASE_URL=https://router.eu.requesty.ai/v1` to keep all \
     traffic in the EU.";

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
    Ok(Box::new(Requesty::new(timeouts)?))
}

fn create_with_auth(
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
    system_prefix: Option<String>,
) -> Box<dyn Provider> {
    Box::new(Requesty::with_auth(auth, timeouts).with_system_prefix(system_prefix))
}

inventory::submit!(SPEC.config_row());

pub struct Requesty {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    key_pool: Option<KeyPool>,
    system_prefix: Option<String>,
}

impl Requesty {
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

/// A catalog entry, read one field at a time. A missing or `null` field is the
/// normal "Requesty does not say" and stays quiet, while a field that is there
/// in a shape we cannot read is upstream drift: it gets a log line and costs us
/// that one value instead of the whole model.
struct Entry<'a> {
    id: &'a str,
    raw: &'a Value,
}

impl Entry<'_> {
    fn read<T>(&self, name: &str, parse: impl Fn(&Value) -> Option<T>) -> Option<T> {
        let value = self.raw.get(name).filter(|v| !v.is_null())?;
        let parsed = parse(value);
        if parsed.is_none() {
            warn!(model = self.id, field = name, value = %value, "requesty: unreadable field, ignoring it");
        }
        parsed
    }

    /// Requesty sends `0` for a limit it does not know. Left as `Some(0)` it
    /// would beat the spec fallback and go out as `"max_tokens": 0`.
    fn limit(&self, name: &str) -> Option<u32> {
        self.read(name, |v| u32::try_from(v.as_u64()?).ok())
            .filter(|n| *n > 0)
    }

    /// Prices arrive per token, `ModelPricing` wants $/M.
    fn price(&self, name: &str) -> Option<f64> {
        self.read(name, Value::as_f64).map(|v| v * PER_MILLION)
    }

    fn flag(&self, name: &str) -> bool {
        self.read(name, Value::as_bool) == Some(true)
    }
}

/// Managed policies and the full catalog share this shape, so one parser reads
/// both.
fn parse_model(m: &Value) -> Option<ModelInfo> {
    let id = m["id"].as_str()?;

    // The catalog also lists embedding and other non chat APIs.
    if m["api"].as_str().is_some_and(|api| api != CHAT_API) {
        return None;
    }

    let entry = Entry { id, raw: m };

    // Half a price is no price: without both sides it would read as free.
    let pricing = match (entry.price("input_price"), entry.price("output_price")) {
        (Some(input), Some(output)) => Some(ModelPricing::per_million(
            input,
            output,
            entry.price("caching_price").unwrap_or(0.0),
            entry.price("cached_price").unwrap_or(0.0),
        )),
        _ => None,
    };

    Some(ModelInfo {
        id: id.to_string(),
        context_window: entry.limit("context_window"),
        max_output_tokens: entry.limit("max_output_tokens"),
        pricing,
        supports_thinking: Some(entry.flag("supports_reasoning")),
        supports_vision: Some(entry.flag("supports_vision")),
        tier: None,
        provider_info: None,
    })
}

/// Managed policies first, then the full catalog, deduplicated by id.
fn merge_models(managed: Vec<ModelInfo>, catalog: Vec<ModelInfo>) -> Vec<ModelInfo> {
    let mut merged = managed;
    for model in catalog {
        if !merged.iter().any(|m| m.id == model.id) {
            merged.push(model);
        }
    }
    merged
}

impl Provider for Requesty {
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

            // Requesty only inserts Anthropic cache breakpoints when asked to.
            // Without this flag Claude pays full input price every turn.
            body["requesty"] = json!({"auto_cache": true});

            if model.supports_thinking() {
                opts.thinking
                    .apply_reasoning_effort(&mut body, &dialect::PREFER_HIGH, model);
            }

            let extra_headers = [("HTTP-Referer", REFERER), ("X-Title", APP_TITLE)];
            self.compat
                .do_stream(model, &extra_headers, &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            // Both listings at once: the picker only shows up once the slowest
            // provider answers, so back to back round trips here cost everyone.
            let (managed, catalog) = futures_lite::future::zip(
                self.compat
                    .fetch_and_parse_models(&auth, MANAGED_MODELS_PATH, parse_model),
                self.compat
                    .fetch_and_parse_models(&auth, MODELS_PATH, parse_model),
            )
            .await;
            match (managed, catalog) {
                (Ok(managed), Ok(catalog)) => Ok(merge_models(managed, catalog)),
                (Ok(managed), Err(e)) => {
                    warn!(error = %e, "requesty: full catalog unavailable, listing managed models only");
                    Ok(managed)
                }
                (Err(e), Ok(catalog)) => {
                    warn!(error = %e, "requesty: managed models unavailable, listing full catalog only");
                    Ok(catalog)
                }
                (Err(e), Err(_)) => Err(e),
            }
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

    const SONNET_ID: &str = "anthropic/claude-sonnet-4-5";
    const UNKNOWN_PRICE_STAYS_UNKNOWN: &str = "a price we cannot read must not become a zero price";
    const EPSILON: f64 = 1e-9;

    fn sonnet_json() -> Value {
        json!({
            "id": SONNET_ID,
            "api": "chat",
            "context_window": 200_000,
            "max_output_tokens": 64_000,
            "input_price": 0.000003,
            "output_price": 0.000015,
            "caching_price": 0.00000375,
            "cached_price": 0.0000003,
            "supports_reasoning": true,
            "supports_vision": true,
        })
    }

    fn assert_price(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected ${expected}/M, got ${actual}/M"
        );
    }

    #[test]
    fn parse_model_reads_an_entry_and_scales_prices_to_per_million() {
        let info = parse_model(&sonnet_json()).expect("model should parse");

        assert_eq!(info.id, SONNET_ID);
        assert_eq!(info.context_window, Some(200_000));
        assert_eq!(info.max_output_tokens, Some(64_000));
        assert_eq!(info.supports_vision, Some(true));
        assert_eq!(info.supports_thinking, Some(true));
        let pricing = info.pricing.expect("pricing should be parsed");
        assert_price(pricing.input, 3.0);
        assert_price(pricing.output, 15.0);
        assert_price(pricing.cache_write, 3.75);
        assert_price(pricing.cache_read, 0.3);
    }

    /// Half a price is no price: an all-zero `ModelPricing` reads as free
    /// everywhere downstream.
    #[test_case(json!(null), json!(null)     ; "no_prices")]
    #[test_case(json!(0.000003), json!(null) ; "no_output_price")]
    #[test_case(json!(null), json!(0.000015) ; "no_input_price")]
    fn parse_model_keeps_unusable_pricing_unknown(input: Value, output: Value) {
        let mut m = sonnet_json();
        m["input_price"] = input;
        m["output_price"] = output;

        let info = parse_model(&m).expect("model should parse");
        assert!(info.pricing.is_none(), "{UNKNOWN_PRICE_STAYS_UNKNOWN}");
    }

    /// Requesty reports `0` for a limit it does not know. Kept as `Some(0)`
    /// it would win over the spec fallback and go out as `"max_tokens": 0`.
    #[test]
    fn parse_model_reads_zero_limits_as_unknown() {
        let mut m = sonnet_json();
        m["context_window"] = json!(0);
        m["max_output_tokens"] = json!(0);

        let info = parse_model(&m).expect("model should parse");
        assert_eq!(info.context_window, None);
        assert_eq!(info.max_output_tokens, None);
    }

    /// One field changing shape upstream costs that field, not the model.
    #[test]
    fn parse_model_keeps_model_when_one_field_is_unreadable() {
        let mut m = sonnet_json();
        m["input_price"] = json!("0.000003");
        m["max_output_tokens"] = json!("64000");

        let info = parse_model(&m).expect("model should still parse");
        assert_eq!(info.context_window, Some(200_000));
        assert_eq!(info.max_output_tokens, None);
        assert!(info.pricing.is_none(), "{UNKNOWN_PRICE_STAYS_UNKNOWN}");
        assert_eq!(info.supports_thinking, Some(true));
    }

    #[test]
    fn parse_model_without_capability_flags_reports_none_supported() {
        let mut m = sonnet_json();
        m["supports_reasoning"] = json!(null);
        m["supports_vision"] = json!(null);

        let info = parse_model(&m).expect("model should parse");
        assert_eq!(info.supports_thinking, Some(false));
        assert_eq!(info.supports_vision, Some(false));
    }

    #[test_case(json!("embedding"), false ; "embedding_is_skipped")]
    #[test_case(json!("image"), false     ; "image_is_skipped")]
    #[test_case(json!(null), true         ; "unlabelled_is_kept")]
    fn parse_model_keeps_only_chat_entries(api: Value, kept: bool) {
        let mut m = sonnet_json();
        m["api"] = api;

        assert_eq!(parse_model(&m).is_some(), kept);
    }

    #[test]
    fn parse_model_without_id_is_skipped() {
        let mut m = sonnet_json();
        m["id"] = json!(null);

        assert!(parse_model(&m).is_none());
    }

    #[test]
    fn merge_models_lists_managed_first_and_dedupes() {
        let managed = vec![
            ModelInfo::id_only("claude-sonnet-4-5".into()),
            ModelInfo::id_only("gpt-5.4-mini".into()),
        ];
        let catalog = vec![
            ModelInfo::id_only(SONNET_ID.into()),
            ModelInfo::id_only("gpt-5.4-mini".into()),
            ModelInfo::id_only("openai/gpt-4o-mini".into()),
        ];

        let ids: Vec<String> = merge_models(managed, catalog)
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(
            ids,
            [
                "claude-sonnet-4-5",
                "gpt-5.4-mini",
                SONNET_ID,
                "openai/gpt-4o-mini",
            ]
        );
    }
}

/// The recorded cases, kept out of the test module so every authoring replays
/// the same list. Recorded against the bespoke `Requesty` impl above.
///
/// The two catalog fetches run concurrently, so every script that answers
/// them is routed by path: arrival order is a race, and a sequential script
/// would hand each listing the other's answer.
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures {
    use crate::model::{Model, ThinkingSupport};
    use crate::providers::replay::Fixture;
    use crate::test_support::Canned;
    use crate::{Effort, ThinkingConfig};

    /// Thinking-capable, so the effort reaches the wire.
    const THINKING_SPEC: &str = "requesty/openai/gpt-5.5";
    /// Listed with `supports_reasoning: false` in the catalog listing, which is
    /// what [`DISCOVERED_NON_THINKING`] leans on.
    pub const NON_THINKING_SPEC: &str = "requesty/openai/gpt-4o-mini";
    /// `prefer-high` tops out at `high`, so the golden shows the snap.
    const EFFORT: Effort = Effort::Max;

    const UNKNOWN_MODEL: &str = "the model spec did not resolve";

    const MANAGED_PATH: &str = "/v1/models/managed";
    const CATALOG_PATH: &str = "/v1/models";
    const CHAT_PATH: &str = "/v1/chat/completions";

    const UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"invalid api key"}}"#;
    const SERVER_ERROR_BODY: &str = r#"{"error":{"message":"internal error"}}"#;

    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"reasoning_content":"weighing the options"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}

data: [DONE]

"#;

    /// Unsorted on purpose, with an id listed twice (equal sort keys, both
    /// kept in arrival order), a zero limit, a price as a string, a flag that
    /// is not a bool, a null, and two entries with no usable id.
    const MANAGED_BODY: &str = r#"{"data":[
{"id":"gpt-5.4-mini","api":"chat","context_window":400000,"max_output_tokens":128000,"input_price":0.00000025,"output_price":0.000002,"cached_price":0.000000025,"supports_reasoning":true,"supports_vision":true},
{"id":"claude-sonnet-4-5","api":"chat","context_window":200000,"max_output_tokens":64000,"input_price":0.000003,"output_price":0.000015,"caching_price":0.00000375,"cached_price":0.0000003,"supports_reasoning":true,"supports_vision":true},
{"id":"claude-sonnet-4-5","api":"chat","context_window":1000000,"max_output_tokens":64000,"input_price":0.000006,"output_price":0.0000225,"supports_reasoning":true,"supports_vision":true},
{"id":"claude-sonnet-4-5@eu","api":"chat","context_window":0,"max_output_tokens":0,"input_price":"0.000003","output_price":0.000015,"supports_reasoning":null,"supports_vision":"yes"},
{"api":"chat","context_window":128000},
{"id":42,"api":"chat"}
]}"#;

    /// Shares `gpt-5.4-mini` with `MANAGED_BODY` (the managed row wins),
    /// lists `vendor/twin` twice (merge keeps the first), and carries a non
    /// chat api, whole and fractional floats for limits, negatives, a limit
    /// past `u32`, a `.125` price, all-null fields, unreadable prices, a zero
    /// price, an entry with no `api` and one that is not an object.
    const CATALOG_BODY: &str = r#"{"data":[
{"id":"vendor/twin","api":"chat","context_window":32000,"input_price":0,"output_price":0},
{"id":"openai/text-embedding-3-small","api":"embedding","context_window":8191,"input_price":0.00000002,"output_price":0},
{"id":"openai/gpt-4o-mini","api":"chat","context_window":128000,"max_output_tokens":16384,"input_price":0.00000015,"output_price":0.0000006,"cached_price":0.000000075,"supports_reasoning":false,"supports_vision":true},
{"id":"gpt-5.4-mini","api":"chat","context_window":272000,"max_output_tokens":32000,"input_price":0.0000005,"output_price":0.000004,"supports_reasoning":false,"supports_vision":false},
{"id":"vendor/floats","api":"chat","context_window":131072.0,"max_output_tokens":8192.5,"input_price":0.000000125,"output_price":0.000000375,"supports_reasoning":1},
{"id":"vendor/negatives","api":"chat","context_window":-1,"max_output_tokens":4294967296,"input_price":-0.000001,"output_price":0.000002},
{"id":"vendor/nulls","api":null,"context_window":null,"max_output_tokens":null,"input_price":null,"output_price":null,"caching_price":null,"cached_price":null,"supports_reasoning":null,"supports_vision":null},
{"id":"vendor/bad-price","api":"chat","context_window":65536,"input_price":"free","output_price":{"per_token":0.000001},"cached_price":"0.1"},
{"id":"vendor/half-cache","input_price":0.000001,"output_price":0.000002,"caching_price":"n/a"},
{"id":"vendor/twin","api":"chat","context_window":64000,"input_price":0.000001,"output_price":0.000002},
"stray"
]}"#;

    const SUCCESS_SCRIPT: &[Canned] = &[Canned::sse(SUCCESS_TRANSCRIPT)];

    pub const SUCCESS: Fixture = Fixture {
        name: "success",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
        session: None,
    };
    /// The same effort on a model without thinking: `reasoning_effort` stays
    /// off the wire.
    pub const SUCCESS_NON_THINKING: Fixture = Fixture {
        name: "success_non_thinking",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
        session: None,
    };
    /// The gate fed by discovery rather than an override: the listing marks
    /// [`NON_THINKING_SPEC`] as not reasoning, so the turn after it sends no
    /// effort.
    pub const DISCOVERED_NON_THINKING: Fixture = Fixture {
        name: "discovered_non_thinking",
        script: &[
            Canned::at(MANAGED_PATH, Canned::json(200, MANAGED_BODY)),
            Canned::at(CATALOG_PATH, Canned::json(200, CATALOG_BODY)),
            Canned::at(CHAT_PATH, Canned::sse(SUCCESS_TRANSCRIPT)),
        ],
        thinking: ThinkingConfig::Effort(EFFORT),
        session: None,
    };
    pub const MODELS: Fixture = Fixture {
        name: "models",
        script: &[
            Canned::at(MANAGED_PATH, Canned::json(200, MANAGED_BODY)),
            Canned::at(CATALOG_PATH, Canned::json(200, CATALOG_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// Lists the catalog alone.
    pub const MODELS_MANAGED_DOWN: Fixture = Fixture {
        name: "models_managed_down",
        script: &[
            Canned::at(MANAGED_PATH, Canned::json(500, SERVER_ERROR_BODY)),
            Canned::at(CATALOG_PATH, Canned::json(200, CATALOG_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// Lists the managed policies alone.
    pub const MODELS_CATALOG_DOWN: Fixture = Fixture {
        name: "models_catalog_down",
        script: &[
            Canned::at(MANAGED_PATH, Canned::json(200, MANAGED_BODY)),
            Canned::at(CATALOG_PATH, Canned::json(500, SERVER_ERROR_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// Two different failures, so the golden shows it is the managed one that
    /// surfaces.
    pub const MODELS_BOTH_DOWN: Fixture = Fixture {
        name: "models_both_down",
        script: &[
            Canned::at(MANAGED_PATH, Canned::json(401, UNAUTHORIZED_BODY)),
            Canned::at(CATALOG_PATH, Canned::json(500, SERVER_ERROR_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };

    pub fn model(spec: &str) -> Model {
        Model::from_spec(spec).expect(UNKNOWN_MODEL)
    }

    /// Support is pinned rather than left to discovery or the models.dev
    /// cache, so the gate reads the same in every process.
    fn pinned(spec: &str, support: ThinkingSupport) -> Model {
        let mut model = model(spec);
        model.thinking_override = Some(support);
        model
    }

    pub fn thinking_model() -> Model {
        pinned(THINKING_SPEC, ThinkingSupport::Yes)
    }

    pub fn non_thinking_model() -> Model {
        pinned(NON_THINKING_SPEC, ThinkingSupport::No)
    }
}

/// The bespoke impl, recorded one exchange at a time.
#[cfg(test)]
mod replay_tests {
    use test_case::test_case;

    use crate::model::Model;
    use crate::providers::replay::{self, Fixture};

    use super::SLUG;
    use super::fixtures::{
        DISCOVERED_NON_THINKING, MODELS, MODELS_BOTH_DOWN, MODELS_CATALOG_DOWN,
        MODELS_MANAGED_DOWN, NON_THINKING_SPEC, SUCCESS, SUCCESS_NON_THINKING, model,
        non_thinking_model, thinking_model,
    };

    #[test_case(&SUCCESS, thinking_model() ; "success")]
    #[test_case(&SUCCESS_NON_THINKING, non_thinking_model() ; "success_non_thinking")]
    #[test_case(&replay::UNAUTHORIZED, thinking_model() ; "unauthorized")]
    #[test_case(&replay::SLOW_DOWN, thinking_model() ; "rate_limited_with_retry_after")]
    #[test_case(&replay::RATE_LIMITED, thinking_model() ; "rate_limited")]
    #[test_case(&replay::SERVER_ERROR, thinking_model() ; "server_error")]
    #[test_case(&replay::MALFORMED_SSE, thinking_model() ; "malformed_sse")]
    #[test_case(&replay::EMPTY_SSE_ERROR, thinking_model() ; "empty_sse_error_frame")]
    #[test_case(&replay::TRUNCATED_STREAM, thinking_model() ; "truncated_stream")]
    fn the_bespoke_impl_replays_the_recorded_exchange(fixture: &Fixture, model: Model) {
        replay::bespoke(SLUG).stream(fixture, &model);
    }

    #[test_case(&MODELS ; "models")]
    #[test_case(&MODELS_MANAGED_DOWN ; "models_managed_down")]
    #[test_case(&MODELS_CATALOG_DOWN ; "models_catalog_down")]
    #[test_case(&MODELS_BOTH_DOWN ; "models_both_down")]
    fn the_bespoke_impl_lists_the_recorded_catalogs(fixture: &Fixture) {
        replay::bespoke(SLUG).models(fixture);
    }

    #[test]
    fn the_bespoke_impl_gates_the_effort_on_discovery() {
        replay::bespoke(SLUG).discovered(&DISCOVERED_NON_THINKING, &model(NON_THINKING_SPEC));
    }
}
