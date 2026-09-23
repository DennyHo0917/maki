use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::sync::Arc;

use futures_lite::future::zip;
use jiff::Timestamp;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use maki_config::providers::Protocol;

use crate::model::{ModelFamily, ModelInfo, ModelPricing};
use crate::provider::BoxFuture;
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, ProviderSpec};
use crate::types::{ModelUsageRow, ProviderUsage, UsageLimit, rfc3339_millis};
use crate::{AgentError, dialect};

use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use super::plugin::{
    self, EffortField, Hook, OpenAiWire, ProviderDecl, ProviderHooks, ThinkingWire,
};
use super::{ResolvedAuth, Timeouts};

const SLUG: &str = "regolo";
const DISPLAY_NAME: &str = "Regolo";
const ENV_VAR: &str = "REGOLO_API_KEY";
const BASE_URL: &str = "https://api.regolo.ai/v1";
const DEFAULT_MODEL: &str = "regolo/qwen3-coder-next";
const LOGIN_URL: &str = "https://dashboard.regolo.ai";
const MAX_TOKENS_FIELD: &str = "max_completion_tokens";
const FEATURES: &str = "EU-hosted open-weight models with tool calling. The catalogue and prices are listed live from the API";
const NET_HOST: &str = "api.regolo.ai";

const KEY_INFO_PATH: &str = "/key/info";
const ACTIVITY_PATH: &str = "/global/activity";
const SPEND_LOGS_PATH: &str = "/spend/logs/v2";
const MODEL_GROUP_INFO_PATH: &str = "/model_group/info";
const VERSION_SEGMENT: &str = "/v1";
const CHAT_MODE: &str = "chat";
const PER_MILLION: f64 = 1_000_000.0;
const SECONDS_PER_DAY: i64 = 86_400;

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
    fallback_max_output: Some(120_000),
    fallback_context_window: 120_000,
    models_toml: include_str!("../../models/regolo.toml"),
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
        catalog: CatalogDoc::Table,
        trailing_notes: &[],
    },
};

inventory::submit!(SPEC.config_row());

/// Regolo as a declaration, plus the catalogue and the usage report the openai
/// codec cannot fetch, see [`hooks`].
///
/// Only what the codec cannot guess is stated here: Regolo takes
/// `max_completion_tokens` and the `standard` effort ladder. Claiming a
/// built-in slug inherits the rest of [`SPEC`], and streamed usage is already
/// the codec's default.
///
/// The bundled `regolo` Lua plugin says all of this again on the surface a
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
            max_tokens_field: Some(MAX_TOKENS_FIELD.to_owned()),
            thinking: Some(ThinkingWire {
                dialect: &dialect::STANDARD,
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
        list_models: Some(Arc::new(Catalogue)),
        fetch_usage: Some(Arc::new(Spend)),
        ..ProviderHooks::default()
    }
}

fn registered() -> Result<(ResolvedAuth, OpenAiCompatProvider), AgentError> {
    let auth = plugin::registered_auth(SLUG)?;
    let compat = OpenAiCompatProvider::new(&CONFIG, Timeouts::default());
    Ok((auth, compat))
}

/// Regolo's management endpoints live at the host root, outside `/v1`, so
/// they must not be absolute urls: a custom base url or a gateway (Aperture)
/// keeps serving them, an absolute one would bypass it and 401 on its key.
fn root_url(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    trimmed
        .strip_suffix(VERSION_SEGMENT)
        .unwrap_or(trimmed)
        .to_string()
}

/// The callable ids from `/v1/models`, joined with `/model_group/info`.
struct Catalogue;

impl Hook<(), Vec<ModelInfo>> for Catalogue {
    fn call(&self, (): ()) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let (auth, compat) = registered()?;
            let ids: Vec<String> = compat
                .do_list_models(&auth)
                .await?
                .into_iter()
                .map(|info| info.id)
                .collect();
            let root = root_url(&compat.base_url(&auth));
            let url = format!("{root}{MODEL_GROUP_INFO_PATH}");
            let groups = fetch_optional::<ModelGroupInfoResponse>(&compat, &auth, &url).await;
            Ok(match groups {
                Some(groups) => join_model_info(ids, groups.data),
                // The endpoint hiccups (it has 500ed): keep the live ids
                // without metadata instead of dropping to the static few.
                None => ids.into_iter().map(ModelInfo::id_only).collect(),
            })
        })
    }
}

/// The key's spend against its budget, which the report cannot do without,
/// then today's activity and per-model spend, fetched together and each
/// dropped on its own when it fails.
struct Spend;

impl Hook<(), Option<ProviderUsage>> for Spend {
    fn call(&self, (): ()) -> BoxFuture<'_, Result<Option<ProviderUsage>, AgentError>> {
        Box::pin(async move {
            let (auth, compat) = registered()?;
            let root = root_url(&compat.base_url(&auth));
            let key_body = compat
                .get_text(&auth, &format!("{root}{KEY_INFO_PATH}"))
                .await?;
            let key_info: KeyInfoResponse = serde_json::from_str(&key_body)?;
            let today = Timestamp::now().to_zoned(TimeZone::UTC).date();
            let (activity, spend_logs) = zip(
                fetch_optional::<ActivityResponse>(
                    &compat,
                    &auth,
                    &day_url(&root, ACTIVITY_PATH, today),
                ),
                fetch_optional::<SpendLogsResponse>(
                    &compat,
                    &auth,
                    &day_url(&root, SPEND_LOGS_PATH, today),
                ),
            )
            .await;
            let mut limits = vec![UsageLimit::from(key_info.info)];
            limits.extend(activity.map(UsageLimit::from));
            Ok(Some(ProviderUsage {
                plan: None,
                limits,
                by_model_today: spend_logs
                    .map(SpendLogsResponse::into_rows)
                    .unwrap_or_default(),
            }))
        })
    }
}

/// A side answer the caller can do without: any failure, the status or the
/// shape, reads as no answer.
async fn fetch_optional<T: DeserializeOwned>(
    compat: &OpenAiCompatProvider,
    auth: &ResolvedAuth,
    url: &str,
) -> Option<T> {
    let body = compat.get_text(auth, url).await.ok()?;
    serde_json::from_str(&body).ok()
}

/// `/global/activity` is key-scoped despite its name and answers per UTC day.
/// `/spend/logs/v2` takes the same end-inclusive YYYY-MM-DD range and answers
/// one row per model per hour, so its rows are summed per model.
fn day_url(root: &str, path: &str, day: Date) -> String {
    format!("{root}{path}?start_date={day}&end_date={day}")
}

#[derive(Deserialize)]
struct SpendLogsResponse {
    data: Vec<SpendLogEntry>,
}

#[derive(Deserialize)]
struct SpendLogEntry {
    model_group: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    /// USD; rendered as micro-dollars at the boundary to keep `ProviderUsage`
    /// in `Eq` land.
    spend: f64,
}

impl SpendLogsResponse {
    fn into_rows(self) -> Vec<ModelUsageRow> {
        let mut by_model: BTreeMap<String, SpendLogEntry> = BTreeMap::new();
        for entry in self.data {
            by_model
                .entry(entry.model_group.clone())
                .and_modify(|acc| {
                    acc.prompt_tokens += entry.prompt_tokens;
                    acc.completion_tokens += entry.completion_tokens;
                    acc.total_tokens += entry.total_tokens;
                    acc.spend += entry.spend;
                })
                .or_insert(entry);
        }
        let mut rows: Vec<ModelUsageRow> = by_model
            .into_values()
            .map(|e| ModelUsageRow {
                model: e.model_group,
                input_tokens: e.prompt_tokens,
                output_tokens: e.completion_tokens,
                total_tokens: e.total_tokens,
                spend_microdollars: (e.spend * PER_MILLION).round() as u64,
            })
            .collect();
        rows.sort_by_key(|row| Reverse(row.spend_microdollars));
        rows
    }
}

#[derive(Deserialize)]
struct KeyInfoResponse {
    info: KeyInfo,
}

#[derive(Deserialize)]
struct KeyInfo {
    spend: f64,
    max_budget: Option<f64>,
    budget_reset_at: Option<String>,
}

impl From<KeyInfo> for UsageLimit {
    fn from(info: KeyInfo) -> Self {
        let mut detail = format!("${:.2} spent", info.spend);
        if let Some(budget) = info.max_budget {
            detail.push_str(&format!(" of ${budget:.2} budget"));
        }
        let percentage = info
            .max_budget
            .filter(|budget| *budget > 0.0)
            .map(|budget| ((info.spend / budget * 100.0) as u32).min(100));
        Self {
            label: "Spend".into(),
            percentage,
            reset_at: info.budget_reset_at.as_deref().and_then(rfc3339_millis),
            detail: Some(detail),
        }
    }
}

#[derive(Deserialize)]
struct ActivityResponse {
    sum_api_requests: u64,
    sum_total_tokens: u64,
}

fn next_utc_midnight(now: Timestamp) -> Option<u64> {
    let day = now.as_second().div_euclid(SECONDS_PER_DAY) + 1;
    Timestamp::from_second(day * SECONDS_PER_DAY)
        .ok()
        .map(|at| at.as_millisecond() as u64)
}

impl From<ActivityResponse> for UsageLimit {
    fn from(activity: ActivityResponse) -> Self {
        Self {
            label: "Today".into(),
            // Regolo tracks a per-account daily token cap (free trial: 1M) but
            // no endpoint reports it, so there is no honest percentage to show.
            percentage: None,
            reset_at: next_utc_midnight(Timestamp::now()),
            detail: Some(format!(
                "{} requests · {} tokens",
                activity.sum_api_requests, activity.sum_total_tokens
            )),
        }
    }
}

#[derive(Deserialize)]
struct ModelGroupInfoResponse {
    data: Vec<ModelGroup>,
}

#[derive(Deserialize)]
struct ModelGroup {
    model_group: String,
    mode: String,
    input_cost_per_token: Option<f64>,
    output_cost_per_token: Option<f64>,
    max_input_tokens: Option<f64>,
    max_output_tokens: Option<f64>,
    max_tokens: Option<f64>,
    supports_reasoning: bool,
    supports_vision: bool,
}

/// Joins the callable IDs from `/v1/models` with the metadata from
/// `/model_group/info`. Groups in a non-chat `mode` (embedding, rerank, ocr,
/// image, audio) and IDs without a chat group are not agent models. Order
/// follows `ids`, which the compat lister already sorts.
fn join_model_info(ids: Vec<String>, groups: Vec<ModelGroup>) -> Vec<ModelInfo> {
    let by_group: BTreeMap<&str, &ModelGroup> = groups
        .iter()
        .filter(|group| group.mode == CHAT_MODE)
        .map(|group| (group.model_group.as_str(), group))
        .collect();
    ids.into_iter()
        .filter_map(|id| {
            let group = by_group.get(id.as_str())?;
            let pricing = match (group.input_cost_per_token, group.output_cost_per_token) {
                (Some(input), Some(output)) => Some(ModelPricing::per_million(
                    input * PER_MILLION,
                    output * PER_MILLION,
                    0.00,
                    0.00,
                )),
                _ => None,
            };
            Some(ModelInfo {
                context_window: group
                    .max_input_tokens
                    .or(group.max_tokens)
                    .and_then(|tokens| u32::try_from(tokens as u64).ok()),
                max_output_tokens: group
                    .max_output_tokens
                    .and_then(|tokens| u32::try_from(tokens as u64).ok()),
                pricing,
                supports_thinking: Some(group.supports_reasoning),
                supports_vision: Some(group.supports_vision),
                ..ModelInfo::id_only(id)
            })
        })
        .collect()
}

/// The recorded cases, kept out of the test modules so both authorings replay
/// the same list: [`decl`] plus [`hooks`], and the bundled `regolo` Lua plugin.
///
/// Loopback serves `<origin>/v1`, so the chat and `/models` requests land under
/// `/v1` and the management endpoints at the root, which is what
/// `root_url` strips the version segment for. Every script that reaches more
/// than one endpoint is routed by path: the usage side calls may race, and a
/// routed script keeps the observation free of arrival order.
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures {
    use crate::model::Model;
    use crate::providers::replay::Fixture;
    use crate::test_support::Canned;
    use crate::{Effort, ThinkingConfig};

    /// Spelled out instead of reusing `DEFAULT_MODEL`, so a new default can
    /// never move the goldens, which are not recorded again.
    const MODEL_SPEC: &str = "regolo/qwen3-coder-next";
    /// Above what the `standard` dialect accepts, so the wire shows it snapped
    /// to `high` rather than passed through.
    const EFFORT: Effort = Effort::XHigh;
    const UNKNOWN_MODEL: &str = "the curated table has no such model";

    const MODELS_PATH: &str = "/v1/models";
    const MODEL_GROUP_INFO_PATH: &str = "/model_group/info";
    const KEY_INFO_PATH: &str = "/key/info";
    const ACTIVITY_PATH: &str = "/global/activity";
    const SPEND_LOGS_PATH: &str = "/spend/logs/v2";

    const UNAUTHORIZED_BODY: &str =
        r#"{"error":{"message":"Authentication Error, Invalid proxy server token passed."}}"#;
    const SERVER_ERROR_BODY: &str = r#"{"error":{"message":"internal server error"}}"#;

    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"reasoning_content":"weighing the options"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17,"prompt_tokens_details":{"cached_tokens":4}}}

data: [DONE]

"#;

    const SUCCESS_SCRIPT: &[Canned] = &[Canned::sse(SUCCESS_TRANSCRIPT)];

    pub const SUCCESS: Fixture = Fixture {
        name: "success",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
        session: None,
    };
    /// The `standard` dialect has no spelling for off, so nothing reaches the
    /// wire.
    pub const THINKING_OFF: Fixture = Fixture {
        name: "thinking_off",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Off,
        session: None,
    };

    /// Unsorted, with a duplicate id, an upper-case id that sorts first
    /// bytewise, an id no group describes, and three entries without a string
    /// id. The per-model fields the compat lister reads are ignored here.
    const MODELS_BODY: &str = r#"{"object":"list","data":[
        {"id":"qwen3.5-9b","object":"model","owned_by":"regolo","context_length":4096},
        {"id":"glm5.2","object":"model"},
        {"id":"Qwen3-Embedding-8B","object":"model"},
        {"id":"qwen3-coder-next","object":"model"},
        {"id":"gpt-oss-120b","object":"model"},
        {"id":"glm5.2","object":"model"},
        {"id":"llama-huge","object":"model"},
        {"id":"brand-new-model","object":"model"},
        {"id":"legacy-negative","object":"model"},
        {"id":"half-priced","object":"model"},
        {"id":null,"object":"model"},
        {"object":"model"},
        {"id":42,"object":"model"}
    ]}"#;

    /// `glm5.2` is described three times: the later chat group replaces the
    /// first, and the trailing non-chat one is filtered out before it could.
    /// Token limits come as floats, fractions, integers, negatives, nulls and
    /// values past `u32`. `llama-huge` shows an out-of-range input window does
    /// not fall back to `max_tokens`. `gpt-oss-120b` prices at `0.125` per
    /// million, `half-priced` has one price null, and `orphan-chat` has no id.
    const MODEL_GROUPS_BODY: &str = r#"{"data":[
        {"model_group":"glm5.2","mode":"chat","input_cost_per_token":2e-06,"output_cost_per_token":5.2e-06,"max_input_tokens":96000.0,"max_output_tokens":96000.0,"max_tokens":null,"supports_reasoning":true,"supports_vision":false},
        {"model_group":"qwen3-coder-next","mode":"chat","input_cost_per_token":5e-07,"output_cost_per_token":2e-06,"max_input_tokens":null,"max_output_tokens":120000,"max_tokens":240000.0,"supports_reasoning":false,"supports_vision":true,"supports_function_calling":true},
        {"model_group":"qwen3.5-9b","mode":"chat","input_cost_per_token":7e-08,"output_cost_per_token":3.5e-07,"max_input_tokens":80000.7,"max_output_tokens":120000.0,"max_tokens":200000.0,"supports_reasoning":true,"supports_vision":false},
        {"model_group":"gpt-oss-120b","mode":"chat","input_cost_per_token":1.25e-07,"output_cost_per_token":6.25e-07,"max_input_tokens":131072,"max_output_tokens":null,"max_tokens":null,"supports_reasoning":true,"supports_vision":false},
        {"model_group":"llama-huge","mode":"chat","input_cost_per_token":1e-06,"output_cost_per_token":1e-06,"max_input_tokens":5000000000.0,"max_output_tokens":4294967296,"max_tokens":131072.0,"supports_reasoning":false,"supports_vision":false},
        {"model_group":"legacy-negative","mode":"chat","input_cost_per_token":-1e-06,"output_cost_per_token":0.0,"max_input_tokens":-1.0,"max_output_tokens":-512,"max_tokens":null,"supports_reasoning":false,"supports_vision":false},
        {"model_group":"half-priced","mode":"chat","input_cost_per_token":1e-06,"output_cost_per_token":null,"max_input_tokens":null,"max_output_tokens":null,"max_tokens":null,"supports_reasoning":false,"supports_vision":false},
        {"model_group":"Qwen3-Embedding-8B","mode":"embedding","input_cost_per_token":0.0,"output_cost_per_token":0.0,"max_input_tokens":32000.0,"max_output_tokens":null,"max_tokens":null,"supports_reasoning":false,"supports_vision":false},
        {"model_group":"orphan-chat","mode":"chat","input_cost_per_token":1e-06,"output_cost_per_token":1e-06,"max_input_tokens":8192.0,"max_output_tokens":8192.0,"max_tokens":null,"supports_reasoning":false,"supports_vision":false},
        {"model_group":"glm5.2","mode":"chat","input_cost_per_token":2e-06,"output_cost_per_token":5.2e-06,"max_input_tokens":64000.0,"max_output_tokens":32000.0,"max_tokens":null,"supports_reasoning":true,"supports_vision":true},
        {"model_group":"glm5.2","mode":"rerank","input_cost_per_token":null,"output_cost_per_token":null,"max_input_tokens":null,"max_output_tokens":null,"max_tokens":null,"supports_reasoning":false,"supports_vision":false}
    ]}"#;

    /// One price as a string fails the whole response, not the one group.
    const MALFORMED_MODEL_GROUPS_BODY: &str = r#"{"data":[
        {"model_group":"qwen3-coder-next","mode":"chat","input_cost_per_token":5e-07,"output_cost_per_token":2e-06,"max_input_tokens":null,"max_output_tokens":120000.0,"max_tokens":240000.0,"supports_reasoning":false,"supports_vision":true},
        {"model_group":"glm5.2","mode":"chat","input_cost_per_token":"0.000002","output_cost_per_token":5.2e-06,"max_input_tokens":96000.0,"max_output_tokens":96000.0,"max_tokens":null,"supports_reasoning":true,"supports_vision":false}
    ]}"#;

    const MODELS_OK: Canned = Canned::at(MODELS_PATH, Canned::json(200, MODELS_BODY));

    pub const MODELS: Fixture = Fixture {
        name: "models",
        script: &[
            MODELS_OK,
            Canned::at(MODEL_GROUP_INFO_PATH, Canned::json(200, MODEL_GROUPS_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// The group endpoint has 500ed in the wild: the live ids stay, bare.
    pub const MODELS_WITHOUT_GROUPS: Fixture = Fixture {
        name: "models_without_groups",
        script: &[
            MODELS_OK,
            Canned::at(MODEL_GROUP_INFO_PATH, Canned::json(500, SERVER_ERROR_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    pub const MODELS_MALFORMED_GROUPS: Fixture = Fixture {
        name: "models_malformed_groups",
        script: &[
            MODELS_OK,
            Canned::at(
                MODEL_GROUP_INFO_PATH,
                Canned::json(200, MALFORMED_MODEL_GROUPS_BODY),
            ),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// The group answer is an upper bound: a listing that fails on `/models`
    /// never asks for it.
    pub const MODELS_UNAUTHORIZED: Fixture = Fixture {
        name: "models_unauthorized",
        script: &[
            Canned::at(MODELS_PATH, Canned::json(401, UNAUTHORIZED_BODY)),
            Canned::at(MODEL_GROUP_INFO_PATH, Canned::json(200, MODEL_GROUPS_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };

    /// `0.125` and `0.375` are exact binary ties for `{:.2}`, rounding opposite
    /// ways under half-to-even. The reset carries an offset and a fraction.
    const KEY_INFO_BODY: &str = r#"{"key":"88dc28d0f030c55ed4ab77ed8faf0981","info":{"key_name":"sk-...abcd","key_alias":"me@example.com","spend":0.125,"max_budget":0.375,"budget_duration":"30d","budget_reset_at":"2026-10-01T02:00:00.25+02:00","models":[],"metadata":{}}}"#;
    /// A null budget, a reset that is absent rather than null, and a negative
    /// spend.
    const NO_BUDGET_KEY_INFO_BODY: &str =
        r#"{"key":"k","info":{"key_alias":null,"spend":-0.5,"max_budget":null}}"#;
    /// Past the budget, so the percentage caps, with a reset that is no
    /// timestamp.
    const OVER_BUDGET_KEY_INFO_BODY: &str =
        r#"{"key":"k","info":{"spend":12.5,"max_budget":10,"budget_reset_at":"next month"}}"#;

    const ACTIVITY_BODY: &str = r#"{"daily_data":[{"date":"2026-09-23","metrics":{"api_requests":14,"total_tokens":5000000000}}],"sum_api_requests":14,"sum_total_tokens":5000000000}"#;
    const EMPTY_ACTIVITY_BODY: &str =
        r#"{"daily_data":[],"sum_api_requests":0,"sum_total_tokens":0}"#;
    const MALFORMED_ACTIVITY_BODY: &str = r#"{"sum_api_requests":14.0,"sum_total_tokens":29027}"#;

    /// Hourly rows out of name order. `glm5.2` sums two hours to the same
    /// micro-dollars `gpt-oss-120b` spends in one, a tie the name order breaks.
    /// `qwen3.5-9b` spends half a micro-dollar past twelve and counts tokens
    /// past `u32`. `llama-refund` is a negative spend.
    const SPEND_LOGS_BODY: &str = r#"{"data":[
        {"request_id":"h1","api_key":"k","model_group":"gpt-oss-120b","key_alias":"a","startTime":"2026-09-23T09:00:00+00:00","endTime":"2026-09-23T10:00:00+00:00","completionStartTime":null,"api_requests":1,"prompt_tokens":1000,"completion_tokens":200,"total_tokens":1200,"spend":0.0009,"request_duration_ms":3600000},
        {"request_id":"h2","api_key":"k","model_group":"qwen3-coder-next","key_alias":"a","startTime":"2026-09-23T09:00:00+00:00","endTime":"2026-09-23T10:00:00+00:00","completionStartTime":null,"api_requests":4,"prompt_tokens":15000,"completion_tokens":5000,"total_tokens":20000,"spend":0.01,"request_duration_ms":3600000},
        {"request_id":"h3","api_key":"k","model_group":"glm5.2","key_alias":"a","startTime":"2026-09-23T09:00:00+00:00","endTime":"2026-09-23T10:00:00+00:00","completionStartTime":null,"api_requests":1,"prompt_tokens":120,"completion_tokens":80,"total_tokens":200,"spend":0.0004,"request_duration_ms":3600000},
        {"request_id":"h4","api_key":"k","model_group":"qwen3-coder-next","key_alias":"a","startTime":"2026-09-23T10:00:00+00:00","endTime":"2026-09-23T11:00:00+00:00","completionStartTime":null,"api_requests":3,"prompt_tokens":5000,"completion_tokens":3000,"total_tokens":8000,"spend":0.0042,"request_duration_ms":3600000},
        {"request_id":"h5","api_key":"k","model_group":"glm5.2","key_alias":"a","startTime":"2026-09-23T10:00:00+00:00","endTime":"2026-09-23T11:00:00+00:00","completionStartTime":null,"api_requests":1,"prompt_tokens":80,"completion_tokens":80,"total_tokens":160,"spend":0.0005,"request_duration_ms":3600000},
        {"request_id":"h6","api_key":"k","model_group":"qwen3.5-9b","key_alias":"a","startTime":"2026-09-23T11:00:00+00:00","endTime":"2026-09-23T12:00:00+00:00","completionStartTime":null,"api_requests":1,"prompt_tokens":4294967296,"completion_tokens":1,"total_tokens":4294967297,"spend":0.0000125,"request_duration_ms":3600000},
        {"request_id":"h7","api_key":"k","model_group":"llama-refund","key_alias":"a","startTime":"2026-09-23T11:00:00+00:00","endTime":"2026-09-23T12:00:00+00:00","completionStartTime":null,"api_requests":0,"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"spend":-0.0005,"request_duration_ms":3600000}
    ]}"#;
    const EMPTY_SPEND_LOGS_BODY: &str = r#"{"data":[]}"#;
    const MALFORMED_SPEND_LOGS_BODY: &str = r#"{"data":[{"model_group":"glm5.2","prompt_tokens":-5,"completion_tokens":1,"total_tokens":1,"spend":0.001}]}"#;

    const ACTIVITY_OK: Canned = Canned::at(ACTIVITY_PATH, Canned::json(200, ACTIVITY_BODY));
    const SPEND_LOGS_OK: Canned = Canned::at(SPEND_LOGS_PATH, Canned::json(200, SPEND_LOGS_BODY));
    const EMPTY_ACTIVITY: Canned =
        Canned::at(ACTIVITY_PATH, Canned::json(200, EMPTY_ACTIVITY_BODY));
    const EMPTY_SPEND_LOGS: Canned =
        Canned::at(SPEND_LOGS_PATH, Canned::json(200, EMPTY_SPEND_LOGS_BODY));

    pub const USAGE: Fixture = Fixture {
        name: "usage",
        script: &[
            Canned::at(KEY_INFO_PATH, Canned::json(200, KEY_INFO_BODY)),
            ACTIVITY_OK,
            SPEND_LOGS_OK,
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    pub const USAGE_NO_BUDGET: Fixture = Fixture {
        name: "usage_no_budget",
        script: &[
            Canned::at(KEY_INFO_PATH, Canned::json(200, NO_BUDGET_KEY_INFO_BODY)),
            EMPTY_ACTIVITY,
            EMPTY_SPEND_LOGS,
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    pub const USAGE_OVER_BUDGET: Fixture = Fixture {
        name: "usage_over_budget",
        script: &[
            Canned::at(KEY_INFO_PATH, Canned::json(200, OVER_BUDGET_KEY_INFO_BODY)),
            EMPTY_ACTIVITY,
            EMPTY_SPEND_LOGS,
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// The key report alone still makes a usage answer.
    pub const USAGE_SIDE_CALLS_FAIL: Fixture = Fixture {
        name: "usage_side_calls_fail",
        script: &[
            Canned::at(KEY_INFO_PATH, Canned::json(200, KEY_INFO_BODY)),
            Canned::at(ACTIVITY_PATH, Canned::json(500, SERVER_ERROR_BODY)),
            Canned::at(SPEND_LOGS_PATH, Canned::json(500, SERVER_ERROR_BODY)),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// A float where the activity counts are integers, and a negative token
    /// count in the spend logs: each answer is dropped whole, like a failure.
    pub const USAGE_SIDE_CALLS_MALFORMED: Fixture = Fixture {
        name: "usage_side_calls_malformed",
        script: &[
            Canned::at(KEY_INFO_PATH, Canned::json(200, KEY_INFO_BODY)),
            Canned::at(ACTIVITY_PATH, Canned::json(200, MALFORMED_ACTIVITY_BODY)),
            Canned::at(
                SPEND_LOGS_PATH,
                Canned::json(200, MALFORMED_SPEND_LOGS_BODY),
            ),
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };
    /// The side answers are an upper bound: a rejected key asks for neither.
    pub const USAGE_UNAUTHORIZED: Fixture = Fixture {
        name: "usage_unauthorized",
        script: &[
            Canned::at(KEY_INFO_PATH, Canned::json(401, UNAUTHORIZED_BODY)),
            ACTIVITY_OK,
            SPEND_LOGS_OK,
        ],
        thinking: ThinkingConfig::Off,
        session: None,
    };

    pub fn model() -> Model {
        Model::from_spec(MODEL_SPEC).expect(UNKNOWN_MODEL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ProviderRegistry;

    const BUDGETED_KEY_INFO: &str = r#"{"key":"abc","info":{"key_alias":"me@example.com","spend":2.5,"max_budget":10.0,"budget_reset_at":"2026-09-01T00:00:00Z"}}"#;
    const UNBUDGETED_KEY_INFO: &str =
        r#"{"key":"abc","info":{"spend":0.42,"max_budget":null,"budget_reset_at":null}}"#;

    #[test]
    fn budgeted_key_reports_percentage_and_reset() {
        let resp: KeyInfoResponse = serde_json::from_str(BUDGETED_KEY_INFO).unwrap();
        let limit = UsageLimit::from(resp.info);
        assert_eq!(limit.percentage, Some(25));
        assert_eq!(
            limit.detail.as_deref(),
            Some("$2.50 spent of $10.00 budget")
        );
        assert_eq!(limit.reset_at, Some(1_788_220_800_000));
    }

    #[test]
    fn unbudgeted_key_reports_spend_without_percentage() {
        let resp: KeyInfoResponse = serde_json::from_str(UNBUDGETED_KEY_INFO).unwrap();
        let limit = UsageLimit::from(resp.info);
        assert_eq!(limit.percentage, None);
        assert_eq!(limit.reset_at, None);
        assert_eq!(limit.detail.as_deref(), Some("$0.42 spent"));
    }

    const ACTIVITY_TODAY: &str = r#"{"sum_api_requests":14,"sum_total_tokens":29027}"#;

    #[test]
    fn activity_maps_to_daily_counts_without_percentage() {
        let activity: ActivityResponse = serde_json::from_str(ACTIVITY_TODAY).unwrap();
        let limit = UsageLimit::from(activity);
        assert_eq!(limit.label, "Today");
        assert_eq!(limit.percentage, None);
        assert!(limit.reset_at.is_some());
        assert_eq!(limit.detail.as_deref(), Some("14 requests · 29027 tokens"));
    }

    #[test]
    fn next_utc_midnight_is_following_day_start() {
        let now = "2026-08-25T13:45:10Z".parse::<Timestamp>().unwrap();
        assert_eq!(next_utc_midnight(now), Some(1_787_702_400_000));
    }

    #[test]
    fn root_url_strips_the_version_segment() {
        assert_eq!(
            root_url("https://api.regolo.ai/v1"),
            "https://api.regolo.ai"
        );
        assert_eq!(
            root_url("https://api.regolo.ai/v1/"),
            "https://api.regolo.ai"
        );
        assert_eq!(root_url("https://api.regolo.ai"), "https://api.regolo.ai");
    }

    #[test]
    fn manifest_lists_the_catalogued_default_model() {
        let spec = ProviderRegistry::get(&CONFIG.slug).expect("regolo is a builtin");
        assert!(
            spec.models()
                .iter()
                .any(|m| m.prefixes == ["qwen3-coder-next"])
        );
    }

    const MODEL_GROUPS_FIXTURE: &str = r#"{"data":[
        {
            "model_group": "glm5.2", "mode": "chat",
            "input_cost_per_token": 2e-06, "output_cost_per_token": 5.2e-06,
            "max_input_tokens": 96000.0, "max_output_tokens": 96000.0, "max_tokens": null,
            "supports_reasoning": true, "supports_vision": false
        },
        {
            "model_group": "qwen3-coder-next", "mode": "chat",
            "input_cost_per_token": 5e-07, "output_cost_per_token": 2e-06,
            "max_input_tokens": null, "max_output_tokens": 120000.0, "max_tokens": 240000.0,
            "supports_reasoning": false, "supports_vision": true
        },
        {
            "model_group": "Qwen3-Embedding-8B", "mode": "embedding",
            "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
            "max_input_tokens": null, "max_output_tokens": null, "max_tokens": null,
            "supports_reasoning": false, "supports_vision": false
        }
    ]}"#;

    fn join_fixture(ids: &[&str]) -> Vec<ModelInfo> {
        let groups: ModelGroupInfoResponse = serde_json::from_str(MODEL_GROUPS_FIXTURE).unwrap();
        join_model_info(
            ids.iter().map(|id| (*id).to_string()).collect(),
            groups.data,
        )
    }

    #[test]
    fn group_metadata_maps_to_pricing_context_and_capabilities() {
        let mut infos = join_fixture(&["glm5.2"]);
        assert_eq!(infos.len(), 1);
        let info = infos.pop().unwrap();
        assert_eq!(info.id, "glm5.2");
        assert_eq!(info.context_window, Some(96_000));
        assert_eq!(info.max_output_tokens, Some(96_000));
        let pricing = info.pricing.unwrap();
        assert_eq!(pricing.input, 2.0);
        assert_eq!(pricing.output, 5.2);
        assert_eq!(info.supports_thinking, Some(true));
        assert_eq!(info.supports_vision, Some(false));
    }

    #[test]
    fn missing_input_window_falls_back_to_max_tokens() {
        let info = join_fixture(&["qwen3-coder-next"]).pop().unwrap();
        assert_eq!(info.context_window, Some(240_000));
        assert_eq!(info.supports_thinking, Some(false));
    }

    #[test]
    fn non_chat_groups_and_ids_without_group_are_dropped() {
        let infos = join_fixture(&["Qwen3-Embedding-8B", "glm5.2", "brand-new-model"]);
        let ids: Vec<&str> = infos.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["glm5.2"]);
    }

    const SPEND_LOGS_FIXTURE: &str = r#"{
      "data": [
        {
          "request_id": "h1", "api_key": "k",
          "model_group": "qwen3-coder-next", "key_alias": "a",
          "startTime": "2026-08-25T09:00:00+00:00", "endTime": "2026-08-25T10:00:00+00:00",
          "completionStartTime": null,
          "api_requests": 4, "total_tokens": 20000,
          "prompt_tokens": 15000, "completion_tokens": 5000,
          "spend": 0.010000, "request_duration_ms": 3600000
        },
        {
          "request_id": "h2", "api_key": "k",
          "model_group": "qwen3-coder-next", "key_alias": "a",
          "startTime": "2026-08-25T10:00:00+00:00", "endTime": "2026-08-25T11:00:00+00:00",
          "completionStartTime": null,
          "api_requests": 3, "total_tokens": 8000,
          "prompt_tokens": 5000, "completion_tokens": 3000,
          "spend": 0.004200, "request_duration_ms": 3600000
        },
        {
          "request_id": "h3", "api_key": "k",
          "model_group": "glm5.2", "key_alias": "a",
          "startTime": "2026-08-25T09:00:00+00:00", "endTime": "2026-08-25T10:00:00+00:00",
          "completionStartTime": null,
          "api_requests": 2, "total_tokens": 360,
          "prompt_tokens": 200, "completion_tokens": 160,
          "spend": 0.000900, "request_duration_ms": 3600000
        }
      ]
    }"#;

    #[test]
    fn spend_logs_aggregate_by_model_and_rank_by_spend() {
        let resp: SpendLogsResponse = serde_json::from_str(SPEND_LOGS_FIXTURE).unwrap();
        let rows = resp.into_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].model, "qwen3-coder-next");
        assert_eq!(rows[0].input_tokens, 20_000);
        assert_eq!(rows[0].output_tokens, 8_000);
        assert_eq!(rows[0].total_tokens, 28_000);
        assert_eq!(rows[0].spend_microdollars, 14_200);
        assert_eq!(rows[1].model, "glm5.2");
        assert_eq!(rows[1].spend_microdollars, 900);
    }
}

/// Regolo as [`decl`] plus [`hooks`] put it on the wire, one recorded exchange
/// at a time.
#[cfg(test)]
mod replay_tests {
    use test_case::test_case;

    use crate::providers::replay::{self, Fixture};

    use super::SLUG;
    use super::fixtures;

    #[test_case(&fixtures::SUCCESS ; "success")]
    #[test_case(&fixtures::THINKING_OFF ; "thinking_off")]
    #[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
    #[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
    #[test_case(&replay::RATE_LIMITED ; "rate_limited")]
    #[test_case(&replay::SERVER_ERROR ; "server_error")]
    #[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
    #[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
    #[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
    fn the_declaration_replays_the_recorded_exchange(fixture: &Fixture) {
        replay::declared(replay::rust_authoring, SLUG, fixture, &fixtures::model());
    }

    /// The golden pins both requests as well as the rows: the group call is
    /// made after a good `/models` answer only, and its failure keeps the ids.
    #[test_case(&fixtures::MODELS ; "models")]
    #[test_case(&fixtures::MODELS_WITHOUT_GROUPS ; "models_without_groups")]
    #[test_case(&fixtures::MODELS_MALFORMED_GROUPS ; "models_malformed_groups")]
    #[test_case(&fixtures::MODELS_UNAUTHORIZED ; "models_unauthorized")]
    fn the_declaration_lists_the_recorded_catalogue(fixture: &Fixture) {
        replay::declared_models(replay::rust_authoring, SLUG, fixture);
    }

    #[test_case(&fixtures::USAGE ; "usage")]
    #[test_case(&fixtures::USAGE_NO_BUDGET ; "usage_no_budget")]
    #[test_case(&fixtures::USAGE_OVER_BUDGET ; "usage_over_budget")]
    #[test_case(&fixtures::USAGE_SIDE_CALLS_FAIL ; "usage_side_calls_fail")]
    #[test_case(&fixtures::USAGE_SIDE_CALLS_MALFORMED ; "usage_side_calls_malformed")]
    #[test_case(&fixtures::USAGE_UNAUTHORIZED ; "usage_unauthorized")]
    fn the_declaration_reads_the_recorded_usage(fixture: &Fixture) {
        replay::declared_usage(replay::rust_authoring, SLUG, fixture);
    }
}
