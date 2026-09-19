use maki_config::providers::Protocol;

use crate::dialect;
use crate::model::ModelFamily;
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, ProviderSpec};

use super::plugin::ProviderDecl;

const SLUG: &str = "synthetic";
const DISPLAY_NAME: &str = "Synthetic";
const ENV_VAR: &str = "SYNTHETIC_API_KEY";
const BASE_URL: &str = "https://api.synthetic.new/openai/v1";
const DEFAULT_MODEL: &str = "synthetic/hf:moonshotai/Kimi-K2.5";
const LOGIN_URL: &str = "https://synthetic.new";
const MAX_TOKENS_FIELD: &str = "max_completion_tokens";
const FEATURES: &str = "Reasoning effort support (low/medium/high), open-weight models";
const NET_HOST: &str = "api.synthetic.new";

pub(crate) const SPEC: ProviderSpec = ProviderSpec {
    slug: SLUG,
    display_name: DISPLAY_NAME,
    api_key_env: ENV_VAR,
    family: ModelFamily::Synthetic,
    supports_thinking: true,
    accepts_arbitrary_models: false,
    fallback_max_output: Some(32_000),
    fallback_context_window: 128_000,
    models_toml: include_str!("../../models/synthetic.toml"),
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

/// Synthetic as a declaration, which is all of it: the openai codec spells the
/// whole wire, so there is not one hook here.
///
/// Only what the codec cannot guess is stated. Claiming a built-in slug
/// inherits the whole [`SPEC`] row, and restating any of it is a registration
/// error rather than a second home for the same fact.
///
/// The bundled `synthetic` Lua plugin says all of this again on the surface a
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
        max_tokens_field: Some(MAX_TOKENS_FIELD.to_owned()),
        include_stream_usage: Some(false),
        thinking_dialect: Some(&dialect::STANDARD),
        models: Vec::new(),
        net_hosts: vec![NET_HOST.to_owned()],
    }
}

/// The recorded cases, kept out of the test module so both authorings replay
/// the same list: [`decl`] above, and the bundled `synthetic` Lua plugin that
/// outranks it at every real startup.
///
/// Every case was recorded while the bespoke `impl Provider` this module used
/// to hold was still here, with both sides run against the same artifact. The
/// impl is gone and the artifacts are not, so each fixture still pins the bytes
/// and the events that provider produced on the day it was ported.
#[cfg(any(test, feature = "test-support"))]
pub mod fixtures {
    use crate::model::Model;
    use crate::providers::replay::Fixture;
    use crate::test_support::Canned;
    use crate::{Effort, ThinkingConfig};

    pub const MODEL_SPEC: &str = "synthetic/hf:moonshotai/Kimi-K2.5";
    /// Reaches the wire as `reasoning_effort`, which is the one thing
    /// [`super::decl`]'s `thinking_dialect` is there to do.
    const EFFORT: Effort = Effort::High;
    const UNKNOWN_MODEL: &str = "the curated table has no such model";

    pub fn model() -> Model {
        Model::from_spec(MODEL_SPEC).expect(UNKNOWN_MODEL)
    }

    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"reasoning_content":"weighing the options"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}

data: [DONE]

"#;

    pub const SUCCESS: Fixture = Fixture {
        name: "success",
        script: &[Canned::sse(SUCCESS_TRANSCRIPT)],
        thinking: ThinkingConfig::Effort(EFFORT),
    };
}

/// Synthetic as [`decl`] puts it on the wire, one recorded exchange at a time.
#[cfg(test)]
mod tests {
    use test_case::test_case;

    use crate::providers::replay::{self, Fixture};

    use super::SLUG;
    use super::fixtures;

    #[test_case(&fixtures::SUCCESS ; "success")]
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
}
