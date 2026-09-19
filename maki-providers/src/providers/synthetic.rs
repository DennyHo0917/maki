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

/// This provider as a declaration, which is all of it: the openai codec spells
/// Synthetic's whole wire, so there is not one hook here.
///
/// Nothing the [`SPEC`] row already holds is restated. Claiming the built-in
/// slug inherits the display name, the key env var, the family, the fallback
/// limits and the curated model table, and restating any of them is a
/// registration error rather than a second home for the same fact.
///
/// Staged at every load from [`crate::providers::plugin`]'s list of the
/// declarations maki authors, and outranked there by the bundled `synthetic`
/// Lua plugin, which restates this same declaration on the authoring surface a
/// third-party plugin uses.
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

/// Synthetic as [`decl`] puts it on the wire, one recorded exchange at a time.
///
/// Every case was recorded while the bespoke `impl Provider` this module used
/// to hold was still here, running both against the same artifact. The impl is
/// gone and the artifacts are not: each fixture is still pinned to the bytes
/// and the events that provider produced on the day it was ported.
#[cfg(test)]
mod tests {
    use tempfile::TempDir;
    use test_case::test_case;

    use crate::model::Model;
    use crate::providers::replay::{self, Fixture};
    use crate::providers::{Timeouts, plugin};
    use crate::test_support::Canned;
    use crate::{Effort, ThinkingConfig};

    use super::{ENV_VAR, SLUG};

    const MODEL_SPEC: &str = "synthetic/hf:moonshotai/Kimi-K2.5";
    const BASE_URL_ENV: &str = "SYNTHETIC_BASE_URL";
    const API_KEY: &str = "sk-replay";
    /// Reaches the wire as `reasoning_effort`, which is the one thing
    /// [`super::decl`]'s `thinking_dialect` is there to do.
    const EFFORT: Effort = Effort::High;

    const TEMPDIR_FAILED: &str = "no temporary state directory";
    const UNKNOWN_MODEL: &str = "the curated table has no such model";
    const CREATE_FAILED: &str = "the provider could not be built";

    const UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"invalid api key"}}"#;
    const RATE_LIMITED_BODY: &str = r#"{"error":{"message":"too many requests"}}"#;
    const SERVER_ERROR_BODY: &str = r#"{"error":{"message":"internal error"}}"#;
    const RETRY_AFTER_HEADERS: &[(&str, &str)] =
        &[("content-type", "application/json"), ("retry-after", "7")];

    const SUCCESS_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"reasoning_content":"weighing the options"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}

data: [DONE]

"#;

    /// One unparseable frame between two good ones: the bad frame is skipped
    /// and the turn still ends, rather than the whole stream failing.
    const MALFORMED_TRANSCRIPT: &str = r#"data: {"choices": [ this is not json

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: [DONE]

"#;

    /// An error frame on a 200, carrying a tag but no message. The substituted
    /// message is a real bug fix (`EMPTY_SSE_ERROR_MESSAGE`): without it the
    /// turn ended with an empty assistant message and no retry.
    const EMPTY_ERROR_TRANSCRIPT: &str = r#"data: {"error":{"type":"server_error","message":""}}

"#;

    /// Ends mid-frame, with no `finish_reason` and no `[DONE]`.
    const TRUNCATED_TRANSCRIPT: &str = r#"data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"con"#;

    const SUCCESS_SCRIPT: &[Canned] = &[Canned::sse(SUCCESS_TRANSCRIPT)];
    const MALFORMED_SCRIPT: &[Canned] = &[Canned::sse(MALFORMED_TRANSCRIPT)];
    const EMPTY_ERROR_SCRIPT: &[Canned] = &[Canned::sse(EMPTY_ERROR_TRANSCRIPT)];
    const TRUNCATED_SCRIPT: &[Canned] = &[Canned::sse(TRUNCATED_TRANSCRIPT)];
    /// A second answer neither side is expected to ask for: a run that replays
    /// the rejected key is recorded as a second request rather than parking on
    /// an `accept` that never returns.
    const UNAUTHORIZED_SCRIPT: &[Canned] = &[
        Canned::json(401, UNAUTHORIZED_BODY),
        Canned::json(401, UNAUTHORIZED_BODY),
    ];
    const RATE_LIMITED_SCRIPT: &[Canned] = &[Canned::json(429, RATE_LIMITED_BODY)];
    const SLOW_DOWN_SCRIPT: &[Canned] = &[Canned {
        status: 429,
        headers: RETRY_AFTER_HEADERS,
        body: RATE_LIMITED_BODY,
    }];
    const SERVER_ERROR_SCRIPT: &[Canned] = &[Canned::json(500, SERVER_ERROR_BODY)];

    const SUCCESS: Fixture = Fixture {
        name: "success",
        script: SUCCESS_SCRIPT,
        thinking: ThinkingConfig::Effort(EFFORT),
    };
    const UNAUTHORIZED: Fixture = Fixture {
        name: "unauthorized",
        script: UNAUTHORIZED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const SLOW_DOWN: Fixture = Fixture {
        name: "rate_limited_with_retry_after",
        script: SLOW_DOWN_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const RATE_LIMITED: Fixture = Fixture {
        name: "rate_limited",
        script: RATE_LIMITED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const SERVER_ERROR: Fixture = Fixture {
        name: "server_error",
        script: SERVER_ERROR_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const MALFORMED: Fixture = Fixture {
        name: "malformed_sse",
        script: MALFORMED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const EMPTY_ERROR: Fixture = Fixture {
        name: "empty_sse_error_frame",
        script: EMPTY_ERROR_SCRIPT,
        thinking: ThinkingConfig::Off,
    };
    const TRUNCATED: Fixture = Fixture {
        name: "truncated_stream",
        script: TRUNCATED_SCRIPT,
        thinking: ThinkingConfig::Off,
    };

    /// Points every base directory at a throwaway tree and publishes the key
    /// both sides resolve, so neither reads this machine's credentials,
    /// `providers.toml` or saved origins.
    fn isolated() -> TempDir {
        let dir = TempDir::new().expect(TEMPDIR_FAILED);
        for var in [
            "HOME",
            "XDG_STATE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
        ] {
            unsafe { std::env::set_var(var, dir.path()) };
        }
        unsafe { std::env::set_var(ENV_VAR, API_KEY) };
        dir
    }

    /// The one door onto the recorded server that both sides go through.
    ///
    /// The base-url precedence is `auth.base_url` > `<SLUG>_BASE_URL` /
    /// `providers.toml` > the static default, and only the middle rung is open
    /// to both: the declaration's static `base_url` is the codec's *last*
    /// resort, so writing loopback there would mean registering a declaration
    /// that is not the one being ported, and `auth.base_url` is only ever
    /// written by an auth hook, which a zero-hook declaration has none of.
    fn point_at(base_url: &str) {
        unsafe { std::env::set_var(BASE_URL_ENV, base_url) };
    }

    /// The startup path rather than a hand-built registration: a load with no
    /// plugin in it stages exactly the declarations maki authors, and
    /// `create` resolves the *inherited* `api_key_env` into a key pool
    /// eagerly, so the claim on the built-in slug is exercised instead of
    /// assumed.
    fn register_decl() {
        plugin::begin_load();
        plugin::commit_load();
    }

    /// The registry, the environment and the credential store are all
    /// process-global; `cargo nextest` gives each case its own process, which
    /// is what keeps one fixture's origin out of the next one's.
    #[test_case(&SUCCESS ; "success")]
    #[test_case(&UNAUTHORIZED ; "unauthorized")]
    #[test_case(&SLOW_DOWN ; "rate_limited_with_retry_after")]
    #[test_case(&RATE_LIMITED ; "rate_limited")]
    #[test_case(&SERVER_ERROR ; "server_error")]
    #[test_case(&MALFORMED ; "malformed_sse")]
    #[test_case(&EMPTY_ERROR ; "empty_sse_error_frame")]
    #[test_case(&TRUNCATED ; "truncated_stream")]
    fn the_declaration_replays_the_recorded_exchange(fixture: &Fixture) {
        let _isolated = isolated();
        let model = Model::from_spec(MODEL_SPEC).expect(UNKNOWN_MODEL);
        register_decl();

        let declared = replay::run(fixture, &model, |base_url| {
            point_at(base_url);
            plugin::create(SLUG, Timeouts::default()).expect(CREATE_FAILED)
        });
        replay::assert_golden(SLUG, fixture, &declared);
    }
}
