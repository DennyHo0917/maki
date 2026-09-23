//! The bundled `mistral` plugin against the exchanges the Rust authoring is
//! pinned to.

use maki_providers::mistral_fixtures as mistral;
use maki_providers::replay::{self, Fixture};
use test_case::test_case;

use super::{MISTRAL, bundled};

#[test_case(&mistral::SUCCESS ; "success")]
#[test_case(&mistral::THINKING_OFF ; "thinking_off")]
#[test_case(&mistral::IN_SESSION ; "in_session")]
#[test_case(&replay::UNAUTHORIZED ; "unauthorized")]
#[test_case(&replay::SLOW_DOWN ; "rate_limited_with_retry_after")]
#[test_case(&replay::RATE_LIMITED ; "rate_limited")]
#[test_case(&replay::SERVER_ERROR ; "server_error")]
#[test_case(&replay::MALFORMED_SSE ; "malformed_sse")]
#[test_case(&replay::EMPTY_SSE_ERROR ; "empty_sse_error_frame")]
#[test_case(&replay::TRUNCATED_STREAM ; "truncated_stream")]
fn the_bundled_mistral_plugin_replays_the_recorded_exchange(fixture: &Fixture) {
    replay::declared(bundled(MISTRAL), MISTRAL, fixture, &mistral::model());
}

/// The assistant-turn rewrite, which needs a history to act on.
#[test]
fn the_bundled_mistral_plugin_rewrites_the_same_turns() {
    replay::declared_with(
        bundled(MISTRAL),
        MISTRAL,
        &mistral::HISTORY,
        &mistral::model(),
        &mistral::history(),
        &replay::tools(),
    );
}

/// The catalogue goes through `maki.provider_parse`, so every number shape the
/// fixture feeds it has to be rejected the way serde_json rejects it, and a
/// refused listing fails with the Rust authoring's error.
#[test_case(&mistral::MODELS ; "models")]
#[test_case(&mistral::MODELS_UNAUTHORIZED ; "models_unauthorized")]
fn the_bundled_mistral_plugin_lists_the_recorded_models(fixture: &Fixture) {
    replay::declared_models(bundled(MISTRAL), MISTRAL, fixture);
}
