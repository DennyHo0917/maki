//! Golden replay: one recorded exchange, one provider, one artifact on disk.
//!
//! Four things about a provider are worth comparing and this compares all
//! four: the request bytes it put on the wire, the [`ProviderEvent`]s it
//! emitted in order, the error it failed with, and the usage it came back
//! with. Each case lands in a golden file, so the suite keeps its teeth once
//! the implementation it was first written against is deleted. A differential
//! test dies with either of its two sides, a golden does not.
//!
//! Every entry point takes the authoring to replay, because an artifact that
//! only ever saw the declaration maki *doesn't* ship proves the wrong thing:
//! the bundled Lua decl outranks the Rust one at every real startup. This
//! crate can stage its own declarations ([`rust_authoring`]) and `maki-lua`
//! boots the plugin host and registers the bundled decl over the top, both
//! against these same files.
//!
//! Regenerate with `UPDATE_GOLDENS=1 cargo nextest run -p maki-providers`.
//! A *missing* golden always fails, because a suite that records whatever it
//! sees on its first run has asserted nothing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use maki_config::providers::base_url_env_var;
use serde_json::{Value, json};
use tempfile::TempDir;

use crate::model::Model;
use crate::provider::Provider;
use crate::spec::ProviderRegistry;
use crate::test_support::{Canned, Recorded, Requests, serve};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse, ThinkingConfig};

use super::{Timeouts, plugin};

const GOLDEN_DIR: &str = "tests/goldens";
const UPDATE_ENV: &str = "UPDATE_GOLDENS";
const UPDATE_ON: &str = "1";
const HOST_HEADER: &str = "host";
const USER_AGENT_HEADER: &str = "user-agent";
const VOLATILE_VALUE: &str = "<volatile>";
const REQUESTS_KEY: &str = "requests";
const BODY_KEY: &str = "body";

const PROMPT: &str = "read a.txt";
const SYSTEM: &str = "You are a replay fixture.";
const TOOL_NAME: &str = "read";
const TOOL_DESCRIPTION: &str = "Read a file";

const API_KEY: &str = "sk-replay";
const HOME_VARS: &[&str] = &[
    "HOME",
    "XDG_STATE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
];

const UNAUTHORIZED_BODY: &str = r#"{"error":{"message":"invalid api key"}}"#;
const RATE_LIMITED_BODY: &str = r#"{"error":{"message":"too many requests"}}"#;
const SERVER_ERROR_BODY: &str = r#"{"error":{"message":"internal error"}}"#;
const RETRY_AFTER_HEADERS: &[(&str, &str)] =
    &[("content-type", "application/json"), ("retry-after", "7")];

const NO_GOLDEN_DIR: &str = "the golden directory has no parent";
const WRITE_FAILED: &str = "the golden could not be recorded";
const BAD_GOLDEN: &str = "the golden on disk is not json";
const NOT_AN_OBJECT: &str = "an observation is always a json object";
const NO_REQUESTS: &str = "every observation records the requests it sent";
const TEMPDIR_FAILED: &str = "no temporary state directory";
const NOT_A_BUILTIN: &str = "a replayed slug is a builtin";
const CREATE_FAILED: &str = "the provider could not be built";

/// One replayed exchange: what the server answers with, and what the request
/// asks for beyond the fixed prompt.
pub struct Fixture {
    pub name: &'static str,
    /// An upper bound on the requests a run may send, not a promise that it
    /// sends them all: a provider that retries internally draws a second
    /// entry, and one that does not leaves it unserved.
    pub script: &'static [Canned],
    pub thinking: ThinkingConfig,
}

// The failures below look the same whichever provider hits them, so they are
// written once here and every port replays the ones it answers. Each provider
// still gets its own golden, since the name of a fixture is the name of its
// file inside the provider's directory.

/// A rejected key. The second answer is one neither side is expected to ask
/// for: a provider that replays the rejected key is recorded as a second
/// request instead of parking on an `accept` that never returns.
pub const UNAUTHORIZED: Fixture = Fixture {
    name: "unauthorized",
    script: &[
        Canned::json(401, UNAUTHORIZED_BODY),
        Canned::json(401, UNAUTHORIZED_BODY),
    ],
    thinking: ThinkingConfig::Off,
};

pub const RATE_LIMITED: Fixture = Fixture {
    name: "rate_limited",
    script: &[Canned::json(429, RATE_LIMITED_BODY)],
    thinking: ThinkingConfig::Off,
};

/// The same 429 with the header that tells us how long to wait, which is the
/// one thing downstream backoff reads off a rate limit.
pub const SLOW_DOWN: Fixture = Fixture {
    name: "rate_limited_with_retry_after",
    script: &[Canned {
        status: 429,
        headers: RETRY_AFTER_HEADERS,
        body: RATE_LIMITED_BODY,
    }],
    thinking: ThinkingConfig::Off,
};

pub const SERVER_ERROR: Fixture = Fixture {
    name: "server_error",
    script: &[Canned::json(500, SERVER_ERROR_BODY)],
    thinking: ThinkingConfig::Off,
};

/// One unparseable frame between two good ones: the bad frame is skipped and
/// the turn still ends, rather than the whole stream failing.
pub const MALFORMED_SSE: Fixture = Fixture {
    name: "malformed_sse",
    script: &[Canned::sse(
        r#"data: {"choices": [ this is not json

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: [DONE]

"#,
    )],
    thinking: ThinkingConfig::Off,
};

/// An error frame on a 200, carrying a tag but no message. The substituted
/// message is a real bug fix (`EMPTY_SSE_ERROR_MESSAGE`): without it the turn
/// ended with an empty assistant message and no retry.
pub const EMPTY_SSE_ERROR: Fixture = Fixture {
    name: "empty_sse_error_frame",
    script: &[Canned::sse(
        r#"data: {"error":{"type":"server_error","message":""}}

"#,
    )],
    thinking: ThinkingConfig::Off,
};

/// Ends mid-frame, with no `finish_reason` and no `[DONE]`.
pub const TRUNCATED_STREAM: Fixture = Fixture {
    name: "truncated_stream",
    script: &[Canned::sse(
        r#"data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"con"#,
    )],
    thinking: ThinkingConfig::Off,
};

/// The question every fixture asks, so two providers driven by this harness
/// are never answering different ones.
pub fn tools() -> Value {
    json!([{
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
        },
    }])
}

fn turn() -> Vec<Message> {
    vec![Message::user(PROMPT.to_owned())]
}

/// The stage step that leaves maki's own declarations in place: `begin_load`
/// has already registered them, so a load with no plugin over the top serves
/// exactly those. It is the same declaration `impl = "rust"` picks at runtime.
pub fn rust_authoring() {}

/// Replays `fixture` through the declaration `stage` left serving `slug` and
/// pins everything that came back.
pub fn declared<T>(stage: impl FnOnce() -> T, slug: &str, fixture: &Fixture, model: &Model) {
    declared_with(stage, slug, fixture, model, &turn(), &tools());
}

/// [`declared`] for a provider whose body work reads the history or the tool
/// list. What it does to an assistant turn is invisible against the lone user
/// message [`declared`] sends, and what it does only when tools are present is
/// invisible when they always are.
pub fn declared_with<T>(
    stage: impl FnOnce() -> T,
    slug: &str,
    fixture: &Fixture,
    model: &Model,
    messages: &[Message],
    tools: &Value,
) {
    let (provider, requests, _world) = build(slug, fixture, stage);

    let (tx, rx) = flume::unbounded();
    let result = smol::block_on(provider.stream_message(
        model,
        messages,
        SYSTEM,
        tools,
        &tx,
        RequestOptions {
            thinking: fixture.thinking,
            fast: false,
        },
        None,
    ));
    drop(tx);
    let events: Vec<ProviderEvent> = rx.drain().collect();

    assert_golden(
        slug,
        fixture,
        &json!({
            REQUESTS_KEY: recorded(&requests),
            "events": events,
            "outcome": outcome(&result),
        }),
    );
}

/// The same recorded exchange for the other endpoint a provider answers on.
/// Kept apart from [`declared`] rather than folded into the `Fixture`: a usage
/// call sends no messages, emits no events and has no thinking mode, so a
/// shared entry point would carry three fields it never reads.
pub fn declared_usage<T>(stage: impl FnOnce() -> T, slug: &str, fixture: &Fixture) {
    let (provider, requests, _world) = build(slug, fixture, stage);
    let result = smol::block_on(provider.fetch_usage());

    assert_golden(
        slug,
        fixture,
        &json!({
            REQUESTS_KEY: recorded(&requests),
            "outcome": match &result {
                Ok(usage) => json!({ "usage": usage }),
                Err(e) => failure(e),
            },
        }),
    );
}

/// Stands up the whole world one exchange needs: a throwaway home with the
/// key the slug reads, the recorded server, and `slug` built through the
/// startup path. `stage` registers inside the load window the way a plugin
/// load does, and `create` resolves the inherited `api_key_env` into a key
/// pool right away, so the claim on the built-in slug is exercised instead of
/// assumed.
///
/// Staging and isolation come back as one guard the caller has to hold: the
/// temporary tree is read while the request is built, and a hook whose plugin
/// host has died answers nothing. In that order, so a host that writes on its
/// way out still finds the home it was told to use.
///
/// Every base directory moves, so no run touches this machine's credentials,
/// `providers.toml` or saved origins. The registry, the environment and the
/// credential store are all process-global, and `cargo nextest` gives each
/// test its own process, which is what keeps one fixture's key and origin out
/// of the next one's.
///
/// Loopback is published through `<SLUG>_BASE_URL` because that is the only
/// rung of the precedence a test can reach. The declaration's own `base_url`
/// is the codec's *last* resort, so writing loopback there would mean
/// registering a declaration that is not the one being ported, and
/// `auth.base_url` is only ever written by an auth hook.
fn build<T>(
    slug: &str,
    fixture: &Fixture,
    stage: impl FnOnce() -> T,
) -> (Box<dyn Provider>, Requests, (T, TempDir)) {
    let home = TempDir::new().expect(TEMPDIR_FAILED);
    for var in HOME_VARS {
        unsafe { std::env::set_var(var, home.path()) };
    }
    let key_env = ProviderRegistry::get(slug)
        .expect(NOT_A_BUILTIN)
        .api_key_env;
    unsafe { std::env::set_var(key_env, API_KEY) };

    let (base_url, requests) = serve(fixture.script);
    unsafe { std::env::set_var(base_url_env_var(slug), base_url) };
    plugin::begin_load();
    let staged = stage();
    plugin::commit_load();
    let provider = plugin::create(slug, Timeouts::default()).expect(CREATE_FAILED);
    (provider, requests, (staged, home))
}

fn recorded(requests: &Requests) -> Value {
    Value::Array(
        requests
            .lock()
            .unwrap()
            .iter()
            .map(request_value)
            .collect::<Vec<_>>(),
    )
}

/// The request as the observation keeps it: method, path, the header set and
/// the body verbatim.
///
/// The body stays the string the codec wrote rather than a parsed `Value`, so
/// nothing between here and the golden can quietly repair malformed JSON. What
/// reaches disk is canonicalised instead, see [`canonical_observation`].
fn request_value(recorded: &Recorded) -> Value {
    let headers: BTreeMap<&str, &str> = recorded
        .headers
        .iter()
        .map(|(name, value)| {
            let value = if is_volatile(name) {
                VOLATILE_VALUE
            } else {
                value.as_str()
            };
            (name.as_str(), value)
        })
        .collect();
    json!({
        "method": recorded.method,
        "path": recorded.path,
        "headers": headers,
        "body": String::from_utf8_lossy(&recorded.body),
    })
}

/// `host` carries the loopback port the kernel happened to hand out and
/// `user-agent` carries the build's git hash, so for those two the comparison
/// is that the header was sent at all.
fn is_volatile(name: &str) -> bool {
    name == HOST_HEADER || name == USER_AGENT_HEADER
}

fn outcome(result: &Result<StreamResponse, AgentError>) -> Value {
    match result {
        Ok(response) => json!({
            "message": response.message,
            "usage": response.usage,
            "stop_reason": response.stop_reason,
            // The one number a session's gauge takes from a response, and the
            // separate usage fields above do not show which of them it sums.
            "context_size": response.usage.total_input(),
        }),
        Err(e) => failure(e),
    }
}

/// `AgentError` cannot be `PartialEq`, so [`AgentError::projection`] is the
/// comparison. `error.rs` carries a test proving two equal projections agree
/// on every observable predicate, which a hand-rolled `(discriminant, status,
/// message)` tuple would not. It is written through `Debug` because the
/// projection is a structural enum over `PartialEq` fields, so its debug form
/// separates exactly what `==` does.
///
/// The rendered message rides along because the projection reads the message
/// only through those predicates, and some behaviour lives nowhere else: a
/// provider that substitutes a message for an error frame that carried none
/// projects identically to one that does not.
fn failure(e: &AgentError) -> Value {
    json!({ "error": format!("{:?}", e.projection()), "message": e.to_string() })
}

fn golden_path(provider: &str, case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(GOLDEN_DIR)
        .join(provider)
        .join(format!("{case}.json"))
}

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).expect(NOT_AN_OBJECT)
}

/// The observation as an artifact on disk: the same file whatever else was in
/// the build.
///
/// `serde_json::Map` is an `IndexMap` whenever anything in the build graph
/// turns on `preserve_order`. `agent-client-protocol-schema` does, so a
/// workspace build has it and `-p maki-providers` does not, and cargo unifies
/// features across the graph rather than per crate. Key order would then be a
/// property of the `-p` flags, both in the golden itself and inside the
/// recorded body, which is a JSON document carried as a string. Sorting every
/// object at every depth, and the body's after parsing it, leaves one
/// canonical form for both builds to agree on.
fn canonical_observation(observed: &Value) -> Value {
    let mut canonical = sorted(observed);
    let requests = canonical
        .get_mut(REQUESTS_KEY)
        .and_then(Value::as_array_mut)
        .expect(NO_REQUESTS);
    for request in requests {
        let Some(body) = request.get_mut(BODY_KEY) else {
            continue;
        };
        // A body that is not JSON, or is empty, keeps the raw string: there is
        // no key order in it to leak.
        if let Some(parsed) = body
            .as_str()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        {
            *body = Value::String(serde_json::to_string(&sorted(&parsed)).expect(NOT_AN_OBJECT));
        }
    }
    canonical
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), sorted(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        scalar => scalar.clone(),
    }
}

/// Compares one observation against the artifact on disk, or records it when
/// `UPDATE_GOLDENS=1`. Both sides are canonicalised, so this asserts what the
/// provider did and never which crates the test binary was linked against.
fn assert_golden(provider: &str, fixture: &Fixture, observed: &Value) {
    let path = golden_path(provider, fixture.name);
    let observed = canonical_observation(observed);
    if std::env::var(UPDATE_ENV).is_ok_and(|value| value == UPDATE_ON) {
        std::fs::create_dir_all(path.parent().expect(NO_GOLDEN_DIR)).expect(WRITE_FAILED);
        std::fs::write(&path, pretty(&observed) + "\n").expect(WRITE_FAILED);
        return;
    }
    let recorded = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "no golden at {}: {e}\nrecord it with {UPDATE_ENV}={UPDATE_ON}",
            path.display()
        )
    });
    let expected =
        canonical_observation(&serde_json::from_str::<Value>(&recorded).expect(BAD_GOLDEN));
    assert!(
        observed == expected,
        "{} drifted from {}\n--- recorded\n{}\n--- observed\n{}",
        fixture.name,
        path.display(),
        pretty(&expected),
        pretty(&observed)
    );
}
