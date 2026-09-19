//! Golden replay: one recorded exchange, one provider, one artifact on disk.
//!
//! A provider is worth comparing on four things, and this compares all four:
//! the request bytes it put on the wire, the [`ProviderEvent`]s it emitted in
//! order, the error it failed with, and the usage it came back with. Every
//! case is recorded as a golden file, so the suite keeps its teeth when the
//! implementation it was originally written to compare against is deleted --
//! a differential test dies with either of its two sides, a golden does not.
//!
//! Regenerate with `UPDATE_GOLDENS=1 cargo nextest run -p maki-providers`.
//! A *missing* golden always fails: a suite that records whatever it sees the
//! first time it runs has asserted nothing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::model::{Model, TokenUsage};
use crate::provider::Provider;
use crate::test_support::{Canned, Recorded, serve};
use crate::tokens::ContextGauge;
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse, ThinkingConfig};

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

const NO_GOLDEN_DIR: &str = "the golden directory has no parent";
const WRITE_FAILED: &str = "the golden could not be recorded";
const BAD_GOLDEN: &str = "the golden on disk is not json";
const NOT_AN_OBJECT: &str = "an observation is always a json object";

/// One replayed exchange: what the server answers with, and what the request
/// asks for beyond the fixed prompt.
pub(crate) struct Fixture {
    pub name: &'static str,
    /// An upper bound on the requests a run may send, not a promise that it
    /// sends them all: a provider that retries internally draws a second
    /// entry, and one that does not leaves it unserved.
    pub script: &'static [Canned],
    pub thinking: ThinkingConfig,
}

/// The question every fixture asks, so two providers driven by this harness
/// are never answering different ones.
fn tools() -> Value {
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

/// Runs `fixture` against the provider `build` returns, pointed at a freshly
/// bound recorded server.
///
/// `build` takes the origin rather than the harness publishing it, because
/// which of the three base-url mechanisms reaches a given provider is the
/// caller's problem: a provider reads its origin once, at construction, so
/// the closure runs after the port is known.
pub(crate) fn run(
    fixture: &Fixture,
    model: &Model,
    build: impl FnOnce(&str) -> Box<dyn Provider>,
) -> Value {
    let (base_url, requests) = serve(fixture.script);
    let provider = build(&base_url);

    let messages = [Message::user(PROMPT.to_owned())];
    let (tx, rx) = flume::unbounded();
    let result = smol::block_on(provider.stream_message(
        model,
        &messages,
        SYSTEM,
        &tools(),
        &tx,
        RequestOptions {
            thinking: fixture.thinking,
            fast: false,
        },
        None,
    ));
    drop(tx);
    let events: Vec<ProviderEvent> = rx.drain().collect();

    json!({
        REQUESTS_KEY: requests
            .lock()
            .unwrap()
            .iter()
            .map(request_value)
            .collect::<Vec<_>>(),
        "events": events,
        "outcome": outcome(&result),
    })
}

/// The request as the observation keeps it: method, path, the header set and
/// the body verbatim.
///
/// The body stays a string rather than a parsed `Value` so the differential
/// half compares the bytes a codec actually wrote, key order included. What
/// reaches disk is canonicalised instead; see [`canonical_observation`].
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
            "context_size": gauge_size(&response.usage),
        }),
        // `AgentError` cannot be `PartialEq`, so [`AgentError::projection`] is
        // the comparison -- `error.rs` carries a test proving two equal
        // projections agree on every observable predicate, which a
        // hand-rolled `(discriminant, status, message)` tuple would not.
        // Written through `Debug` because the projection is a structural enum
        // over `PartialEq` fields, so its debug form separates exactly what
        // `==` does.
        //
        // The rendered message rides along because the projection reads the
        // message only through those predicates, and some behaviour lives
        // nowhere else: a provider that substitutes a message for an error
        // frame that carried none projects identically to one that does not.
        Err(e) => json!({ "error": format!("{:?}", e.projection()), "message": e.to_string() }),
    }
}

/// What a session's gauge learns from this response, which is the whole of
/// what a `StreamResponse`'s usage does downstream.
fn gauge_size(usage: &TokenUsage) -> u32 {
    let mut gauge = ContextGauge::default();
    gauge.record(usage.total_input());
    gauge.size()
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
/// turns on `preserve_order` -- `agent-client-protocol-schema` does, so a
/// workspace build has it and `-p maki-providers` does not -- and cargo
/// unifies features across the graph rather than per crate. Key order would
/// then be a property of the `-p` flags, both in the golden itself and inside
/// the recorded body, which is a JSON document carried as a string. Sorting
/// every object at every depth, and the body's after parsing it, leaves one
/// canonical form for both builds to agree on.
fn canonical_observation(observed: &Value) -> Value {
    let mut canonical = sorted(observed);
    let requests = canonical
        .get_mut(REQUESTS_KEY)
        .and_then(Value::as_array_mut)
        .expect(NOT_AN_OBJECT);
    for request in requests {
        let Some(body) = request.get_mut(BODY_KEY) else {
            continue;
        };
        // A fixture whose request body is not JSON (or is empty) keeps the raw
        // string: there is no key order in it to leak.
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
pub(crate) fn assert_golden(provider: &str, fixture: &Fixture, observed: &Value) {
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

/// The extra assertion the differential half is made of: that the
/// implementation being ported *away from* puts the same bytes on the wire as
/// the declaration-driven one.
///
/// Deliberately *not* canonicalised, which is the whole difference between
/// this assertion and [`assert_golden`]. Both observations are produced by one
/// process under one feature set, so key order is not noise here -- it is part
/// of what the port has to reproduce, and sorting it away would let a body
/// that renamed or reordered a field pass as a match.
///
/// Deleting this function once the bespoke impl is gone costs no coverage:
/// every case is still asserted against its golden.
pub(crate) fn assert_ported(fixture: &Fixture, declared: &Value, ported: &Value) {
    assert!(
        ported == declared,
        "{} differs between the two implementations\n--- expected\n{}\n--- got\n{}",
        fixture.name,
        pretty(declared),
        pretty(ported)
    );
}
