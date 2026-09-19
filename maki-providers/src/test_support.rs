//! A recorded loopback server: the one copy every replay suite in the
//! workspace serves its transcripts from.
//!
//! Synchronous and dependency-light on purpose. It is the thing a provider is
//! compared *against*, so it shares nothing with the async stack under test:
//! a blocking `TcpListener`, a fixed script, and the request bytes kept
//! verbatim. Verbatim matters, because a recorder that parsed the body first
//! would quietly repair whatever the client got wrong, and each suite is left
//! to decide for itself what counts as a difference.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use serde_json::Value;

const LOOPBACK: &str = "127.0.0.1:0";
const REASON_PHRASE: &str = "Recorded";
const CONTENT_LENGTH: &str = "content-length";
const AUTHORIZATION: &str = "authorization";
const BIND_FAILED: &str = "cannot bind loopback";
const IO_FAILED: &str = "the recorded connection broke";

pub const SSE_HEADERS: &[(&str, &str)] = &[("content-type", "text/event-stream")];
pub const JSON_HEADERS: &[(&str, &str)] = &[("content-type", "application/json")];

/// One recorded response, replayed in script order.
pub struct Canned {
    pub status: u16,
    pub headers: &'static [(&'static str, &'static str)],
    pub body: &'static str,
}

impl Canned {
    pub const fn sse(body: &'static str) -> Self {
        Self {
            status: 200,
            headers: SSE_HEADERS,
            body,
        }
    }

    pub const fn json(status: u16, body: &'static str) -> Self {
        Self {
            status,
            headers: JSON_HEADERS,
            body,
        }
    }
}

/// What the client actually put on the wire, before anything parsed it.
pub struct Recorded {
    pub method: String,
    pub path: String,
    /// Lowercased names, in name order, so a comparison reads the same on
    /// every run whatever order the client emitted them in.
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn authorization(&self) -> &str {
        self.headers.get(AUTHORIZATION).map_or("", String::as_str)
    }

    /// The body as the provider meant it, for assertions about content rather
    /// than about bytes.
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

/// Every request the server has answered, in order. Each entry lands before
/// its response is written, so a request the client has an answer to is
/// already here.
pub type Requests = Arc<Mutex<Vec<Recorded>>>;

/// Serves `script` in order on loopback, one connection per entry.
///
/// The log is shared rather than joined and the server thread is detached: a
/// script is an upper bound on what a run sends, and a caller replaying one
/// script through two providers has to be able to read what the shorter of
/// them sent without parking forever on an `accept` that will never return.
pub fn serve(script: &'static [Canned]) -> (String, Requests) {
    let listener = TcpListener::bind(LOOPBACK).expect(BIND_FAILED);
    let base_url = format!("http://{}/v1", listener.local_addr().expect(BIND_FAILED));
    let requests = Requests::default();
    let log = Arc::clone(&requests);
    std::thread::spawn(move || {
        for canned in script {
            let Ok((stream, _)) = listener.accept() else {
                return;
            };
            log.lock().unwrap().push(read_request(&stream));
            write_canned(&stream, canned);
        }
    });
    (base_url, requests)
}

fn read_request(stream: &TcpStream) -> Recorded {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).expect(IO_FAILED);
    let mut start = request_line.split_whitespace();
    let method = start.next().unwrap_or_default().to_owned();
    let path = start.next().unwrap_or_default().to_owned();

    let mut headers = BTreeMap::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect(IO_FAILED);
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }

    let length = headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).expect(IO_FAILED);
    Recorded {
        method,
        path,
        headers,
        body,
    }
}

fn write_canned(mut stream: &TcpStream, canned: &Canned) {
    let mut response = format!(
        "HTTP/1.1 {} {REASON_PHRASE}\r\n{CONTENT_LENGTH}: {}\r\nconnection: close\r\n",
        canned.status,
        canned.body.len()
    );
    for (name, value) in canned.headers {
        response.push_str(&format!("{name}: {value}\r\n"));
    }
    response.push_str("\r\n");
    response.push_str(canned.body);
    stream.write_all(response.as_bytes()).expect(IO_FAILED);
    stream.flush().expect(IO_FAILED);
}
