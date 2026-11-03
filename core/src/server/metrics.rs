//! The figures behind GET /metrics and the monitor page: what the engine thread is doing now, and what
//! every recent request did. The engine thread is the only writer (`begin`, `token`, `end`); the HTTP
//! handler reads a copy under a lock that is never held across engine work. All request numbers come
//! from the engine worker (`engine::run_job` / `engine::run_batch`), which measures them for its log.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Finished requests kept for the page, newest first.
const HISTORY: usize = 100;

fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Which API served a request, from the id prefix `request_id` builds: "chatcmpl-" (OpenAI chat),
/// "msg_" (Anthropic), "resp_" (OpenAI responses).
fn api_of(id: &str) -> &'static str {
    for (prefix, api) in [("chatcmpl-", "openai"), ("msg_", "anthropic"), ("resp_", "responses")] {
        if id.starts_with(prefix) {
            return api;
        }
    }
    "api"
}

/// One served request, as /metrics reports it and the monitor page shows it.
#[derive(Debug, Clone, Serialize)]
pub struct RequestRecord {
    /// unix seconds when the request finished
    pub time: f64,
    pub id: String,
    pub api: &'static str,
    pub prompt_tokens: usize,
    /// prompt tokens served from a prefix cache
    pub cached_prompt_tokens: usize,
    pub generated: usize,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// prompt tokens read per second, the cached ones not counted (None: nothing was read)
    pub prefill_tok_s: Option<f64>,
    /// tokens per second over the tokens after the first (None: one token or fewer)
    pub decode_tok_s: Option<f64>,
    /// "stop", "length", "cancelled" or "error"
    pub finish: &'static str,
    /// the request's whole stay in the engine: queue, prefill and decode
    pub duration_s: f64,
}

/// The worker's durations for one request: the queue wait, the prefill until the first token, the decode
/// of the rest, and the wall time the engine spent on the request (the queue wait is not part of it).
#[derive(Debug, Clone, Copy)]
pub struct Timings {
    pub queued: Duration,
    pub prefill: Duration,
    pub decode: Duration,
    pub total: Duration,
}

impl RequestRecord {
    /// The request's numbers as the worker measured them: `prefill` and `decode` are the engine's own
    /// timings when it reports them, else the worker's fallbacks.
    pub fn new(
        id: &str,
        prompt_tokens: usize,
        cached_prompt_tokens: usize,
        generated: usize,
        t: Timings,
        finish: &'static str,
    ) -> Self {
        let cached = cached_prompt_tokens.min(prompt_tokens);
        let read = prompt_tokens - cached;
        let secs = |d: Duration| d.as_secs_f64();
        Self {
            time: unix_now(),
            id: id.into(),
            api: api_of(id),
            prompt_tokens,
            cached_prompt_tokens: cached,
            generated,
            queue_ms: secs(t.queued) * 1e3,
            prefill_ms: secs(t.prefill) * 1e3,
            decode_ms: secs(t.decode) * 1e3,
            prefill_tok_s: (read > 0 && secs(t.prefill) > 0.0).then(|| read as f64 / secs(t.prefill)),
            decode_tok_s: (generated > 1 && secs(t.decode) > 0.0).then(|| (generated - 1) as f64 / secs(t.decode)),
            finish,
            duration_s: secs(t.total),
        }
    }
}

/// What the engine is doing right now.
#[derive(Debug, Clone, Serialize)]
pub struct Live {
    /// "idle", "reading" (prefilling the prompt) or "generating"
    pub state: &'static str,
    /// the running request's id ("" when idle)
    pub id: String,
    /// the running request's prompt size (None on the batched engine)
    pub prompt_tokens: Option<usize>,
    /// tokens the running request has produced (None on the batched engine)
    pub generated: Option<usize>,
    /// sequences running at once; 1 on the single-engine path
    pub active: usize,
    /// seconds since the request entered the engine (None when idle)
    pub elapsed_s: Option<f64>,
    /// tokens per second while generating (None until the second token)
    pub tok_s: Option<f64>,
}

/// Totals over every finished request since the server started.
#[derive(Debug, Default, Clone, Serialize)]
pub struct Totals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub generated_tokens: u64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// the best decode rate of any finished answer, and when it finished (unix seconds)
    pub peak_decode_tok_s: Option<f64>,
    pub peak_decode_time: Option<f64>,
}

/// The engine's part of GET /metrics; the route adds the server facts (model, context, queue).
#[derive(Debug, Serialize)]
pub struct Snapshot {
    /// unix seconds when the engine thread started
    pub since: f64,
    pub uptime_s: f64,
    pub live: Live,
    pub totals: Totals,
    /// newest first, at most [`HISTORY`] entries
    pub requests: Vec<RequestRecord>,
}

/// The live engine figures behind GET /metrics: the running request and the recent finished ones.
/// The engine thread is the only writer; readers copy a snapshot under the lock.
#[derive(Debug)]
pub struct Metrics {
    started: Instant,
    since: f64,
    s: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    live: InnerLive,
    totals: Totals,
    requests: VecDeque<RequestRecord>,
}

#[derive(Debug)]
struct InnerLive {
    state: &'static str,
    id: String,
    prompt_tokens: Option<usize>,
    generated: Option<usize>,
    active: usize,
    started: Option<Instant>,
    first: Option<Instant>,
}

impl Default for InnerLive {
    fn default() -> Self {
        Self { state: "idle", id: String::new(), prompt_tokens: None, generated: None, active: 0, started: None, first: None }
    }
}

impl InnerLive {
    fn running(state: &'static str, id: &str, prompt_tokens: Option<usize>, active: usize) -> Self {
        Self { state, id: id.into(), prompt_tokens, generated: Some(0), active, started: Some(Instant::now()), first: None }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self { started: Instant::now(), since: unix_now(), s: Mutex::new(Inner::default()) }
    }

    /// The single-engine path: request `id` entered the engine and its prefill starts.
    pub fn begin(&self, id: &str, prompt_tokens: usize) {
        self.s.lock().unwrap().live = InnerLive::running("reading", id, Some(prompt_tokens), 1);
    }

    /// The single-engine path: one token was produced. The first one ends the prefill.
    pub fn token(&self) {
        let mut s = self.s.lock().unwrap();
        let l = &mut s.live;
        if l.state == "idle" {
            return;
        }
        l.state = "generating";
        l.first.get_or_insert_with(Instant::now);
        if let Some(g) = l.generated.as_mut() {
            *g += 1;
        }
    }

    /// The single-engine path: the request ended. Records it and the engine is idle again.
    pub fn end(&self, rec: RequestRecord) {
        self.push(rec, true);
    }

    /// The batched engine: record one finished sequence. The live state is set separately by
    /// [`Metrics::batch_active`], because several sequences end at once there.
    pub fn record(&self, rec: RequestRecord) {
        self.push(rec, false);
    }

    /// The batched engine: `active` sequences are running (0 = idle). Its steps mix prefill and
    /// decode, so "generating" covers both; per-token detail is not tracked on that path.
    pub fn batch_active(&self, active: usize) {
        let mut s = self.s.lock().unwrap();
        s.live = if active == 0 { InnerLive::default() } else { InnerLive::running("generating", "", None, active) };
    }

    /// A copy for GET /metrics.
    pub fn snapshot(&self) -> Snapshot {
        let s = self.s.lock().unwrap();
        let now = Instant::now();
        let secs = |d: Duration| d.as_secs_f64();
        let l = &s.live;
        let live = Live {
            state: l.state,
            id: l.id.clone(),
            prompt_tokens: l.prompt_tokens,
            generated: l.generated,
            active: l.active,
            elapsed_s: l.started.map(|t| secs(now - t)),
            tok_s: match (l.first, l.generated) {
                (Some(f), Some(g)) if g > 1 => Some((g - 1) as f64 / secs(now - f).max(1e-9)),
                _ => None,
            },
        };
        Snapshot { since: self.since, uptime_s: secs(self.started.elapsed()), live, totals: s.totals.clone(), requests: s.requests.iter().cloned().collect() }
    }

    /// Totals, peak and history from one record; `idle` clears the live state.
    fn push(&self, rec: RequestRecord, idle: bool) {
        let mut s = self.s.lock().unwrap();
        let t = &mut s.totals;
        t.requests += 1;
        t.prompt_tokens += rec.prompt_tokens as u64;
        t.cached_prompt_tokens += rec.cached_prompt_tokens as u64;
        t.generated_tokens += rec.generated as u64;
        t.prefill_ms += rec.prefill_ms;
        t.decode_ms += rec.decode_ms;
        if rec.decode_tok_s.is_some_and(|v| v > t.peak_decode_tok_s.unwrap_or(0.0)) {
            t.peak_decode_tok_s = rec.decode_tok_s;
            t.peak_decode_time = Some(rec.time);
        }
        s.requests.push_front(rec);
        while s.requests.len() > HISTORY {
            s.requests.pop_back();
        }
        if idle {
            s.live = InnerLive::default();
        }
    }
}
