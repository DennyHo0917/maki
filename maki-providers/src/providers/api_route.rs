use std::sync::{Arc, Mutex};

use flume::Sender;
use maki_config::providers::Protocol;
use maki_storage::id::SessionRef;
use serde_json::Value;

use crate::model::{Model, ModelFamily, ModelInfo};
use crate::provider::{BoxFuture, Provider};
use crate::providers::aperture::DEFAULT_PATH_PREFIX;
use crate::spec::{
    ApertureRoute, AuthDoc, CatalogDoc, GeneratedDocs, LoginConfig, NO_CURATED_MODELS, Native,
    ProviderSpec,
};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse};

use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use super::{KeyHeader, KeyPool, KeyRotation, ResolvedAuth, Timeouts};

const SLUG: &str = "api-route";
const DISPLAY_NAME: &str = "API Route";
const ENV_VAR: &str = "API_ROUTE_API_KEY";
const BASE_URL: &str = "https://global.api-route.com/v1";
const DEFAULT_MODEL: &str = "api-route/gpt-4o-mini";
const LOGIN_URL: &str = "https://www.api-route.com";
const DISCOVERY_NOTE: &str = "API Route lists models available to your key from its API. \
    Use their model IDs with the `api-route/` prefix (e.g. `api-route/gpt-4o-mini`).";

static CONFIG: OpenAiCompatConfig = OpenAiCompatConfig {
    slug: SLUG,
    api_key_env: ENV_VAR,
    base_url: BASE_URL,
    max_tokens_field: "max_tokens",
    include_stream_usage: true,
    provider_name: DISPLAY_NAME,
};

pub(crate) const SPEC: ProviderSpec = ProviderSpec {
    slug: SLUG,
    display_name: DISPLAY_NAME,
    api_key_env: ENV_VAR,
    family: ModelFamily::Generic,
    supports_thinking: false,
    accepts_arbitrary_models: true,
    fallback_max_output: None,
    fallback_context_window: 128_000,
    models_toml: NO_CURATED_MODELS,
    pricing_schedule: None,
    native: Some(Native {
        new: create,
        with_auth: create_with_auth,
        aperture: Some(ApertureRoute {
            path_prefix: DEFAULT_PATH_PREFIX,
        }),
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
        features: None,
        auth: AuthDoc::EnvVar,
        catalog: CatalogDoc::Discovered(DISCOVERY_NOTE),
        trailing_notes: &[],
    },
};

fn create(timeouts: Timeouts) -> Result<Box<dyn Provider>, AgentError> {
    Ok(Box::new(ApiRoute::new(timeouts)?))
}

fn create_with_auth(
    auth: Arc<Mutex<ResolvedAuth>>,
    timeouts: Timeouts,
    system_prefix: Option<String>,
) -> Box<dyn Provider> {
    Box::new(ApiRoute::with_auth(auth, timeouts).with_system_prefix(system_prefix))
}

inventory::submit!(SPEC.config_row());

pub struct ApiRoute {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    key_pool: Option<KeyPool>,
    system_prefix: Option<String>,
}

impl ApiRoute {
    fn new(timeouts: Timeouts) -> Result<Self, AgentError> {
        let pool = KeyPool::resolve(CONFIG.slug, CONFIG.api_key_env)?;
        Ok(Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth: Arc::new(Mutex::new(ResolvedAuth::bearer(
                CONFIG.slug,
                pool.current(),
            )?)),
            key_pool: Some(pool),
            system_prefix: None,
        })
    }

    fn with_auth(auth: Arc<Mutex<ResolvedAuth>>, timeouts: Timeouts) -> Self {
        Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth,
            key_pool: None,
            system_prefix: None,
        }
    }

    fn with_system_prefix(mut self, prefix: Option<String>) -> Self {
        self.system_prefix = prefix;
        self
    }
}

impl Provider for ApiRoute {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        _opts: RequestOptions,
        _session_id: Option<&'a SessionRef>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            let mut buf = String::new();
            let system = super::with_prefix(&self.system_prefix, system, &mut buf);
            let body = self.compat.build_body(model, messages, system, tools);
            self.compat
                .do_stream(model, &[], &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            self.compat.do_list_models(&auth).await
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
