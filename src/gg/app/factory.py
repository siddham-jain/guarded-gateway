from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx2
import structlog
from fastapi import FastAPI
from redis.asyncio import Redis

from gg import __version__
from gg.api.deps import DEFAULT_CONSUMED_EXTENSIONS, ApiServices, BuildInfo
from gg.api.install import drain, install_exception_handlers, install_middleware, install_routes
from gg.app.redis import connect_redis
from gg.app.spend import SpendGuardedAdapter
from gg.auth.config import load_keys
from gg.auth.failures import AuthFailureLimiter, InMemoryAuthFailureLimiter, RedisAuthFailureLimiter
from gg.auth.resolver import CachingKeyResolver
from gg.auth.store import YamlKeyStore
from gg.cache.config import CacheConfig
from gg.cache.embedders import build_embedder
from gg.cache.setup import BuiltCache, build_cache
from gg.config.hashing import combined_hash
from gg.config.loader import ConfigFile, CrossValidator, load_bundle
from gg.config.settings import Settings
from gg.core.aio import CpuExecutor, TaskSupervisor
from gg.core.clock import Clock, SystemClock
from gg.core.log import configure_logging
from gg.core.usage import UsageRecord
from gg.guardrails.registry import RemoteClients
from gg.guardrails.setup import Guardrails, build_guardrails
from gg.limits.base import ProviderSpendGuard
from gg.limits.config import LIMITS_CONFIG_FILE, LimitsConfig
from gg.limits.cost import CostCalculator
from gg.limits.setup import BuiltLimits, build_limits
from gg.limits.spend_guard import InMemorySpendGuard, RedisSpendGuard, SpendGuardSettings
from gg.observability.cache import CacheMetrics
from gg.observability.guardrails import GuardrailMetrics
from gg.observability.langfuse import (
    LangfuseTraceBuilder,
    LangfuseTracer,
    auth_headers,
    otlp_traces_url,
)
from gg.observability.limits import LimitsMetrics
from gg.observability.loop_lag import EventLoopLagSampler
from gg.observability.metrics import Metrics
from gg.observability.observer import MetricsObserver
from gg.observability.otlp import OtlpExporter
from gg.observability.reliability import ReliabilityMetrics
from gg.observability.routing import RoutingMetrics
from gg.observability.stage import ObservabilityStage
from gg.pipeline.probes import ConcurrentProbesStage
from gg.pipeline.runner import Pipeline
from gg.providers.base import ProviderAdapter
from gg.providers.catalog.catalog import Catalog
from gg.providers.catalog.loader import build_catalog, check_models_config
from gg.providers.catalog.schema import ModelsConfig
from gg.providers.http import HttpClientFactory
from gg.providers.registry import build_adapters
from gg.providers.runtime import AdapterDeps
from gg.providers.usage import estimate_prompt_tokens
from gg.reliability.breaker import BreakerRegistry
from gg.reliability.executor import Executor
from gg.reliability.policy import RetryPolicy
from gg.routing.config import RoutingConfig
from gg.routing.factory import RouterDeps, build_router
from gg.routing.stage import RoutingStage

log = structlog.get_logger("gg.app")

CONFIG_FILES: tuple[ConfigFile[Any], ...] = (
    ConfigFile("models", "models.yaml", ModelsConfig),
    ConfigFile("routing", "routing.yaml", RoutingConfig),
    ConfigFile("cache", "cache.yaml", CacheConfig),
    LIMITS_CONFIG_FILE,
)
CROSS_VALIDATORS: tuple[CrossValidator, ...] = (check_models_config,)


@dataclass(frozen=True, slots=True)
class Overrides:
    """test seams; production passes none"""

    clock: Clock | None = None
    transports: Mapping[str, httpx2.AsyncBaseTransport] = field(default_factory=lambda: {})
    default_transport: httpx2.AsyncBaseTransport | None = None
    spend_guard: ProviderSpendGuard | None = None


@dataclass(slots=True)
class Runtime:
    """long-lived resources the lifespan starts and closes"""

    http: HttpClientFactory
    adapters: Mapping[str, ProviderAdapter]
    lag_sampler: EventLoopLagSampler
    redis: Redis | None
    cpu: CpuExecutor
    cache: BuiltCache
    limits: BuiltLimits
    guardrails: Guardrails
    exporter: OtlpExporter | None = None

    async def start(self) -> None:
        await self.lag_sampler.start()
        if self.exporter is not None:
            await self.exporter.start()
        await self.guardrails.start()
        await self.cache.start()
        await self.limits.start()

    async def aclose(self) -> None:
        await self.lag_sampler.stop()
        await self.limits.stop()
        await self.cache.stop()
        for adapter in self.adapters.values():
            await adapter.aclose()
        if self.exporter is not None:
            # after drain, so the last requests' traces are flushed before the http clients close
            await self.exporter.stop()
        await self.http.aclose()
        if self.redis is not None:
            await self.redis.aclose()
        self.cpu.shutdown()


# static so metrics exist before the components that report into them
STAGE_NAMES = (
    "observability",
    "limits",
    "guard_in_pre",
    "guard_out",
    "cache_exact",
    ConcurrentProbesStage.name,
    RoutingStage.name,
)


def _metrics(catalog: Catalog, stage_names: tuple[str, ...]) -> Metrics:
    deployments = catalog.deployments
    return Metrics(
        aliases=catalog.alias_names,
        deployments=[d.id for d in deployments],
        providers=sorted({d.provider for d in deployments}),
        stages=(*stage_names, "terminal"),
        tiers=catalog.group_names,
    )


def _spend_guard(
    settings: Settings, redis: Redis | None, clock: Clock, metrics: Metrics
) -> ProviderSpendGuard:
    spend_settings = SpendGuardSettings()
    if redis is not None:
        return RedisSpendGuard(redis, spend_settings, clock=clock, telemetry=metrics)
    if settings.env == "prod":
        log.warning("spend_guard.in_memory", reason="GG_REDIS_URL is not set; caps reset on restart")
    return InMemorySpendGuard(spend_settings, clock=clock, telemetry=metrics)


def _auth_failure_limiter(settings: Settings, redis: Redis | None, clock: Clock) -> AuthFailureLimiter:
    cfg = settings.auth_failures
    if redis is not None:
        return RedisAuthFailureLimiter(redis, clock, max_failures=cfg.max_failures, window_s=cfg.window_s)
    return InMemoryAuthFailureLimiter(clock, max_failures=cfg.max_failures, window_s=cfg.window_s)


def _remote_detectors(settings: Settings, http: HttpClientFactory) -> RemoteClients | None:
    keys = (
        {"promptguard": settings.promptguard_api_key.get_secret_value()}
        if settings.promptguard_api_key
        else {}
    )
    return RemoteClients(client=http.client, api_keys=keys) if keys else None


def _langfuse(
    settings: Settings, http: HttpClientFactory, catalog: Catalog, clock: Clock, metrics: Metrics
) -> tuple[LangfuseTracer, OtlpExporter] | None:
    cfg = settings.langfuse
    if not cfg.enabled or cfg.public_key is None or cfg.secret_key is None:
        return None
    exporter = OtlpExporter(
        http.client("langfuse", cfg.host),
        otlp_traces_url(cfg.host),
        headers=auth_headers(cfg.public_key.get_secret_value(), cfg.secret_key.get_secret_value()),
        resource={
            "service.name": "gg-gateway",
            "service.version": __version__,
            "deployment.environment": settings.env,
        },
        on_dropped=lambda n: metrics.dropped("langfuse", n),
        queue_max=cfg.queue_max,
        batch_max=cfg.batch_max,
        flush_interval_s=cfg.flush_interval_s,
        timeout_s=cfg.timeout_s,
    )
    builder = LangfuseTraceBuilder(
        cfg,
        environment=settings.env,
        release=__version__,
        deployment={d.id: d for d in catalog.deployments}.get,
    )
    return LangfuseTracer(builder, exporter, clock=clock, sample_rate=cfg.sample_rate), exporter


def _cache_pricer(
    catalog: Catalog, costs: CostCalculator, clock: Clock
) -> Callable[[UsageRecord], float | None]:
    def price(usage: UsageRecord) -> float | None:
        breakdown = costs.cost(usage, catalog.get(usage.deployment_id), clock.now())
        return None if breakdown is None else breakdown.total / 1_000_000

    return price


def build_app(settings: Settings | None = None, *, overrides: Overrides | None = None) -> FastAPI:
    settings = settings or Settings()
    overrides = overrides or Overrides()
    configure_logging(settings.log_level, settings.log_format)
    clock = overrides.clock or SystemClock()

    bundle = load_bundle(CONFIG_FILES, CROSS_VALIDATORS, base_dir=settings.config_dir)
    catalog = build_catalog(bundle.section(ModelsConfig), settings, profile=settings.model_profile)
    keys_config = load_keys(settings.resolved_keys_file)
    keys = CachingKeyResolver(YamlKeyStore(keys_config), clock, allow_test_keys=settings.env != "prod")
    supervisor = TaskSupervisor(settings.runtime.background_task_max)
    cpu = CpuExecutor(settings.runtime.cpu_workers, settings.runtime.cpu_queue_max)
    metrics = _metrics(catalog, STAGE_NAMES)
    cache_config = bundle.section(CacheConfig)
    embedder = build_embedder(
        cache_config.semantic.embedder,
        cpu,
        cache_dir=settings.models_dir / "fastembed" if settings.models_dir else None,
    )
    http = HttpClientFactory(transports=overrides.transports, default_transport=overrides.default_transport)
    guardrails = build_guardrails(
        settings.config_dir / "policies",
        clock=clock,
        supervisor=supervisor,
        keys=[entry.policy() for entry in keys_config.keys],
        metrics=GuardrailMetrics(metrics),
        cpu=cpu,
        embedder=embedder,
        models_dir=settings.models_dir,
        remote=_remote_detectors(settings, http),
    )
    guard_stages = (guardrails.input_stage, guardrails.output_stage)
    costs = CostCalculator(metrics)
    redis = (
        connect_redis(
            settings.redis_url.get_secret_value(),
            max_connections=settings.runtime.redis_max_connections,
            pool_timeout_s=settings.runtime.redis_pool_timeout_s,
        )
        if settings.redis_url
        else None
    )
    cache = build_cache(
        cache_config,
        redis=redis,
        clock=clock,
        supervisor=supervisor,
        embedder=embedder,
        pricer=_cache_pricer(catalog, costs, clock),
        metrics_hooks=CacheMetrics(metrics),
    )
    limits = build_limits(
        bundle.section(LimitsConfig),
        redis=redis,
        clock=clock,
        catalog=catalog,
        costs=costs,
        estimate_prompt=estimate_prompt_tokens,
        metrics_hooks=LimitsMetrics(metrics),
    )
    jev_key = settings.jev_api_key.get_secret_value() if settings.jev_api_key else None
    router = build_router(
        bundle.section(RoutingConfig),
        RouterDeps(
            catalog=catalog,
            group_names=catalog.group_names,
            clock=clock,
            config_dir=settings.config_dir,
            http_client=lambda base_url: http.client("jev", base_url),
            jev_api_key=jev_key,
            hooks=RoutingMetrics(metrics),
        ),
    )
    config_hash = combined_hash(
        {
            **bundle.section_hashes,
            "catalog": catalog.hash,
            "routing_assets": router.assets_hash,
            "policies": guardrails.hash,
            "cache": cache.hash,
        }
    )

    probes = ConcurrentProbesStage(
        [p for p in (guardrails.tier2_probe, cache.semantic_probe, router.probe) if p is not None], clock
    )
    routing = RoutingStage(catalog)
    guard = overrides.spend_guard or _spend_guard(settings, redis, clock, metrics)

    adapters = build_adapters(catalog, AdapterDeps(http=http, clock=clock, env=settings.env))
    guarded = {name: SpendGuardedAdapter(a, guard, costs, clock) for name, a in adapters.items()}
    reliability = ReliabilityMetrics(metrics)
    executor = Executor(
        guarded, BreakerRegistry(clock, listener=reliability), RetryPolicy(clock), clock, hooks=reliability
    )
    langfuse = _langfuse(settings, http, catalog, clock, metrics)
    routing_config = bundle.section(RoutingConfig)
    strong_group = routing_config.profiles[routing_config.policy.default_profile].strong_group
    pipeline = Pipeline(
        [
            ObservabilityStage(
                metrics,
                clock=clock,
                pricer=costs,
                strong_deployment=lambda ctx: next(iter(catalog.chain(strong_group, ctx.request)), None),
                tracer=langfuse[0] if langfuse else None,
            ),
            limits.stage,
            *guard_stages,
            cache.exact_stage,
            probes,
            routing,
        ],
        executor,
        clock=clock,
        observer=MetricsObserver(metrics),
    )

    build_info = BuildInfo.from_env()
    metrics.set_build_info(version=__version__, git_sha=build_info.git_sha or "", config_hash=config_hash)
    metrics.set_config_info(
        config_hash=config_hash,
        models_hash=bundle.section_hashes.get("models", ""),
        routing_hash=bundle.section_hashes.get("routing", ""),
    )
    services = ApiServices(
        settings=settings,
        clock=clock,
        keys=keys,
        catalog=catalog,
        run_pipeline=pipeline.run,
        config_hash=config_hash,
        supervisor=supervisor,
        metrics_renderer=metrics.render,
        health_checks=(guardrails.health,),
        consumed_extensions=DEFAULT_CONSUMED_EXTENSIONS | cache.consumed_extensions,
        build_info=build_info,
    )
    runtime = Runtime(
        http=http,
        adapters=guarded,
        lag_sampler=EventLoopLagSampler(metrics, clock=clock),
        redis=redis,
        cpu=cpu,
        cache=cache,
        limits=limits,
        guardrails=guardrails,
        exporter=langfuse[1] if langfuse else None,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        await runtime.start()
        services.state.mark_ready()
        log.info(
            "app.startup",
            version=__version__,
            env=settings.env,
            model_profile=settings.model_profile,
            config_hash=config_hash,
            providers=sorted(adapters),
            langfuse=settings.langfuse.host if langfuse else None,
            maintenance=settings.maintenance,
        )
        try:
            yield
        finally:
            await drain(services)
            await runtime.aclose()
            log.info("app.shutdown")

    app = FastAPI(
        lifespan=lifespan,
        title="GG",
        version=__version__,
        docs_url=None if settings.env == "prod" else "/docs",
        redoc_url=None,
        openapi_url=None if settings.env == "prod" else "/openapi.json",
    )
    app.state.services = services
    install_middleware(app, settings, auth_failures=_auth_failure_limiter(settings, redis, clock))
    install_exception_handlers(app)
    install_routes(app)
    return app
