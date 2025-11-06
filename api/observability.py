"""OpenTelemetry configuration and initialization for Autonote."""

import logging
import os

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Service identification
SERVICE_NAME_VALUE = os.getenv("OTEL_SERVICE_NAME", "autonote-api")
SERVICE_VERSION_VALUE = os.getenv("OTEL_SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


def get_resource() -> Resource:
    """Create OpenTelemetry resource with service attributes."""
    return Resource.create(
        {
            SERVICE_NAME: SERVICE_NAME_VALUE,
            SERVICE_VERSION: SERVICE_VERSION_VALUE,
            "deployment.environment": ENVIRONMENT,
        }
    )


def configure_tracing() -> TracerProvider:
    """Configure OpenTelemetry tracing."""
    resource = get_resource()
    provider = TracerProvider(resource=resource)

    # Check if traces should be enabled
    enable_traces = os.getenv("OTEL_ENABLE_TRACES", "true").lower() == "true"

    if enable_traces:
        # Determine exporter based on configuration
        exporter_type = os.getenv("OTEL_TRACES_EXPORTER", "console")

        if exporter_type == "otlp":
            # OTLP exporter for production
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if otlp_endpoint:
                span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                print(f"[OTEL] Using OTLP span exporter: {otlp_endpoint}")
            else:
                print("[OTEL] OTLP endpoint not configured, traces disabled")
                enable_traces = False
        elif exporter_type == "console":
            # Console exporter for development
            span_exporter = ConsoleSpanExporter()
            print("[OTEL] Using console span exporter for development")
        else:
            # 'none' or any other value disables trace export
            print("[OTEL] Trace export disabled")
            enable_traces = False

        if enable_traces:
            # Add batch processor for efficient export
            processor = BatchSpanProcessor(span_exporter)
            provider.add_span_processor(processor)
    else:
        print("[OTEL] Tracing disabled via OTEL_ENABLE_TRACES=false")

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    return provider


def configure_metrics() -> MeterProvider:
    """Configure OpenTelemetry metrics."""
    resource = get_resource()

    # Check if metrics should be enabled
    enable_metrics = os.getenv("OTEL_ENABLE_METRICS", "true").lower() == "true"

    metric_readers = []

    if enable_metrics:
        # Determine exporter based on configuration
        exporter_type = os.getenv("OTEL_METRICS_EXPORTER", "console")

        if exporter_type == "otlp":
            # OTLP exporter for production
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if otlp_endpoint:
                metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
                print(f"[OTEL] Using OTLP metric exporter: {otlp_endpoint}")
            else:
                print("[OTEL] OTLP endpoint not configured, metrics disabled")
                enable_metrics = False
        elif exporter_type == "console":
            # Console exporter for development
            metric_exporter = ConsoleMetricExporter()
            print("[OTEL] Using console metric exporter for development")
        else:
            # 'none' or any other value disables metric export
            print("[OTEL] Metric export disabled")
            enable_metrics = False

        if enable_metrics:
            # Create metric reader with export interval
            reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "60000")),
            )
            metric_readers.append(reader)
    else:
        print("[OTEL] Metrics disabled via OTEL_ENABLE_METRICS=false")

    provider = MeterProvider(resource=resource, metric_readers=metric_readers)

    # Set as global meter provider
    metrics.set_meter_provider(provider)

    return provider


def add_otel_context(logger, method_name, event_dict):
    """Add OpenTelemetry trace context to log events."""
    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    else:
        event_dict["trace_id"] = "0" * 32
        event_dict["span_id"] = "0" * 16
    return event_dict


def configure_logging():
    """Configure structlog with OpenTelemetry integration."""
    # Set log level from environment
    log_level = os.getenv("OTEL_LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()  # json or console

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level),
    )

    # Shared processors for all configurations
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_otel_context,  # Add OpenTelemetry trace context
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    # Choose renderer based on format
    if log_format == "json":
        # JSON format for production (Splunk, ELK, etc.)
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development (human-readable)
        processors = shared_processors + [
            structlog.processors.ExceptionRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Get a logger and log initialization
    logger = structlog.get_logger()
    logger.info("logging_configured", log_level=log_level, log_format=log_format)


def initialize_observability():
    """Initialize all OpenTelemetry components."""
    # Configure logging first
    configure_logging()

    # Get structlog logger
    logger = structlog.get_logger()
    logger.info("initializing_observability")

    # Configure components
    tracer_provider = configure_tracing()
    meter_provider = configure_metrics()

    logger.info(
        "observability_initialized",
        service_name=SERVICE_NAME_VALUE,
        service_version=SERVICE_VERSION_VALUE,
        environment=ENVIRONMENT,
    )

    return tracer_provider, meter_provider


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(name, SERVICE_VERSION_VALUE)


def get_meter(name: str = __name__) -> metrics.Meter:
    """Get a meter instance for creating metrics."""
    return metrics.get_meter(name, SERVICE_VERSION_VALUE)


# Custom metrics for the application
class AppMetrics:
    """Application-specific metrics."""

    def __init__(self):
        meter = get_meter("autonote.metrics")

        # Counters
        self.user_registrations = meter.create_counter(
            name="user.registrations", description="Total number of user registrations", unit="1"
        )

        self.user_logins = meter.create_counter(
            name="user.logins", description="Total number of user logins", unit="1"
        )

        self.chat_messages = meter.create_counter(
            name="chat.messages", description="Total number of chat messages processed", unit="1"
        )

        self.auth_failures = meter.create_counter(
            name="auth.failures", description="Total number of authentication failures", unit="1"
        )

        # Histograms
        self.request_duration = meter.create_histogram(
            name="http.server.request.duration",
            description="HTTP request duration in milliseconds",
            unit="ms",
        )

        self.db_query_duration = meter.create_histogram(
            name="db.query.duration",
            description="Database query duration in milliseconds",
            unit="ms",
        )


# Global metrics instance
app_metrics: AppMetrics | None = None


def get_app_metrics() -> AppMetrics:
    """Get the global application metrics instance."""
    global app_metrics
    if app_metrics is None:
        app_metrics = AppMetrics()
    return app_metrics
