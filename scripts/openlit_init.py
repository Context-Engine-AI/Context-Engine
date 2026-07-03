"""
openlit_init.py - Early OpenLit initialization for instrumentation.

This module MUST be imported before any qdrant_client, openai, or other
instrumented libraries to ensure OpenLit can properly wrap their classes.

Usage:
    import scripts.openlit_init  # Import early in entrypoint
    
Or in __init__.py:
    from scripts import openlit_init  # Ensure early load
"""
import os

_OPENLIT_INITIALIZED = False
_OPENLIT_ENABLED = os.environ.get("OPENLIT_ENABLED", "0").lower() in ("1", "true", "yes")


def init_openlit():
    """Initialize OpenLit instrumentation. Safe to call multiple times."""
    global _OPENLIT_INITIALIZED
    
    if _OPENLIT_INITIALIZED or not _OPENLIT_ENABLED:
        return _OPENLIT_INITIALIZED
    
    try:
        import openlit
        
        _otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://openlit:4318")
        _app_name = os.environ.get("OPENLIT_APP_NAME", "context-engine")
        _environment = os.environ.get("OPENLIT_ENVIRONMENT", "development")
        
        # Initialize OpenLit with modern params
        openlit.init(
            otlp_endpoint=_otel_endpoint,
            application_name=_app_name,
            environment=_environment,
            disabled_instrumentors=None,
            trace_content=True,
        )
        
        print(f"[OpenLit] Initialized with endpoint={_otel_endpoint}, service={_app_name}")
        
        # Get tracer provider for explicit instrumentation
        # OpenLit creates its own tracer provider during init()
        tracer_provider = None
        try:
            from opentelemetry import trace
            tracer_provider = trace.get_tracer_provider()
            print(f"[OpenLit] Got tracer_provider from opentelemetry: {type(tracer_provider).__name__}")
        except ImportError as e:
            print(f"[OpenLit] opentelemetry not available: {e}")
        except Exception as e:
            print(f"[OpenLit] tracer_provider error: {e}")
        
        # Fallback to OpenLit's tracer provider
        if tracer_provider is None and hasattr(openlit, 'tracer_provider'):
            tracer_provider = openlit.tracer_provider
            print(f"[OpenLit] Using openlit.tracer_provider: {type(tracer_provider).__name__ if tracer_provider else 'None'}")
        
        if tracer_provider is None:
            print("[OpenLit] WARNING: tracer_provider is None - hierarchy linking may not work")
        
        # Explicitly instrument OpenAI for GLM API calls
        try:
            from openlit.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument(
                tracer_provider=tracer_provider,
                environment=_environment,
                application_name=_app_name,
            )
            print("[OpenLit] OpenAI instrumentation enabled (for GLM API)")
        except ImportError:
            print("[OpenLit] OpenAI instrumentor not found")
        except Exception as e:
            print(f"[OpenLit] OpenAI instrumentation failed: {e}")
        
        # Explicitly instrument Qdrant
        try:
            from openlit.instrumentation.qdrant import QdrantInstrumentor
            QdrantInstrumentor().instrument(
                tracer_provider=tracer_provider,
                environment=_environment,
                application_name=_app_name,
            )
            print("[OpenLit] Qdrant instrumentation enabled")
        except ImportError:
            print("[OpenLit] Qdrant instrumentor not found")
        except Exception as e:
            print(f"[OpenLit] Qdrant instrumentation failed: {e}")
        
        _OPENLIT_INITIALIZED = True
        
    except ImportError:
        print("[OpenLit] SDK not installed, skipping observability")
    except Exception as e:
        print(f"[OpenLit] Failed to initialize: {e}")
    
    return _OPENLIT_INITIALIZED


# Auto-initialize on import if OPENLIT_ENABLED=1
if _OPENLIT_ENABLED:
    init_openlit()
