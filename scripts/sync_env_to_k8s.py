#!/usr/bin/env python3
"""Sync .env into Kubernetes configmap.yaml and inject envFrom into workloads.

Usage (from repo root):

    python scripts/sync_env_to_k8s.py

This will:
- Read .env at the repo root.
- Regenerate deploy/kubernetes/configmap.yaml so its data matches .env.
- Add envFrom: configMapRef: context-engine-config to every Deployment/Job
  container in deploy/kubernetes/*.yaml (if not already present).

Requires: PyYAML (pip install pyyaml)
"""

import argparse
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires PyYAML. Install with 'pip install pyyaml'.") from exc


def repo_root() -> Path:
    """Return the repo root (one level above scripts/)."""
    return Path(__file__).resolve().parents[1]


def parse_env_file(env_path: Path) -> dict:
    """Parse a simple KEY=VALUE .env file into a dict of strings.

    - Ignores blank lines and lines starting with '#'.
    - Splits on the first '='.
    - Strips surrounding quotes from values if present.
    """
    data: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            commented = False
            if line.startswith("#"):
                # Support commented defaults of the form "# KEY=VALUE" so that
                # template-style .env files can still drive configmap.yaml.
                stripped = line.lstrip("#").strip()
                if "=" not in stripped:
                    # Pure comment, skip.
                    continue

                key_part, value_part = stripped.split("=", 1)
                key_candidate = key_part.strip()

                # Only treat as a default if the key looks like an env var:
                # UPPERCASE letters, digits, and underscores, with no spaces/colons.
                if (
                    not key_candidate
                    or " " in key_candidate
                    or ":" in key_candidate
                    or not key_candidate.replace("_", "").isalnum()
                    or not key_candidate.isupper()
                ):
                    # Looks like prose (e.g. "Repository mode: 0=...") rather than
                    # an environment variable. Leave it as a comment.
                    continue

                line = f"{key_candidate}={value_part.strip()}"
                commented = True

            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            # Do not let commented defaults override explicit KEY=VALUE entries
            # that appeared earlier in the file.
            if commented and key in data:
                continue
            # Strip inline shell-style comments from unquoted values, so that
            # "VALUE  # comment" becomes just "VALUE".
            if "#" in value and not (value.startswith("\"") or value.startswith("'")):
                value = value.split("#", 1)[0].rstrip()
            # Strip single/double quotes if the whole value is quoted
            if (value.startswith("\"") and value.endswith("\"")) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            data[key] = str(value)
    return data


def update_configmap(
    configmap_path: Path,
    env_data: dict,
    name: str,
    namespace: str,
    exclude_keys: list[str] | None = None,
) -> None:
    """Regenerate configmap.yaml so that data matches env_data.

    Metadata (name/namespace) are ensured; any existing labels/annotations are preserved
    if present in the current file.

    Optionally excludes specific keys (e.g. sensitive secrets like GLM_API_KEY)
    from the generated ConfigMap data.
    """
    existing_meta = {}
    if configmap_path.exists():
        with configmap_path.open("r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
        if docs and isinstance(docs[0], dict):
            existing_meta = docs[0].get("metadata", {}) or {}

    metadata = dict(existing_meta)
    metadata.setdefault("name", name)
    metadata.setdefault("namespace", namespace)

    if exclude_keys is None:
        exclude_keys = []
    excluded = set(exclude_keys)

    data = {k: str(v) for k, v in sorted(env_data.items()) if k not in excluded}

    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": metadata,
        "data": data,
    }

    with configmap_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cm, f, default_flow_style=False, sort_keys=False)


def ensure_envfrom_in_doc(doc: dict, configmap_name: str) -> bool:
    """Ensure every container in a Deployment/Job has envFrom for the given ConfigMap.

    Returns True if the document was modified.
    """
    kind = doc.get("kind")
    if kind not in {"Deployment", "Job"}:
        return False

    spec = doc.get("spec") or {}
    template = spec.get("template") or {}
    pod_spec = template.get("spec") or {}
    containers = pod_spec.get("containers") or []
    if not isinstance(containers, list) or not containers:
        return False

    changed = False

    for container in containers:
        env_from = container.get("envFrom") or []
        if not isinstance(env_from, list):
            env_from = [env_from]

        already_present = any(
            isinstance(entry, dict)
            and "configMapRef" in entry
            and isinstance(entry["configMapRef"], dict)
            and entry["configMapRef"].get("name") == configmap_name
            for entry in env_from
        )

        if not already_present:
            env_from.append({"configMapRef": {"name": configmap_name}})
            container["envFrom"] = env_from
            changed = True
        else:
            # Normalise back to list form
            container["envFrom"] = env_from

    return changed


def update_workloads(k8s_dir: Path, configmap_name: str) -> None:
    """Walk deploy/kubernetes and inject envFrom into all Deployments/Jobs."""
    for path in sorted(k8s_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))

        if not docs:
            continue

        changed = False
        new_docs: list[dict] = []

        for doc in docs:
            if not isinstance(doc, dict):
                new_docs.append(doc)
                continue
            if ensure_envfrom_in_doc(doc, configmap_name):
                changed = True
            new_docs.append(doc)

        if changed:
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump_all(new_docs, f, default_flow_style=False, sort_keys=False)
            print(f"Updated envFrom in {path}")


def main() -> None:
    root = repo_root()

    parser = argparse.ArgumentParser(
        description="Sync .env into configmap.yaml and inject envFrom into Kubernetes workloads.",
    )
    parser.add_argument(
        "--env-file",
        default=str(root / ".env"),
        help="Path to .env file (default: repo_root/.env)",
    )
    parser.add_argument(
        "--k8s-dir",
        default=str(root / "deploy" / "kubernetes"),
        help="Path to Kubernetes manifests directory (default: deploy/kubernetes)",
    )
    parser.add_argument(
        "--configmap-name",
        default="context-engine-config",
        help="Name of the ConfigMap to update (default: context-engine-config)",
    )
    parser.add_argument(
        "--namespace",
        default="context-engine",
        help="Namespace for the ConfigMap (default: context-engine)",
    )
    parser.add_argument(
        "--exclude-key",
        action="append",
        default=None,
        help=(
            "Environment key to exclude from the generated ConfigMap data. "
            "May be passed multiple times. Defaults to ['GLM_API_KEY'] if not provided."
        ),
    )

    args = parser.parse_args()

    env_path = Path(args.env_file)
    k8s_dir = Path(args.k8s_dir)

    if not env_path.is_file():
        raise SystemExit(f".env file not found at {env_path}")
    if not k8s_dir.is_dir():
        raise SystemExit(f"Kubernetes directory not found at {k8s_dir}")

    print(f"Loading .env from {env_path}...")
    env_data = parse_env_file(env_path)
    print(f"Loaded {len(env_data)} keys from .env")

    configmap_path = k8s_dir / "configmap.yaml"
    print(f"Updating ConfigMap at {configmap_path}...")
    exclude_keys = args.exclude_key if args.exclude_key is not None else ["GLM_API_KEY"]
    if exclude_keys:
        print(f"Excluding keys from ConfigMap: {', '.join(sorted(set(exclude_keys)))}")
    update_configmap(configmap_path, env_data, args.configmap_name, args.namespace, exclude_keys)

    print(f"Injecting envFrom: configMapRef: {args.configmap_name} into workloads under {k8s_dir}...")
    update_workloads(k8s_dir, args.configmap_name)

    print("Done. Review and commit the updated YAMLs if they look correct.")


if __name__ == "__main__":  # pragma: no cover
    main()
