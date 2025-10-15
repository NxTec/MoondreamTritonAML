from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_ENDPOINT_URL = "https://md-triton.eastus.inference.ml.azure.com/v2/models/md/infer"
DEFAULT_IMAGE_DIR = Path("/data/sarah/moondream_triton/SINGLE-TEST-IMAGES")
DEFAULT_MODES = ("detect", "caption", "query")
DEFAULT_MAX_CONCURRENCY = 16
DEFAULT_TIMEOUT = 180
DEFAULT_QUERIES = [
    "Describe this image in detail.",
    "What is the main subject of this image?",
    "What objects can you see in this image?",
    "What colors are prominent in this image?",
    "What is happening in this scene?",
]


class RequestFailure(Exception):
    """Raised when an inference request fails."""


def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_test_images(image_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")

    if limit is not None:
        image_files = image_files[:limit]

    return [{"path": p, "base64": encode_image_to_base64(p)} for p in image_files]


def build_payload(mode: str, image_b64: str, request_index: int,
                  queries: Iterable[str], detect_object: str) -> Dict[str, Any]:
    if mode == "detect":
        return {"image": image_b64, "mode": "detect", "object": detect_object}
    if mode == "caption":
        return {"image": image_b64, "mode": "caption"}
    if mode == "query":
        qlist = list(queries)
        prompt = qlist[request_index % len(qlist)]
        return {"image": image_b64, "mode": "query", "query": prompt}
    raise ValueError(f"Unsupported mode: {mode}")


def parse_triton_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        top = json.loads(text)
        outputs = top.get("outputs")
        if not outputs:
            return None, "Missing 'outputs' in response"
        data_blob = outputs[0]["data"][0]
        parsed = json.loads(data_blob)
        if isinstance(parsed, dict) and parsed.get("error"):
            return None, f"Model error: {parsed['error']}"
        return parsed, None
    except Exception as e:  # noqa: BLE001
        return None, f"Parse error: {e}"


def send_request(session: requests.Session, endpoint_url: str, api_key: str,
                 payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    payload_json = json.dumps(payload)
    triton_request = {
        "inputs": [
            {"name": "payload", "shape": [1, 1], "datatype": "BYTES", "data": [payload_json]}
        ]
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    t0 = time.perf_counter()
    resp = session.post(endpoint_url, headers=headers, json=triton_request, timeout=timeout)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    result: Dict[str, Any] = {"status_code": resp.status_code, "latency_ms": lat_ms, "raw_text": resp.text}
    if resp.status_code != 200:
        return result

    parsed, perr = parse_triton_json(resp.text)
    if perr:
        result["error"] = perr
    else:
        result["parsed"] = parsed
    return result


def _run_once(
    mode: str,
    concurrency: int,
    endpoint_url: str,
    api_key: str,
    images: List[Dict[str, Any]],
    queries: Iterable[str],
    detect_object: str,
    timeout: int,
    session: requests.Session,
) -> Dict[str, Any]:
    latencies: List[float] = []
    errors: List[str] = []
    status_codes: List[int] = []

    def worker(idx: int) -> Dict[str, Any]:
        img = images[idx % len(images)]
        payload = build_payload(mode, img["base64"], idx, queries, detect_object)
        return send_request(session, endpoint_url, api_key, payload, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, i) for i in range(concurrency)]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                errors.append(f"Unhandled exception: {e}")
                continue

            status_codes.append(r.get("status_code", -1))
            if r.get("status_code") != 200:
                errors.append(f"HTTP {r.get('status_code')}: {r.get('raw_text')[:200]}")
                continue
            if "error" in r:
                errors.append(r["error"])
                continue
            if r.get("latency_ms") is not None:
                latencies.append(r["latency_ms"])

    out: Dict[str, Any] = {
        "mode": mode,
        "concurrency": concurrency,
        "total_requests": concurrency,
        "successes": len(latencies),
        "failures": len(errors),
        "errors": errors,
        "status_codes": status_codes,
        "code_hist": dict(Counter(status_codes)),
        "latencies_ms": latencies,
    }
    if latencies:
        out.update(
            {
                "latency_min_ms": min(latencies),
                "latency_max_ms": max(latencies),
                "latency_avg_ms": mean(latencies),
            }
        )
    return out


def _run_sustained(
    mode: str,
    concurrency: int,
    duration_s: int,
    endpoint_url: str,
    api_key: str,
    images: List[Dict[str, Any]],
    queries: Iterable[str],
    detect_object: str,
    timeout: int,
    session: requests.Session,
) -> Dict[str, Any]:
    """
    Keep exactly N in-flight workers for 'duration_s'. Each worker loops sequentially.
    """
    stop_time = time.time() + duration_s
    lock = threading.Lock()

    successes = 0
    failures = 0
    latencies: List[float] = []
    status_codes: List[int] = []
    errors: List[str] = []

    def loop_worker(worker_id: int) -> None:
        nonlocal successes, failures
        idx = worker_id
        while time.time() < stop_time:
            img = images[idx % len(images)]
            payload = build_payload(mode, img["base64"], idx, queries, detect_object)
            idx += len(images)  
            try:
                r = send_request(session, endpoint_url, api_key, payload, timeout)
            except Exception as e:  
                with lock:
                    failures += 1
                    errors.append(f"Unhandled exception: {e}")
                continue

            with lock:
                status_codes.append(r.get("status_code", -1))
                if r.get("status_code") != 200:
                    failures += 1
                    errors.append(f"HTTP {r.get('status_code')}: {r.get('raw_text')[:200]}")
                elif "error" in r:
                    failures += 1
                    errors.append(r["error"])
                else:
                    successes += 1
                    if r.get("latency_ms") is not None:
                        latencies.append(r["latency_ms"])

    threads = [threading.Thread(target=loop_worker, args=(i,), daemon=True) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration_s + 5)

    out: Dict[str, Any] = {
        "mode": mode,
        "concurrency": concurrency,
        "duration_s": duration_s,
        "total_requests": successes + failures,
        "successes": successes,
        "failures": failures,
        "errors": errors[:50], 
        "status_codes": status_codes,
        "code_hist": dict(Counter(status_codes)),
        "latencies_ms": latencies,
    }
    if latencies:
        out.update(
            {
                "latency_min_ms": min(latencies),
                "latency_max_ms": max(latencies),
                "latency_avg_ms": mean(latencies),
            }
        )
    return out


def execute_level(
    mode: str,
    concurrency: int,
    endpoint_url: str,
    api_key: str,
    images: List[Dict[str, Any]],
    queries: Iterable[str],
    detect_object: str,
    timeout: int,
    session: requests.Session,
    duration: Optional[int],
) -> Dict[str, Any]:
    if duration and duration > 0:
        return _run_sustained(mode, concurrency, duration, endpoint_url, api_key,
                              images, queries, detect_object, timeout, session)
    return _run_once(mode, concurrency, endpoint_url, api_key,
                     images, queries, detect_object, timeout, session)


def run_mode_test(
    mode: str,
    endpoint_url: str,
    api_key: str,
    images: List[Dict[str, Any]],
    queries: Iterable[str],
    detect_object: str,
    concurrency_levels: Iterable[int],
    timeout: int,
    stop_at_first_failure: bool,
    duration: Optional[int],
    session: requests.Session,
) -> Dict[str, Any]:
    print("\n" + "#" * 70)
    print(f"# Testing mode: {mode}")
    print("#" * 70)

    _ = execute_level(mode, 1, endpoint_url, api_key, images, queries, detect_object, timeout, session, duration=0)

    results: List[Dict[str, Any]] = []
    best_success_level = 0
    first_failing_level: Optional[int] = None

    for c in concurrency_levels:
        if c <= 0:
            print(f"Skipping invalid concurrency: {c}")
            continue

        summary = execute_level(mode, c, endpoint_url, api_key, images, queries, detect_object,
                                timeout, session, duration)
        results.append(summary)

        total = summary["total_requests"]
        succ = summary["successes"]
        fail = summary["failures"]
        hist = summary.get("code_hist", {})
        rate = (succ / total) * 100 if total else 0.0

        if fail == 0 and succ == total and total > 0:
            best_success_level = c

        line = (f"Concurrency {c}: {succ}/{total} succeeded ({rate:.1f}%) | "
                f"codes {hist} ")
        if "latency_avg_ms" in summary:
            line += (f"| latency avg/min/max: "
                     f"{summary['latency_avg_ms']:.1f}/"
                     f"{summary['latency_min_ms']:.1f}/"
                     f"{summary['latency_max_ms']:.1f} ms")
        print(line)

        if fail > 0 and first_failing_level is None:
            first_failing_level = c
            if stop_at_first_failure:
                break

    return {
        "mode": mode,
        "results": results,
        "max_successful_concurrency": best_success_level,
        "first_failing_concurrency": first_failing_level,
    }


def save_json(path: Optional[Path], data: Any) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Optional[Path], summaries: List[Dict[str, Any]]) -> None:
    if not path:
        return
    import csv

    rows = []
    for s in summaries:
        mode = s["mode"]
        for r in s["results"]:
            rows.append({
                "mode": mode,
                "concurrency": r["concurrency"],
                "duration_s": r.get("duration_s", 0),
                "total_requests": r.get("total_requests", 0),
                "successes": r.get("successes", 0),
                "failures": r.get("failures", 0),
                "latency_avg_ms": r.get("latency_avg_ms", ""),
                "latency_min_ms": r.get("latency_min_ms", ""),
                "latency_max_ms": r.get("latency_max_ms", ""),
                "code_hist": json.dumps(r.get("code_hist", {})),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode", "concurrency", "duration_s",
                "total_requests", "successes", "failures",
                "latency_avg_ms", "latency_min_ms", "latency_max_ms",
                "code_hist",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concurrent no-break load test for Moondream Triton endpoint")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT_URL, help="Endpoint URL for inference")
    p.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"),
                   help="API key (or set MOONDREAM_API_KEY)")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory of test images")
    p.add_argument("--image-limit", type=int, default=None, help="Limit number of images loaded")
    p.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=list(DEFAULT_MODES),
                   help="Modes to exercise")
    p.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
                   help="Max concurrency to attempt if --sweep is set")
    p.add_argument("--concurrency", type=int, nargs="+",
                   help="Explicit concurrency levels (overrides --max-concurrency)")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep 1..--max-concurrency (default tests only --max-concurrency)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout (seconds)")
    p.add_argument("--detect-object", default="sign", help="Object name for detect mode")
    p.add_argument("--stop-at-first-failure", action="store_true",
                   help="Stop a mode's sweep at the first failing level")
    p.add_argument("--duration", type=int, default=0,
                   help="If >0, hold concurrency for this many seconds at each level (sustained test)")
    p.add_argument("--output-json", type=Path, default=None, help="Path to write JSON results")
    p.add_argument("--output-csv", type=Path, default=None, help="Path to write CSV summary rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("API key is required. Provide --api-key or set MOONDREAM_API_KEY.")

    print("=" * 70)
    print("  MOONDREAM TRITON CONCURRENCY TEST  (no-break)")
    print("=" * 70)
    print(f"Endpoint: {args.endpoint}")
    print(f"Image directory: {args.image_dir}")
    print(f"Modes: {', '.join(args.modes)}")

    if args.concurrency:
        levels = sorted({c for c in args.concurrency if c > 0})
        if not levels:
            raise SystemExit("At least one positive concurrency level is required.")
        print(f"Concurrency levels to test: {levels}")
    elif args.sweep:
        levels = list(range(1, args.max_concurrency + 1))
        print(f"Sweeping 1..{args.max_concurrency}")
    else:
        levels = [args.max_concurrency]
        print(f"Testing single concurrency level: {levels[0]}")

    images = load_test_images(args.image_dir, args.image_limit)
    print(f"Loaded {len(images)} image(s)")

    summaries: List[Dict[str, Any]] = []
    with requests.Session() as session:
        for mode in args.modes:
            summary = run_mode_test(
                mode=mode,
                endpoint_url=args.endpoint,
                api_key=args.api_key,
                images=images,
                queries=DEFAULT_QUERIES,
                detect_object=args.detect_object,
                concurrency_levels=levels,
                timeout=args.timeout,
                stop_at_first_failure=args.stop_at_first_failure,
                duration=args.duration if args.duration > 0 else None,
                session=session,
            )
            summaries.append(summary)

    print("\n" + "=" * 70)
    print("  SUMMARY (max concurrency with 0% errors)")
    print("=" * 70)
    for s in summaries:
        print(
            f"Mode {s['mode']}: max_success={s['max_successful_concurrency']} | "
            f"first_fail={s['first_failing_concurrency']}"
        )

    save_json(args.output_json, summaries)
    save_csv(args.output_csv, summaries)


if __name__ == "__main__":
    main()