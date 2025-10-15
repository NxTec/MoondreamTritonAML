
import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ENDPOINT_URL = "https://md-triton.eastus.inference.ml.azure.com/v2/models/md/infer"
DEFAULT_IMAGE_PATH = Path("/data/sarah/moondream_triton/SINGLE-TEST-IMAGES")
DEFAULT_TIMEOUT = 180
DEFAULT_QUERY = "What objects can you see in this image?"
DEFAULT_DETECT_OBJECT = "sign"


def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def send_request(endpoint_url: str, api_key: str, payload: dict, timeout: int) -> tuple:

    payload_json = json.dumps(payload)
    triton_request = {
        "inputs": [
            {"name": "payload", "shape": [1, 1], "datatype": "BYTES", "data": [payload_json]}
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        t0 = time.perf_counter()
        resp = requests.post(endpoint_url, headers=headers, json=triton_request, timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            return None, latency_ms, f"HTTP {resp.status_code}: {resp.text[:200]}"

        top = resp.json()
        outputs = top.get("outputs")
        if not outputs:
            return None, latency_ms, "Missing 'outputs' in response"

        data_blob = outputs[0]["data"][0]
        parsed = json.loads(data_blob)

        if isinstance(parsed, dict) and parsed.get("error"):
            return None, latency_ms, f"Model error: {parsed['error']}"

        return parsed, latency_ms, None

    except Exception as e:
        return None, 0, f"Request failed: {e}"


def test_caption(endpoint_url: str, api_key: str, image_b64: str, timeout: int, length: str = "normal"):
    print("\n" + "=" * 70)
    print("TESTING: CAPTION MODE")
    print("=" * 70)
    print(f"Length: {length}")

    payload = {"image": image_b64, "mode": "caption", "length": length}
    result, latency, error = send_request(endpoint_url, api_key, payload, timeout)

    print(f"Latency: {latency:.2f} ms")

    if error:
        print(f"ERROR: {error}")
        return

    print(f"\nCaption:\n{result}")


def test_query(endpoint_url: str, api_key: str, image_b64: str, query: str, timeout: int):
    print("\n" + "=" * 70)
    print("TESTING: QUERY MODE")
    print("=" * 70)
    print(f"Query: '{query}'")

    payload = {"image": image_b64, "mode": "query", "query": query}
    result, latency, error = send_request(endpoint_url, api_key, payload, timeout)

    print(f"Latency: {latency:.2f} ms")

    if error:
        print(f"ERROR: {error}")
        return

    print(f"\nResponse:\n{result}")


def test_detect(endpoint_url: str, api_key: str, image_b64: str, image_path: Path, 
                detect_object: str, timeout: int, output_path: Path):
    print("\n" + "=" * 70)
    print("TESTING: DETECT MODE")
    print("=" * 70)
    print(f"Detecting: '{detect_object}'")

    payload = {"image": image_b64, "mode": "detect", "object": detect_object}
    result, latency, error = send_request(endpoint_url, api_key, payload, timeout)

    print(f"Latency: {latency:.2f} ms")

    if error:
        print(f"ERROR: {error}")
        return


    if isinstance(result, list):
        detections = result
    elif isinstance(result, dict) and "detections" in result:
        detections = result["detections"]
    elif isinstance(result, dict) and "objects" in result:
        detections = result["objects"]
    else:
        detections = []

    print(f"\nFound {len(detections)} detection(s)")

    if not detections:
        print("No objects detected.")
        return

    for i, det in enumerate(detections):
        print(f"\nDetection {i + 1}:")
        print(f"  Bounding box: {det}")

    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        img_width, img_height = img.size

        for i, det in enumerate(detections):
            if isinstance(det, dict):
                x1 = det.get("x_min", det.get("x1", 0))
                y1 = det.get("y_min", det.get("y1", 0))
                x2 = det.get("x_max", det.get("x2", 0))
                y2 = det.get("y_max", det.get("y2", 0))
            elif isinstance(det, list) and len(det) >= 4:
                x1, y1, x2, y2 = det[:4]
            else:
                continue
                
            x1_px = int(x1 * img_width)
            y1_px = int(y1 * img_height)
            x2_px = int(x2 * img_width)
            y2_px = int(y2 * img_height)

            draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=3)
            
            label = f"{detect_object} {i + 1}"
            draw.text((x1_px, y1_px - 25), label, fill="red", font=font)

        img.save(output_path)
        print(f"\nVisualization saved to: {output_path}")

    except Exception as e:
        print(f"\nWARNING: Could not create visualization: {e}")


def get_test_image(image_path: Path) -> Path:
    if image_path.is_file():
        return image_path
    
    if image_path.is_dir():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(image_path.glob(ext))
            if images:
                return sorted(images)[0]
        raise FileNotFoundError(f"No images found in directory: {image_path}")
    
    raise FileNotFoundError(f"Image path not found: {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple single-request test for Moondream Triton")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT_URL, help="Endpoint URL")
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"),
                       help="API key (or set MOONDREAM_API_KEY)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH,
                       help="Path to test image or directory")
    parser.add_argument("--query", default=DEFAULT_QUERY,
                       help="Query text for query mode")
    parser.add_argument("--detect-object", default=DEFAULT_DETECT_OBJECT,
                       help="Object to detect in detect mode")
    parser.add_argument("--caption-length", choices=["normal", "short"], default="normal",
                       help="Caption length: normal or short")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                       help="Request timeout in seconds")
    parser.add_argument("--output", type=Path, default=Path("detection_result.jpg"),
                       help="Output path for detection visualization")
    parser.add_argument("--modes", nargs="+", choices=["detect", "caption", "query"],
                       default=["detect", "caption", "query"],
                       help="Modes to test")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("ERROR: API key required. Provide --api-key or set MOONDREAM_API_KEY.")

    try:
        image_file = get_test_image(args.image)
        print("=" * 70)
        print("MOONDREAM TRITON - SIMPLE TEST")
        print("=" * 70)
        print(f"Image: {image_file}")
        print(f"Endpoint: {args.endpoint}")
        print(f"Testing modes: {', '.join(args.modes)}")
        
        image_b64 = encode_image_to_base64(image_file)

    except Exception as e:
        raise SystemExit(f"ERROR: Failed to load image: {e}")

    if "caption" in args.modes:
        test_caption(args.endpoint, args.api_key, image_b64, args.timeout, args.caption_length)

    if "query" in args.modes:
        test_query(args.endpoint, args.api_key, image_b64, args.query, args.timeout)

    if "detect" in args.modes:
        test_detect(args.endpoint, args.api_key, image_b64, image_file, 
                   args.detect_object, args.timeout, args.output)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

