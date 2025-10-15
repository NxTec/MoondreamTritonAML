import requests
import json
import base64
from pathlib import Path
import sys
import os
from time import perf_counter

ENDPOINT_URL = "https://md-triton.eastus.inference.ml.azure.com/v2/models/md/infer"

API_KEY = os.environ.get("AZURE_ML_API_KEY", "")

def encode_image_to_base64(image_path):
    
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_request(payload, description):
    

    payload_json = json.dumps(payload)

    request_data = {
        "inputs": [
            {
                "name": "payload",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [payload_json]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    print(f"\n{'='*70}")
    print(description)
    print(f"{'='*70}")

    total_start = perf_counter()

    try:
        http_start = perf_counter()
        response = requests.post(
            ENDPOINT_URL,
            headers=headers,
            json=request_data,
            timeout=180
        )
        http_elapsed_ms = (perf_counter() - http_start) * 1000
        print(f"Request latency (HTTP round trip): {http_elapsed_ms:.1f} ms")

        process_start = perf_counter()

        if response.status_code == 200:
            result = response.json()
            if 'outputs' in result and len(result['outputs']) > 0:
                response_data = result['outputs'][0]['data'][0]
                parsed = json.loads(response_data)
                process_elapsed_ms = (perf_counter() - process_start) * 1000
                print(f"Response processing: {process_elapsed_ms:.1f} ms")
                print(f"✅ Parsed response: {parsed}")
                return parsed
            else:
                print(f"❌ Unexpected response format: {result}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except requests.exceptions.Timeout:
        print("❌ Request timed out after 180 seconds")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        total_elapsed_ms = (perf_counter() - total_start) * 1000
        print(f"Total elapsed time: {total_elapsed_ms:.1f} ms")

    return None


def query_moondream(image_base64, question):
    payload = {
        "image": image_base64,
        "query": question
    }
    result = send_request(payload, f"Question: {question}")
    if result:
        return result.get('answer')
    return None


def detect_sign(image_base64):
    payload = {
        "image": image_base64,
        "mode": "detect",
        "object": "sign"
    }
    result = send_request(payload, "Detecting signs")
    if result:
        objects = result.get('objects')
        if objects:
            print(f"Detected {len(objects)} object(s): {objects}")
        else:
            print("No objects detected.")


def caption_image(image_base64):
    payload = {
        "image": image_base64,
        "mode": "caption"
    }
    result = send_request(payload, "Generating caption")
    if result:
        caption = result.get('caption')
        if caption:
            print(f"Caption: {caption}")
        else:
            print("No caption returned.")

def test_image(image_path, queries):
    
    print(f"\n{'#'*70}")
    print(f"# Testing Image: {image_path.name}")
    print(f"{'#'*70}")
    
    
    try:
        image_base64 = encode_image_to_base64(image_path)
        print(f"Image encoded successfully ({len(image_base64)} bytes)")
    except Exception as e:
        print(f"Failed to encode image: {e}")
        return
    
    caption_image(image_base64)

    detect_sign(image_base64)

    for query in queries:
        answer = query_moondream(image_base64, query)
        if answer:
            print(f"Answer: {answer}")

def main():
    print("\n" + "="*70)
    print("  MOONDREAM TRITON ENDPOINT TEST")
    print("="*70)
    print(f"Endpoint: {ENDPOINT_URL}")
    print("="*70)
    
    test_images_dir = Path("/data/sarah/moondream_triton/SINGLE-TEST-IMAGES")
    
    if not test_images_dir.exists():
        print(f"Test images directory not found: {test_images_dir}")
        sys.exit(1)
    
    image_files = list(test_images_dir.glob("*.jpg")) + \
                  list(test_images_dir.glob("*.jpeg")) + \
                  list(test_images_dir.glob("*.png"))
    
    if not image_files:
        print(f"No image files found in {test_images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} test image(s)")
    
    test_queries = [
        "Describe this image in detail.",
        "What is the main subject of this image?",
        "What objects can you see in this image?",
        "What colors are prominent in this image?",
        "What is happening in this scene?",
    ]
    
    test_image(image_files[0], test_queries)
    
    if len(image_files) > 1:
        for img_path in image_files[1:]:
            test_image(img_path, ["Describe this image in detail."])
    
    print("\n" + "="*70)
    print("  TEST COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

