#!/bin/bash
# Set your Azure ML API key as an environment variable:
# export AZURE_ML_API_KEY="your-api-key-here"
KEY="${AZURE_ML_API_KEY}"

#./simple_test.py --api-key "${KEY}"

#./simple_test.py --api-key "${KEY}" --modes caption
#./simple_test.py --api-key "${KEY}" --modes caption --caption-length normal
#./simple_test.py --api-key "${KEY}" --modes query
#./simple_test.py --api-key "${KEY}" --modes detect
#./simple_test.py --api-key "${KEY}" --modes query --query "describe this image as concise as possible"
./simple_test.py --api-key "${KEY}" --modes detect --detect-object "number 71346" --output number_71346_detection.jpg
#./simple_test.py --api-key "${KEY}" --image det1.jpg
#./simple_test.py --api-key "${KEY}" --modes caption query
#./simple_test.py --api-key "${KEY}" --modes detect --output my_detection.jpg
