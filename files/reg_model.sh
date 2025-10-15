az ml model create \
  --name moondream_triton \
  --version 3 \
  --type triton_model \
  --path ./model-repo \
  --description "base"
