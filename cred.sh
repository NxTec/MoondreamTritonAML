set -euo pipefail

EP_NAME="md-triton"

SCORING_URI=$(az ml online-endpoint show -n "$EP_NAME" --query scoring_uri -o tsv)
API_KEY=$(az ml online-endpoint get-credentials -n "$EP_NAME" --query primaryKey -o tsv)

echo "$SCORING_URI" > scoring_uri.txt
echo "$API_KEY" > key.txt

export SCORING_URI
export API_KEY

cat > .env <<EOF
SCORING_URI=${SCORING_URI%/}
API_KEY=$API_KEY
EOF

chmod 600 .env
echo "Wrote .env, scoring_uri.txt, key.txt"
echo "SCORING_URI=$SCORING_URI"