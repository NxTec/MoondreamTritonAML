from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def get_ml_client():
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id="8fc2cf5f-c01b-43ab-baaf-8c69466df6b7",
        resource_group_name="collaborateai-spaceport",
        workspace_name="CollaborateAI-spaceport"
    )

