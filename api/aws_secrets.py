import boto3
import json
from functools import lru_cache

@lru_cache()
def get_secret(secret_name: str, region_name: str = "us-east-1"):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return json.loads(secret)
        else:
            # Handle binary secrets if any
            secret = get_secret_value_response['SecretBinary']
            return json.loads(secret)
    except Exception as e:
        raise e
