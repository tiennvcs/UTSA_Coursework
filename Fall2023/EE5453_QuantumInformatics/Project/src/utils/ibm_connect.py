import json
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from qiskit_ibm_provider import IBMProvider


def establish_connect(channel: str, token: str, instance: str) -> QiskitRuntimeService:
    # service = QiskitRuntimeService(channel=channel, token=token)
    # options = Options(optimization_level=1)
    IBMProvider.save_account(token=token, overwrite=True)
    provider = IBMProvider()
    provider = IBMProvider(instance=instance)
    return provider

def load_ibm_config(config_path: str):
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data

def load_running_config(config_path: str):
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data