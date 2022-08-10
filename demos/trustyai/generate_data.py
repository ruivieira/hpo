from typing import Dict, List
import requests
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

EXPERIMENT_NAME = "trustyai"
URL = "http://localhost:8085/experiment_trials"
N = 100

# Data taken from https://github.com/kruize/autotune-results/blob/main/spring-petclinic/tunables-cpu-mem-req/experiment-1/same-requests-limits/experiment-data.csv

INPUTS = ["Responsetime", "cpu_req", "mem_req", "cpu_lim", "mem_lim"]
OUTPUT = ["Throughput"]


def create_tunables(_data: pd.DataFrame, names: List[str]) -> List[Dict]:
    tunables = []
    N = len(_data)
    for name in names:
        d = data[name]
        tunables.append(
            {
                "value_type": "double",
                "lower_bound": d.min(),
                "name": name,
                "upper_bound": d.max(),
                "step": (d.max() - d.min()) / N,
            }
        )
    return tunables


def create_experiment(name: str, id: int, data: pd.DataFrame, tunable_names: List[str]):
    payload = {
        "operation": "EXP_TRIAL_GENERATE_NEW",
        "search_space": {
            "experiment_id": str(id),
            "experiment_name": name,
            "total_trials": len(data),
            "parallel_trials": 1,
            "hpo_algo_impl": "optuna_tpe",
            "objective_function": "Throughput",
            "value_type": "double",
            "direction": "maximize",
            "tunables": create_tunables(data, tunable_names),
        },
    }
    response = requests.post(URL, json=payload)
    print(response)


def get_parameters(name: str, i: int) -> List[Dict]:
    response = requests.get(f"{URL}?experiment_name={name}&trial_number={i}")
    return json.loads(response.text)


def create_payload(trial: int, model, parameters: List[Dict]) -> Dict:
    x = np.array([[t["tunable_value"] for t in parameters]])
    value = model.predict(x)[0]

    return {
        "experiment_name": EXPERIMENT_NAME,
        "operation": "EXP_TRIAL_RESULT",
        "trial_number": trial,
        "trial_result": "success",
        "result_value_type": "double",
        "result_value": value,
    }


def send_trial_data(data: Dict):
    response = requests.post(URL, json=data)
    # TODO: Check status


def stop_experiment(name: str):
    payload = {"operation": "EXP_STOP", "experiment_name": name}
    response = requests.post(URL, json=payload)
    print(response)
    print(response.text)
    print(response.content)


if __name__ == "__main__":
    data = pd.read_csv("data/experiment-data.csv")
    print(data)

    # train the model to generate new data
    model = RandomForestRegressor(max_depth=3)
    X = data[INPUTS].to_numpy()
    Y = data[OUTPUT].to_numpy().ravel()
    model.fit(X, Y)
    create_experiment(EXPERIMENT_NAME, 3, data=data, tunable_names=INPUTS)
    for n in range(0, len(data)):
        print("=" * 80)
        parameters = get_parameters(EXPERIMENT_NAME, n)
        print("Got proposals:")
        param_str = ", ".join([f"{p['tunable_name']}: {'{:.2f}'.format(p['tunable_value'])}" for p in parameters])
        print("\t" + param_str)
        payload = create_payload(n, model, parameters)
        send_trial_data(payload)
        print(
            f"Sending trial observation {n} to Kruize HPO [{payload['result_value']}]"
        )
    stop_experiment(EXPERIMENT_NAME)
