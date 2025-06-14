# Experiment Tracking Lab

**Tools**: Python, MLFlow
<br>
**Topics**: Experiment tracking, Model Registry, Reproducibility

## Introduction

Many ML engineers have outstanding technical and analytical skills. However, as you start working with other teams in the company, you soon realize that as new approaches, features, and model iterations are tried, it becomes growingly complex to keep track of every single experiment carried out. Furthermore, many of the tasks they execute in their projects are repetitive and time-consuming.

In this lab you will learn how to keep track of your experiments.

## Instructions

For this lab, you are given a dataset and a basic load and preprocessing scripts to help you focus on the training phase and track all relevant information in this regard.
To check the correct tracking, you must open the MLFlow Server UI locally (`http://localhost:5000`) in any web browser, find your experiments and check out all the runs with their detailed information.

Prerequisites:
- Python 3.12 or greater
- Install the `requirements.txt` file
- Run in a terminal `mlflow server --host 127.0.0.1 --port 5000` to start the MLFlow tracking server

If you are new to experiment tracking or MLFlow, you can watch the following videos for reference:

* [Experiment tracking introduction](https://www.youtube.com/watch?v=MiA7LQin9c8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK)
* [Getting started with MLFlow](https://www.youtube.com/watch?v=cESCQE9J3ZE&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=9)
* [Experiment tracking with MLFlow](https://www.youtube.com/watch?v=iaJz-T7VWec&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=10)
* [Model management](https://www.youtube.com/watch?v=OVUPIX88q88&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=11)
* [Model registry](https://www.youtube.com/watch?v=TKHU7HAvGH8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=12)
* [MLFlow in practice](https://www.youtube.com/watch?v=1ykg4YmbFVA&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=13)

Repository Structure:
- `src/data` - Contains the dataset and a description file
- `src/utils.py` - Contains the basic functions for loading, preprocessing and splitting the dataset
- `src/autologging.py` - Contains the task regarding the usage of MLFlow autologging functionality
- `src/custom_logging.py` - Contains the task regarding custom logging with MLFlow
- `src/model_registry.py` - Contains the task regarding the usage of MLFlow Model Registry functionality

### Task 1: Experiment tracking using MLFlow autologging
**Time: 10 min**

Implement the required code in `src/autologging.py`, run the script and check in the MLFlow UI whether the required artifacts/information have been correctly logged.
This task will help you understand how easy and fast you can track relevant experiment information using the MLFlow autolog function.


### Task 2: Experiment tracking using MLFlow custom logging
**Time: 20 min**

Implement the required code in `src/model_registry.py`, run the script and check in the MLFlow UI whether the required artifacts/information have been correctly logged.
This task will help you understand the variety of information that can be tracked with MLFlow and how to customize it according to your own needs.

### Task 3: Experiment tracking using MLFlow model registry
**Time: 15 min**

Implement the required code in `src/custom_logging.py`, run the script and check in the MLFlow UI whether the required artifacts/information have been correctly logged.
This task will help you understand the usage of MLFlow Model Registry to manage our best models after a bunch of careful experiments, add useful metadata to them as signature, description and tags, and retrieve them when needed for further operations like deployment or validation.

### Optional task: 

Watch [MLFlow: benefits, limitations and alternatives](https://www.youtube.com/watch?v=Lugy1JPsBRY&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=14)

## Future Work

- Implement MLFlow tracking in training and inference pipelines
- Use AWS to track Sagemaker pipelines
