# deep-path


## Usage

There are two possible ways to run the code: with or without Docker. If you run under Docker, you have to build a container and install nvidia-docker (https://github.com/NVIDIA/nvidia-docker). Otherwise, you can run scripts directly without ./run_container.sh


### 1. Checkout code

    git clone https://github.com/cog-isa/deep-path
    cd deep-path

### 2. Build docker container

    cd docker/<cuda_version>
    ./build.sh
    cd ../..

You should build container, depending on your cuda version - 6.5 or 7.5.

### 3. To experiment in an interactive manner, run Jupyter

    ./run_container.sh

### 4. Prepare data for cross-validation and evaluate

    ./run_container.sh ./split_data.py data/current/imported/paths data/current/folds
    ./run_container.sh ./evaluate.py --env=configs/env/default.yaml --agent=configs/agent/default.yaml --folds=data/current/folds --apply=configs/apply/default.yaml --output=results/default


## Data

Data folders are usually organized in the following way:

* data/dataset_title/raw - tasks in human-readable XML. Contain information about obstacles, start, finish and etalon path
* data/dataset_title/imported/maps - maps in compressed format, imported from XMLs
* data/dataset_title/imported/paths - start, finish and etalon paths in compressed format, imported from XMLs

For import procedure, see import.ipynb.
Default dataset_title is "current".


## Configs

The evaluation process is controlled using the following parameters:

* --env=configs/env/default.yaml is the environment configuration. Ctor is environment title to pass to gym.make and kwargs are arguments to pass to env.configure
* --agent=configs/agent/default.yaml is the agent configuration. Ctor is fully qualified name of agent class and kwargs are arguments to pass to constructor (__init__ method)
* --apply=configs/apply/default.yaml are the arguments to pass to apply_agent function. They control how much experience agent should receive and how often it trains.
* --folds=data/current/folds - folder with cross-validation data, prepared with split_data.sh
* --output=results/default - where to put the results in FGLab format


## Experiments and changes policy

If you would like to add some functionality and/or change agent or environment logic, please, subclass and decompose when possble, to minimize changes and to not to affect existing experiments.