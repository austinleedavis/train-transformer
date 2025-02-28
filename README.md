# Train Transformers


## Key Features

This repository is used to train a chess-playing transformer on UCI move sequences. It provides some very nice features:
- **Automated setup** with Makefile
- **Distributed training** using [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable)
- **Containerized environment** for fast-deployment via [Docker](https://www.docker.com/) or [Apptainer](https://apptainer.org/)
- **Configuration management** via [Hydra](https://hydra.cc/)
- **Training monitoring** via [Weights & Biases](https://wandb.ai/)
- **Start/stop/error notifications** via [Ntfy.sh](https://ntfy.sh/), plus the ability to interrupt training remotely via notify.sh


## Prerequisites

1. [GNU Make](https://www.gnu.org/software/make/) (shortcut command for building/running the container)
1. Docker Engine ([Installation Guide](https://docs.docker.com/engine/install/)): Follow installation steps for your Linux distribution, or use [Docker Desktop](https://docs.docker.com/desktop/) for Windows
1. NVIDIA Container Toolkit ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)): Follow steps for *Installation* **and** *Configuration*

<details><summary><b>Move Docker's default data-dir (Only if neeeded)</b><br></summary>

On my system, I have a lot of free space at `/home`, but very little in docker's default directory. Run the following commands to update Docker to store its data in a different directory.

1. Shutdown Docker service

   ```shell
   sudo systemctl stop docker docker.socket
   sudo systemctl status docker
   ```

2. Move data to the new path (if it's not already there)

   ```shell
   sudo mkdir -p /etc/docker
   sudo rsync -avxP /var/lib/docker/ /home/docker/
   echo '{
     "data-root": "/home/docker"
   }' | sudo tee /etc/docker/daemon.json
   ```

3. Restart the Docker services

   ```shell
   sudo systemctl restart docker
   ```

</details>

#### Useful links:

- [Pytorch compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [Official NVIDIA/CUDA docker images](https://hub.docker.com/r/nvidia/cuda/tags)

## Installation

   1. **Clone the Repository**
      First, clone the repository into your desired project directory:

      ```sh
      git clone https://github.com/austinleedavis/train-transformer.git
      cd train-transformer
      ```

   1. **Build the Docker Container**
      To set up your development environment, build the Docker container with all required dependencies:

      ```sh
      make docker-build
      ```

      Once the build completes, you have multiple options for accessing the environment.


### Accessing the Development Environment
Post-build, there are three options to access the development environment.

#### Option 1: ***VS Code DevContainer***
If you use VS Code, you can work inside the container with **Dev Containers**:

<details><summary>Details...</summary>

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open the project in VS Code.
3. Open the command palette (**Ctrl+Shift+P** / **Cmd+Shift+P**) and select:
   ```
   Dev Containers: Reopen in Container
   ```

This will start a development session inside the Docker environment.

</details>

#### Option 2: ***Running Scripts with Docker***
You can send non-interactive scripts to be executed inside the Docker container. 

<details><summary>Details...</summary> 

Run:

```sh
docker run --rm -v $(pwd):/workspace $(basename $(pwd)):latest bash -c "./scripts/train.sh"
```

</details>

#### Option 3: ***Using Apptainer (For Cluster Environments)***
For systems that use **Apptainer** instead of Docker (e.g., managed HPC clusters), it's easiest to push the container image to [Docker Hub](https://hub.docker.com/) (requires an account), then pull it to the cluster.
<details><summary>Details...</summary>

Follow these steps:

  1. **Allocate a Compute Node (if required)**
  Some clusters require an allocation before running GPU workloads:
      ```sh
      salloc --time=1:00:00 --gres=gpu:1
      ```
      Once granted, note the assigned node and connect to it:
      ```sh
      ssh <assigned_node>
      ```
  
  1. **Load Required Modules**.
      Ensure **Apptainer** and **CUDA** are available:
      ```sh
      module load apptainer
      module load cuda/cuda-12.4.0
      ```
  
  1. **Pull the Container Image**
      To use your Docker container with Apptainer, first push it to [Docker Hub](https://hub.docker.com/) (requires an account). Then, pull it onto the cluster:
      
      ```sh
      apptainer pull docker://<your_username>/train-transformer
      ```
  
  1. **Run the Container and Check GPU Access**
      ```sh
      apptainer run --nv ~/containers/train-transformer_Latest.sif
      ```
      
      Once inside the container (you should see an `Apptainer>` prompt), verify GPU availability:
      
      ```sh
      Apptainer> nvidia-smi
      ```

      If the GPUs are recognized, you're all set!
</details>

---


### Environment Variables

You should create a `.env` to save several environment variables. For example:

```sh
WANDB_API_KEY=... # alternatively, log in to wandb within the container.
NTFY_TOPIC=<your_topic_here> # the topic to which you will publish/subscribe notifications
HYDRA_CONFIG_PATH=configs # the path to your hydra configurations. (Best practice: use a config folder outside the git repository)
```

## Project Structure

```
.
├── configs/      
|   └── ...       # configuration templates. Check train.yaml first
├── scripts/      
|   └── ...       # bash/slurm scripts to facilitate job execution
├── src/          
|   └── ...       # base classes and code for the project
├── Dockerfile 
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Notes

- **Configurations**: Modify `configs/train.yaml` to adjust training settings and paths.
- **Logs & Checkpoints**: Stored in `outputs/` folder organized by date/time of each run.
