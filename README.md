# oss_nlp_tools_demos

This repo contains demos of Open Source NLP tools

## Setup

#### Dev Container (preferred)

The repo could run locally on a virtual environment, but I recommend using the Dev Container setup.

For a dev container setup in VScode, you'd need

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. The Python and Dev Containers VSCode extensions.

    Once installed, check that you see a new icon at the bottom-left of the screen, it should looks like this: `><` with the right bracket a bit higher than the left bracket.

3. Open the repo in the container.

    The next thing to do is to run the Docker container specified in Dockerfile (with Python) and open this repository in that container. To do this, click on the `><` icon bottom-left of the screen and select "Reopen in Container". Once all requirements defined in requirements.txt are installed, the environment is set and you can code forward.

#### Virtual Environment

If you prefer to work on a virtual environment, you can do your usual routine, for example. 

```shell
python3 -m venv nlp_tools
source nlp_tools/bin/activate
pip install -r requirements.txt
```

## Dataset 

#TODO

## Tools

#TODO