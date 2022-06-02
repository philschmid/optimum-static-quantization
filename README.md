# Static Quantization with Hugging Face Optimum

Example on how to static quantization with Hugging Face Optimum and evalute.

## Setup

The repository includes a `create_c6i_instance.sh` which is a bash script that creates a `c6i` instance on AWS, which you can connect to with `ssh` or by using the [vscode remote ssh plugin](https://code.visualstudio.com/docs/remote/ssh). 

```bash
./create_c6i_instance.sh
```

**ssh + jupyter lab**

```bash
ssh -i optimum.pem ubuntu@<public-ip>
```
install jupyter lab
```bash
pip3 install jupyter
jupyter notebook
```

### [Miniconda](https://waylonwalker.com/install-miniconda/#installing-miniconda-on-linux) or [Micromamba](https://labs.epi2me.io/conda-or-mamba-for-production/) setup (conda alternative but smaller)

Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Micromamba
```bash
sudo apt-get install bzip2
# Assuming linux
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest -o test | tar -xvj -C ~/
~/bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc

# Installing packages is mostly similar
micromamba activate
micromamba install python=3.9 jupyter -c conda-forge
```

### Install python dependencies

```bash
pip install -r requirements.txt
```