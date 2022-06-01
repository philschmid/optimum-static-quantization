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
