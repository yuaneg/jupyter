# 获取模型信息

```
ollama show --modelfile llama3.1:latest
```

# jupyter启动命令

```
 docker run --gpus all -d -it -p 8888:8888  -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root -v /f/jovyan:/home/jovyan quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.4.1
```

# chat_template

```
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""

```


