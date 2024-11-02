# Chatbot porject
# Setup instruction
## Windows 
[Download](https://github.com/ollama/ollama) (**recommend install in Windows**)
check ollama is running or not
```ollama serve ```
### Linux 
You can view [Linux instruction](https://github.com/ollama/ollama/blob/main/docs/linux.md) to view download instruction

### Download llm model from ollama
Choose your mode in [Ollama](https://ollama.com/library) then download
```ollama pull <your model> ```

check your model is available or not
```ollama serve```

### Download libraries
```pip install -r requirements.txt```

### Run app 
```chainlit run app.py --port 8000```
