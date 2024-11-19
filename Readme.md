# Chatbot porject
you can view my colab [here](https://colab.research.google.com/drive/1hyhNTFxY5rWf6hOxg6705kXP37H1QjjP?usp=sharing)

# Setup instruction
## Windows 
[Download](https://github.com/ollama/ollama) (**recommend install in Windows**)
check ollama is running or not
```ollama serve ```

### Linux (I have problem while using Linux so you can read other instruction using Linux)
=======
### Linux 

You can view [Linux instruction](https://github.com/ollama/ollama/blob/main/docs/linux.md) to view download instruction

### Download llm model from ollama
Choose your mode in [Ollama](https://ollama.com/library) then download
```ollama pull <your model> ```

check your model is available or not
```ollama serve```

### Download libraries
```pip install -r requirements.txt```


### Download chroma_db
```gdown 1eXcXuDoxJIA8TyJkbpWRLacORumArxXH```
then unzip file
```tar -xvf chroma_db_pypdf.zip```

### Run app 
```chainlit run app.py --port 8000```

