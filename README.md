# nlp-project

milestone 2 link: https://colab.research.google.com/drive/1bqQATiCbIvsXPAFeSdTciOUd8moJ-SSP?usp=sharing

# to run milestone 3

- first download the dataused for the vector database from this link: https://drive.google.com/file/d/10MJ3gxRZHDZvgw0ON7oCvlIKIWbcQ7-t/view?usp=sharing
- then put the downloaded file in the directory milestone 3/narrativeqa_1000

# then to run the backend and frontend of the chatbot

- navigate to milestone 3 folder

- activate the virtual enviroment by running the command

```
.venv/Scripts/activate
```

- run the command ' uvicorn main:app ' to run the backend (it may take 2-3 minutes to run as the data is fed to the vector db)

- finally run the command ' streamlit run frontend.py '
