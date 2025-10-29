# Text-summarizer project

This project is an end-to-end **Text Summarization Project** built using a fine tuned **Pegasus Transformer Model**.
It can generate concise summaries from long texts or paragraphs.

---

the **goal** of this project is to summaries large bolcks of text into a few sentencees while preserving meaning and key information.
Itv uses **Transfer Learning** with the[Google Peagsus] architecture, fine tuned for dialogue and news summarization tasks.

## Workflows

1. update config.yaml
2. update params.yaml
3. update entity
4. update the configuration manager file in src config
5. update components
6. update pipeline
7. update main.py
8. update app.py



Checkout the live Demo: [Live Demo](https://huggingface.co/spaces/Anuj-Verma/Text-summarizer-demo)



# How to run?

## Steps:

Clone the repository

```
https://github.com/AnujVerma-hub/Text-summarizer.git
```


STEP 1-create a virtual environment after opening the repo 

```
python3 -m venv texts
```
### For Windows

```
texts/scripts/activate
```



STEP 2-install the requirements

```
pip install -r requirements.txt
```
```
# Then run the following command
python app.py
```

I've created two application files, app.py and app2.py.
First use fastapi interface
and second use gradio interface.


Now,

```
open up your local host and port
```

```
Author: Anuj verma
Email: anujvermamarch@gmail.com
```



## Fastapi app:


<img width="1875" height="891" alt="Screenshot 2025-10-11 113118" src="https://github.com/user-attachments/assets/7af26102-9dd3-499e-8ffa-ee8c03f694f7" />



## Geadio app:


<img width="1670" height="836" alt="Screenshot 2025-10-29 191110" src="https://github.com/user-attachments/assets/3737e888-28f3-4387-82b5-eb109977f60d" />



# Deployment Guide

## Steps:

1. Create an account on Huggingface.

  -> [Huggingface](https://huggingface.co)

3. Create a new huggingface space for deployment.

   - Space name: (e.g., text summarizer)
   - SDK: choose gradio
   - Visibility: Public or Private.

4. Upload your project files.

   - app.py
   - requirements.txt
   - models
   - Readme.md

5. Define requirements.
   

Then the huggingface spaces automatically detect the app.py file and launch the app.


















