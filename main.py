import PyPDF2
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import VectorStore
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv, find_dotenv

caminho_arquivo = r"C:\Users\sergi\Downloads\Frederico Wanderley Tavares_ Iuri Soter Viana Segtovich_ Fernando de Azevedo Medeiros - Termodinâmica .pdf"

with open(caminho_arquivo,"rb") as arquivo:
    leitor_pdf = PyPDF2.PdfReader(arquivo)   # Leitura do PDF 
   

    texto = "" #armazena todas as pastar no formato de texto (srt)

    for pagina in leitor_pdf.pages:

        total_paginas = len(leitor_pdf.pages)
        texto += pagina.extract_text()

    
print("PDF Lido ")

# Dividindo o texto 

divisor = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=50,
    separators=["\n\n","\n",".",", "]

)

print("Vamos Dividir")

chunks = divisor.split_text(texto)

#for chunk in chunks: 
    #print("\n".join(chunk[:3]))

print(f"📜 Texto dividido em {len(chunks)} pedaços!")

# Transformando os pedaços em vetores | Embedding

print("Incorporando os pedaços")

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

model = OpenAIEmbeddings(openai_api_key = os.getenv("OPENAI_API_KEY"))

db = FAISS.from_texts(chunks, model)  # O FAISS cria um banco de dados pros vetores 

#print(vetor)

print("Bando de vetores Criados")

# Criando o Chatbot 

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

# Criando um Chain com RAG


qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Perguntas 

print("Vamos começar as perguntas ?")

while True:

    pergunta = input("\n Faça uma pergunta ou escreva 'sair' para encerrar XD : ")

    if pergunta.lower() == "sair":  # o lower converte todas as letras em minusculas
        print ("Foi legal te encontrar! Volte mais vezes ^^")
        break

    resposta = qa.invoke({"query": pergunta})
    print(f"\n Hummm acredito que : {resposta}")

