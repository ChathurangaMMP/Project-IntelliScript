import torch
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def knowleadgebase_create(folder_path):
    file_reader = DirectoryLoader(folder_path, glob="**/*.txt")
    documents = file_reader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=1600, separators=["\n\n", "\n", " ", ""])
    docs = text_splitter.split_documents(documents)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cpu'})

    knowledgeBase = FAISS.from_documents(docs, embeddings)

    knowledgeBase.save_local(DB_FAISS_PATH)

    return knowledgeBase


data_source = ""  # input text files
DB_FAISS_PATH = ''  # output vectorstore

knowledgeBase = knowleadgebase_create(data_source)

embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device': 'cpu'})


knowledgeBase_1 = FAISS.load_local(
    DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


# model loading


def retriever(knowledgeBase, query, k):
    retriever_ = knowledgeBase.as_retriever(search_kwargs={"k": k})

    docs = retriever_.invoke(query)
    return docs
