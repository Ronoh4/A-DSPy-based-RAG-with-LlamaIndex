DSPyRAG README

This repository contains code implementing a RAG (Retriever-Answer-Generator) system using DSPy, a framework for enhancing language model (LM) performance. Below are the key components and functionalities of this code:

Import Modules: The code imports necessary modules and sets up environmental variables for the project.

Configure Models: It configures the language model (GPT-3.5 Turbo) and the retriever model (ColBERTv2) using DSPy.

Parse File and Create Index: Utilizing LlamaParse, the code parses a PDF file and creates a search index using VectorStoreIndex.

Define Signature and Modules: It defines a signature for generating answers and creates a RAG module to handle question answering tasks.

Define Validation Logic and Create Training Set: Validation logic is defined to evaluate predicted answers. Additionally, a training set with example questions and answers is created.

Set up Teleprompter: A BootstrapFewShot teleprompter is set up to compile the RAG module with the training set.

Usage: The compiled RAG module is used to answer questions about the PDF file.

For more details on usage and functionality, refer to the code comments and outputs provided in the repository.
