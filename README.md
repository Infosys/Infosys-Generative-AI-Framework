[
 Copyright 2024 Infosys Ltd.                                                                                    
 Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  
http://www.apache.org/licenses/                                                                           
]:#
# Infosys Generative AI Framework 3.0.0

Infosys Generative AI Framework is a python library that provides various APIs listed below

## Prerequisites 

- Python =3.10

## APIs

The details of each API and its core functionality is given below. For more details, please read the [docs](docs).

S# | API	| Description | 
---|-------|---------------|
1 | audio to text | This API is used for converting audio files to text. It uses the OpenAI API to perform the conversion.
2 | code translate | This API is used for translating code from one programming language to another. It uses the OpenAI API for the translation process.
3 | single code documentation | This API is used for generating documentation for a given source code file. It uses the OpenAI API to generate the documentation based on the content of the source code.
4 | unit test generation | This API is used for automatically generating unit tests for a given source code file. It uses the OpenAI API to generate the tests based on the content of the source code.
5 | multiple code documentation |  This API is used for generating documentation for multiple source code files at once. It uses the OpenAI API to generate the documentation based on the content of each source code file.
6 | summarize PDF document | This API is used for summarizing the content of PDF files. It uses the OpenAI API to generate the summary based on the content of the PDF.
7 | generate insights | This API is used for generating insights from given data. It uses the OpenAI API to generate the insights based on the content of the data.
8 | generate metadata and description | This API is used for generating meta descriptions from a text file. It uses the OpenAI API to generate the meta descriptions based on the content of the text file.
9 | add searchable embeddings | This API is used for indexing the pdf documents to vectordb (eg:chromadb). It leverages OpenAI (text-embedding-ada-002) and OpenSource (all-MiniLM-L6-v2) models.
10 | retrieve context |  This API is used for generating closest matches from the embeddings stored in vectordb (eg:chromadb) based on the input query. It leverages OpenAI (text-embedding-ada-002) and OpenSource (all-MiniLM-L6-v2) models.
11 | generate answer  | This API is used for generating answers from the embeddings stored in vectordb (eg:chromadb) based on the input query.It leverages OpenAI (text-embedding-ada-002 and gpt-4) and OpenSource (all-MiniLM-L6-v2 and roberta-base-squad2) models.
12 | document reset   | This API is used for deleting the pdf document(s) from vectordb (eg:chromadb). 
13 | retrieve video moments | This API is used for generating the images and video moments from the input video file based on input query. It leverages OpenSource (clip-ViT-B-32) model.
14 | build knowledge graph |  This API is used for building and filtering the Knowledge graph based on the input query. It leverages OpenAI (gpt-4) model.
15 | search knowledge graph |  This API is used for searching the knowledge graph based on the input query. It leverages OpenAI (gpt-4) model.


The API logical input/output is given below. 

Step | API | Input | Output  
---|---|---|---|
1 | audio to text | `audio file path` , `mom_required (bool)`  | `transcripted text`, `mom text (optional)` 
2 | code translate | `source code file path` , `source language` , `target language` | `translated code` 
3 | single code documentation  | `source code file path` , `source language` | `documentation of the code`
4 | unit test generation  | `source code file path` , `source language` | `generated test cases`
5 | multiple code documentation |`source code file paths` , ` source language`  | `documentation of multiple files` 
6 | summarize PDF document| `pdf file path` , `summarization type` | `summary of the pdf file`
7 | generate insights| `csv or xlsx file path` , `number of completions`  | `summary of the file`
8 | generate metadata and description| `text file path` , `number of completions` | `keywords` , `meta description` 
9 | add searchable embeddings | `file paths` , `embedding type` , `vocab dir path` , `vector db directory` | `embeddings added to the vectordb`
10 | retrieve context | `query` , `embedding type` , `top k value` , `vector db directory` | `list of closest matches`
11 | generate answer | `query` , `embedding type` , `vector db directory` , `top k value` , `rag(bool)`| `generated answer`
12 | document reset | `vector db directory` , `collection name` | `status message`
13 |retrieve video moments | `query` , `display results count` , `video file path` , `clip duration` , `output root path` |`output file paths of images and videos `
14 |build knowledge graph | `query` , `mode` , ` root path` , `output file prefix` , `graph json file path (optional)` | `graph output image file path` , `graph output json file path` , `message text`
15 |search knowledge graph | `query` , ` root path` , `output file prefix` , `graph json file path (optional)` | `graph output image file path` , `message text`
 

## Examples

For code examples, please read [docs/notebook](docs/notebook).



## List of models
S# | Model Name	| Type | Dependent API
---|-------------------|-----|----------|
1  | whisper-base      | `OpenSource` | `audio to text`
2  | gpt2              | `OpenSource` | `code translate` , `summarise PDF document` 
3  | all-MiniLM-L6-v2  | `OpenSource` |  `add searchable embeddings` , `retrieve context`
4  | roberta-base-squad2 | `OpenSource` | `generate answer` 
5  | clip-ViT-B-32  | `OpenSource` | `retrieve video moments`
6  | text-embedding-ada-002 | `OpenAI` | `add searchable embeddings` , `retrieve context` , `generate answer`     
7  | gpt-4  | `OpenAI` | `audio to text` , `code translate` ,  `single code documentation` , `unit test generation` , `multiple code documentation` , `summarize PDF document` , `generate insights` , `generate metadata and description` , `generate answer` , `build knowledge graph` , `search knowledge graph`

## Steps to download OpenSource models
 
 
```bash
mkdir C:\MyProgramFiles\AI\models
cd C:\MyProgramFiles\AI\models
git lfs install
#To download 'whisper_base' model,use below command
git clone https://huggingface.co/openai/whisper-base
#To download 'gpt2' model,use below command
git clone https://huggingface.co/openai/gpt2
#To download 'all-MiniLM-L6-v2' model,use below command
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# To download 'roberta-base-squad2' model,use below command
git clone https://huggingface.co/deepset/roberta-base-squad2
# To download 'clip-ViT-B-32' model,use below command
git clone https://huggingface.co/sentence-transformers/clip-ViT-B-32
```

## Steps required for settingup chromadb
Install Microsoft Visual Studio C++ Build Tools >= 14.0 

## The following combinations of the `generate answer` functionality are working:

S# | Rag  | Embedding | Inference 
---|------|-----------|-----------
1  | True | text-embedding-ada-002 | gpt-4
2  | True | all-MiniLM-L6-v2    | roberta-base-squad2 
3  | False | NA | gpt-4
