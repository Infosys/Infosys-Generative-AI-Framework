[
 Copyright 2024 Infosys Ltd.                                                                                    
 Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  
http://www.apache.org/licenses/                                                                           
]:#
# Infosys Generative AI Framework 2.0.0

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


The API logical input/output is given below. 

Step | API | Input | Output  
---|---|---|---|
1 | audio to text | `audio file path` ,`mom_required(bool)`  | `translated text`, `mom text(optional)` 
2 | code translate | `source code file path`,`source language`,`target language` | `translated code` 
3 | single code documentation  | `source code file path`, `language` | `documentation of the code`
4 | unit test generation  | `source code file path`, `language` | `generated test cases`
5 | multiple code documentation |`source code file paths`, `language`  | `documentation of multiple files` 
6 | summarize PDF document| `pdf file path`, `summarization type` | `summary of the pdf file`
7 | generate insights| `csv or xlsx file path`, `number of completions`  | `summary of the file`
8 | generate metadata and description| `text file path`, `number of completions` | `keywords`, `meta description` 
 

## Examples

For code examples, please read [docs/notebook](docs/notebook).



## List of models
S# | Model Name	| Dependent API |
---|-------------------|-----|
1  | whisper-base      | `audio to text`
2  | gpt2              | `code translate`, `summarise PDF document`


## Steps to download models
 
 
```bash
mkdir C:\MyProgramFiles\AI\models
cd C:\MyProgramFiles\AI\models
git lfs install
#To download 'whisper_base' model,use below command
git clone https://huggingface.co/openai/whisper-base
#To download 'gpt2' model,use below command
git clone https://huggingface.co/openai/gpt2
```
