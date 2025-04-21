# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from infy_gen_ai_fmwk.common.file_constants import FileConstants
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

class OpenSourceAPI:
    def __init__(self):
        self.rag = None

    def get_chat_completion(self, prompt, model_mapping, completions=1):
        query, context = self.get_query_and_context(prompt)
        completions_model = AutoModelForQuestionAnswering.from_pretrained(
            model_mapping)
        tokenizer_model = AutoTokenizer.from_pretrained(model_mapping)
        nlp = pipeline(FileConstants.PIPELINE_TASK_QA,
                       model=completions_model, tokenizer=tokenizer_model)
        qa_ans = nlp({'question': query, 'context': context})
        return [qa_ans['answer']]
    
    def get_embedding(self, text, engine):
        self.embedding_model = SentenceTransformer(engine)
        embedding = self.embedding_model.encode(text)
        return embedding

    def get_query_and_context(self, prompt):
        if prompt.find('Question : ') < 0:
            return prompt, ''
        question_index = prompt.index('Question :')
        context = prompt[: question_index - 1]
        question = prompt[question_index:]
        return question, context
