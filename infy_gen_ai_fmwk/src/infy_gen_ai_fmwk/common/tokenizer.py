# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from transformers import GPT2TokenizerFast
from spacy.lang.en import English
from infy_gen_ai_fmwk.common.pypdf import read_pdf_from_file


class Tokenizer:
    def __init__(self, vocab_dir_path, config_data):
        self.vocab_dir_path = vocab_dir_path
        self.config_data = config_data

    def chunkify(self, text):
        token_chunks = list()
        nlp = English()
        nlp.add_pipe('sentencizer')
        doc = nlp(text)
        local_chunk = ''
        token_count = 0
        for sent in doc.sents:

            sent_tokens = self.tokenize(str(sent))
            # A very rare case when a single sentence exceeds the CHUNK_LENGTH.
            if len(sent_tokens) < self.config_data.chunk_length:
                if token_count + len(sent_tokens) > self.config_data.chunk_length:
                    token_chunks.append(local_chunk)
                    # local_chunk = [].extend(sent_tokens)
                    local_chunk = '' + str(sent)
                    token_count = len(sent_tokens)
                else:
                    # local_chunk.extend(sent_tokens)
                    local_chunk += str(sent)
                    token_count += len(sent_tokens)

        if len(local_chunk) > 0:
            token_chunks.append(local_chunk)

        return token_chunks

    def tokenize(self, page_text_chunk):
        tokenizer = GPT2TokenizerFast.from_pretrained(
            self.config_data.vocab_dir_path)
        tokens = tokenizer(page_text_chunk)
        tokens = tokens['input_ids']
        return [tokenizer.decode(x) for x in tokens]

    def tokenize_pdf_from_file(self, file):
        pdf_pages = read_pdf_from_file(file)
        token_chunks = list()

        for page in pdf_pages:
            page_text_chunks = self.chunkify(page)
            for text_chunk in page_text_chunks:
                assert len(self.tokenize(text_chunk)
                           ) <= self.config_data.token_group_size, 'The tokens in this text chunk exceeds 2048!!'
                token_chunks.append(text_chunk)

        return token_chunks
