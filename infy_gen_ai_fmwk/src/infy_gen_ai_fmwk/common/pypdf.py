# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import PyPDF2

def read_pdf_from_file(file):
    pages = list()
    pdfReader = PyPDF2.PdfReader(file)
    num_pages = len(pdfReader.pages)

    for i in range(num_pages):
        pageObj = pdfReader.pages[i]
        pages.append(pageObj.extract_text())

    return pages
