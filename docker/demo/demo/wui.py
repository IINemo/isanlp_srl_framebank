#!/root/.pyenv/shims/python
# -*- coding: utf8 -*-

import cgitb
import cgi
import json
import sys
import os

from convert_srl_to_brat import SrlConverter
from isanlp_srl_framebank.pipeline_default import PipelineDefault


ip_address = os.environ['IP_ADDRESS']
morph_port = int(os.environ['MORPH_PORT'])
syntax_port = int(os.environ['SYNTAX_PORT'])
sem_port = int(os.environ['SEM_PORT'])

bratUrl = '/demo/brat/'

def cgiOutput(data):
    sys.stdout.buffer.write("Content-type: text/html\n\n".encode('utf8'))
    sys.stdout.buffer.write(data.encode('utf8'))


def prepareTemplate(data, text, bratUrl):
    templateFile = open("index.tmpl", "r")
    template = templateFile.read()

    template = template.replace("$____COL_DATA_SYN____", json.dumps(data.get('coll_data', '')))
    template = template.replace("$____DOC_DATA_SYN____", json.dumps(data.get('doc_data', '')))
    template = template.replace("$____COL_DATA_SEM____", json.dumps(data.get('coll_data_sem', '')))
    template = template.replace("$____DOC_DATA_SEM____", json.dumps(data.get('doc_data_sem', '')))

    template = template.replace("$____TEXT____", text)
    template = template.replace("$____BRAT_URL____", bratUrl)

    return template


cgitb.enable()
inputData = cgi.FieldStorage()
text = inputData.getfirst("text")

# DEBUG:
#text = """\"Мама мыла раму.\""""

if not text:
    template = prepareTemplate(data = {},
                               text = "",
                               bratUrl = bratUrl)
    cgiOutput(template)
    exit(0)

ppl_remote = PipelineDefault((ip_address, morph_port),
                             (ip_address, syntax_port),
                             (ip_address, sem_port))

res = ppl_remote(text)
brat_json = SrlConverter()(res)

template = prepareTemplate(brat_json,
                           text,
                           bratUrl)
cgiOutput(template)
