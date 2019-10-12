from flask import request
from flask import jsonify
from flask import Flask
import json
import os

from isanlp_srl_framebank.pipeline_default import PipelineDefault

class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        return json.dumps(obj, default=lambda x: x.__dict__, ensure_ascii=False)

app = Flask(__name__)
app.json_encoder = CustomEncoder

MORPH_ADDR, MORPH_PORT = os.environ['MORPH_ADDRESS'].split(':')
SYNTAX_ADDR, SYNTAX_PORT = os.environ['SYNTAXNET_ADDRESS'].split(':')
SRL_ADDR, SRL_PORT = os.environ['SRL_ADDRESS'].split(':')

ppl = PipelineDefault(address_morph=(MORPH_ADDR, int(MORPH_PORT)),
                      address_syntax=(SYNTAX_ADDR, int(SYNTAX_PORT)),
                      address_srl=(SRL_ADDR, int(SRL_PORT)))

@app.route('/api/srl', methods=['POST'])
def text_qa():
    data = json.loads(request.data)
    text = data.get('text', None)
    if not text:
        raise ValueError("invalid parameters")
    answer = ppl(text)
    return jsonify(answer)

@app.errorhandler(ValueError)
def handle_bad_request(e):
    return 'invalid request parameters!', 400

if __name__ == '__main__':
   app.run('0.0.0.0', 3333, debug=False)
