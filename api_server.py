from fastapi import FastAPI
import re
from pathlib import Path
from pydantic import BaseModel
from graphrag.cli.query import run_local_search
import uvicorn

app = FastAPI(title='Assistant API', version='0.1.0')

def remove_pattern(text):
    pattern = r'\s*\[Data:.*?\]'   
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def remove_text_before_phrase(text, phrase):
    pattern = re.escape(phrase)
    result = re.sub(f'^.*?{pattern}', '', text, flags=re.DOTALL)
    return result.strip()

class InputText(BaseModel):
    question: str


@app.post("/send/")
def send_response(input: InputText):
    root_path = Path('./ragtest')
    text, _ = run_local_search(
        config_filepath=None, data_dir=None, root_dir=root_path,
        query=input.question, community_level=2, response_type='Single Paragraph', streaming=False)
    out = remove_pattern(text)
    update_text = remove_text_before_phrase(out, 'Local Search Response:')
    return {
        'answer': update_text
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9875)