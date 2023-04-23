from ctypes import Union
from pydantic import BaseModel
from fastapi import FastAPI
from chat import Chat
from typing import Union
import os
import json

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"


chat_model = Chat("../model/rwkv.pth", "cuda fp16i8")
chat_model.load_model()

with open(f"./pure.json", 'r', encoding='utf-8') as f:
    char = json.loads(f.read())
    chat_model.load_init_prompt(char['user'], char['bot'], char['bot_persona'], char['example_dialogue'])
    f.close()


app = FastAPI()


class chatLog(BaseModel):
    log: str
    top_p: Union[float, None] = 0.7
    temperature: Union[float, None] = 1.5
    presence_penalty: Union[float, None] = 0.3
    frequency_penalty: Union[float, None] = 0.3


@app.get("/")
async def root():
    return "Hello, This is ChatRWKV!"


@app.get("/reset")
async def root():
    chat_model.reset_bot()
    return "OK"


@app.post("/chat/")
async def create_chat(chatLog: chatLog):
    print([chatLog.log, chatLog.top_p, chatLog.temperature, chatLog.presence_penalty, chatLog.frequency_penalty])
    output = chat_model.on_message(
        chatLog.log, chatLog.top_p, chatLog.temperature, chatLog.presence_penalty, chatLog.frequency_penalty)
    print(output)
    return output
