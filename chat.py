import json
import torch
import copy
from prompt_toolkit import prompt
import numpy as np
import os
import types
import gc
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
from rwkv.utils import PIPELINE
from rwkv.model import RWKV

np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class Chat:
    model_tokens = []
    model_state = None
    srv_chat = 'dummy_server'

    model = None
    pipline = None
    model_path = None
    strategy = None
    AVOID_REPEAT_TOKENS = []
    CHUNK_LEN = 256
    END_OF_TEXT = 0
    END_OF_LINE = 187
    CHAT_LEN_SHORT = 40
    CHAT_LEN_LONG = 150
    all_state = {}
    user = "Bob"
    bot = "Alice"
    
    rep = ""
    last_rep = ""

    def __init__(self, model_path, strategy):
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        self.model_path = model_path
        self.strategy = strategy

    def load_model(self):
        self.model = RWKV(model=self.model_path, strategy=self.strategy)
        self.pipeline = PIPELINE(self.model, f"./20B_tokenizer.json")
        AVOID_REPEAT = '，：？！'
        for i in AVOID_REPEAT:
            dd = self.pipeline.encode(i)
            assert len(dd) == 1
            self.AVOID_REPEAT_TOKENS += dd

    def run_rnn(self, tokens, newline_adj=0):
        tokens = [int(x) for x in tokens]
        self.model_tokens += tokens
        while len(tokens) > 0:
            out, self.model_state = self.model.forward(
                tokens[:self.CHUNK_LEN], self.model_state)
            tokens = tokens[self.CHUNK_LEN:]

        out[self.END_OF_LINE] += newline_adj  # adjust \n probability

        if self.model_tokens[-1] in self.AVOID_REPEAT_TOKENS:
            out[self.model_tokens[-1]] = -999999999
        return out

    def save_all_stat(self, srv, name, last_out):
        n = f'{name}_{srv}'
        self.all_state[n] = {}
        self.all_state[n]['out'] = last_out
        self.all_state[n]['rnn'] = copy.deepcopy(self.model_state)
        self.all_state[n]['token'] = copy.deepcopy(self.model_tokens)

    def load_all_stat(self, srv, name):
        n = f'{name}_{srv}'
        self.model_state = copy.deepcopy(self.all_state[n]['rnn'])
        self.model_tokens = copy.deepcopy(self.all_state[n]['token'])
        return self.all_state[n]['out']

    def load_init_prompt(self, user, bot, bot_persona, example_dialogue):
        self.user = user
        self.bot = bot
        self.model_tokens = []
        self.model_state = None
        init_prompt = ""

        if (len(bot_persona) > 0):
            init_prompt += f"The following is a coherent verbose detailed conversation between a Chinese girl named {bot} and her friend {user}. {bot_persona}\n\n"

        if (len(example_dialogue) > 0):
            init_prompt += f"{example_dialogue}".replace(
                '{{user}}', user).replace('{{bot}}', bot)
                
        init_prompt = init_prompt.strip().split('\n')
        for c in range(len(init_prompt)):
            init_prompt[c] = init_prompt[c].strip().strip(
                '\u3000').strip('\r')
        init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'


        out = self.run_rnn(self.pipeline.encode(init_prompt))
        self.save_all_stat('', 'chat_init', out)
        gc.collect()
        torch.cuda.empty_cache()
        self.save_all_stat(self.srv_chat, 'chat', out)

    def reset_bot(self):
        out = self.load_all_stat('', 'chat_init')
        self.save_all_stat(self.srv_chat, 'chat', out)
        return None

    def on_message(self, message, top_p, temperature, presence_penalty, frequency_penalty):
        msg = message.replace('\\n', '\n').strip()
        out = self.load_all_stat(self.srv_chat, 'chat')
        new = f"{self.user}: {msg}\n\n{self.bot}:"
        # new = f" {msg}\n{self.bot}: "
        out = self.run_rnn(self.pipeline.encode(new), newline_adj=-999999999)
        self.save_all_stat(self.srv_chat, 'chat_pre', out)

        self.rep = self.gen_msg(out, top_p, temperature, presence_penalty, frequency_penalty)
        while (self.rep == self.last_rep):
            self.load_all_stat('', 'chat_init')
            out = self.run_rnn(self.pipeline.encode(new), newline_adj=-999999999)
            self.save_all_stat(self.srv_chat, 'chat_pre', out)
            self.rep = self.gen_msg(out, top_p, temperature, presence_penalty, frequency_penalty)

        self.last_rep = self.rep

        return self.rep


    def gen_msg(self, out, top_p, temperature, presence_penalty, frequency_penalty):
        begin = len(self.model_tokens)
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= self.CHAT_LEN_SHORT:
                newline_adj = (i - self.CHAT_LEN_SHORT) / 10
            elif i <= self.CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = min(3, (i - self.CHAT_LEN_LONG)
                                  * 0.25)  # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (presence_penalty + occurrence[n]
                           * frequency_penalty)
            token = self.pipeline.sample_logits(
                out, temperature=temperature, top_p=top_p)
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            out = self.run_rnn([token], newline_adj=newline_adj)
            out[self.END_OF_TEXT] = -999999999  # disable <|endoftext|>

            send_msg: str = self.pipeline.decode(self.model_tokens[begin:])

            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break

            idx = send_msg.find(f'{self.user}:')
            if idx >= 0:
                send_msg = f' {send_msg[:idx].strip()}\n\n'
                tokens = self.pipeline.encode(send_msg)
                self.load_all_stat(self.srv_chat, 'chat_pre')
                out = self.run_rnn(tokens)
                send_msg = send_msg.strip()

            idx = send_msg.find(f'{self.bot}:')
            if idx >= 0:
                send_msg = f' {send_msg[:idx].strip()}\n\n'
                tokens = self.pipeline.encode(send_msg)
                out = self.load_all_stat(self.srv_chat, 'chat_pre')
                out = self.run_rnn(tokens)
                send_msg = send_msg.strip()
        
         

        self.save_all_stat(self.srv_chat, 'chat', out)


        return send_msg
