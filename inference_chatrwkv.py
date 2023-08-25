#!pip install ninja tokenizers

#!git clone https://github.com/BlinkDL/ChatRWKV

'''#@title Select/Download Model { display-mode: "form" }
import urllib

#@markdown Select the model you'd like to use:
model_file = "/content/drive/MyDrive/Copy of rwkv-12.pth" #@param {type:"string"}
#@markdown It will first search `model_dir` for `model_file`.
#@markdown If it isn't valid path, it will attempt to download a `RWKV-v4-Raven` model from huggingface.
#@markdown To see which options you have, take a look at the [repo](https://huggingface.co/BlinkDL/rwkv-4-raven/).

#@markdown ---

#@markdown For example:
#@markdown - RWKV-v4-Raven-14B-v11x: `RWKV-4-Raven-14B-v11x-Eng99%-Other1%-20230501-ctx8192.pth`
#@markdown - RWKV-v4-Raven-7B-v11x: `RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192.pth`
#@markdown - RWKV-v4-Raven-3B-v11: `RWKV-4-Raven-3B-v11-Eng99%-Other1%-20230425-ctx4096.pth`
#@markdown - RWKV-v4-Raven-1B5-v11: `RWKV-4-Raven-1B5-v11-Eng99%-Other1%-20230425-ctx4096.pth`
#@markdown - Custom Model: `/rwkv-subdirectory/custom-rwkv.pth`

model_path = f"{model_dir_path}/{model_file}"
if not os.path.exists(model_path):
    model_repo = f"https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main"
    model_url = f"{model_repo}/{urllib.parse.quote_plus(model_file)}"
    try:
        print(f"Downloading '{model_file}' from {model_url} this may take a while")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Using {model_path} as base")
    except Exception as e:
        print(f"Model '{model_file}' doesn't exist")
        raise Exception
else:
    print(f"Using {model_path} as base")'''

#@title Select/Download Model { display-mode: "form" }
import urllib
import os
#@markdown Select the model you'd like to use:
model_file = "C:\\Users\\Admin\\PycharmProjects\\character-embbeding\\rwkv-12.pth" #@param {type:"string"}
#@markdown It will first search `model_dir` for `model_file`.
#@markdown If it isn't valid path, it will attempt to download a `RWKV-v4-Raven` model from huggingface.
#@markdown To see which options you have, take a look at the [repo](https://huggingface.co/BlinkDL/rwkv-4-raven/).

#@markdown ---

#@markdown For example:
#@markdown - RWKV-v4-Raven-14B-v11x: `RWKV-4-Raven-14B-v11x-Eng99%-Other1%-20230501-ctx8192.pth`
#@markdown - RWKV-v4-Raven-7B-v11x: `RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192.pth`
#@markdown - RWKV-v4-Raven-3B-v11: `RWKV-4-Raven-3B-v11-Eng99%-Other1%-20230425-ctx4096.pth`
#@markdown - RWKV-v4-Raven-1B5-v11: `RWKV-4-Raven-1B5-v11-Eng99%-Other1%-20230425-ctx4096.pth`
#@markdown - Custom Model: `/rwkv-subdirectory/custom-rwkv.pth`

'''model_path = f"{model_dir_path}/{model_file}"
if not os.path.exists(model_path):
    model_repo = f"https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main"
    model_url = f"{model_repo}/{urllib.parse.quote_plus(model_file)}"
    try:
        print(f"Downloading '{model_file}' from {model_url} this may take a while")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Using {model_path} as base")
    except Exception as e:
        print(f"Model '{model_file}' doesn't exist")
        raise Exception
else:
    print(f"Using {model_path} as base")'''
model_path = model_file

if not os.path.exists(model_path):
    print(f"Model '{model_file}' doesn't exist")
    raise Exception
else:
    print(f"Using {model_path} as base")

#@title Load Model {"display-mode": "form"}
import os, copy, types, gc, sys
sys.path.append('ChatRWKV/rwkv_pip_package/src')

import numpy as np
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

print('ChatRWKV v4 https://github.com/BlinkDL/ChatRWKV')

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

strategy = 'cuda fp16' #@param {"type": "string"}

#@markdown Strategy Examples:
#@markdown - `cpu fp32`
#@markdown - `cuda:0 fp16 -> cuda:1 fp16`
#@markdown - `cuda fp16i8 *10 -> cuda fp16`
#@markdown - `cuda fp16i8`
#@markdown - `cuda fp16i8 -> cpu fp32 *10`
#@markdown - `cuda fp16i8 *10+`

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 256

CHUNK_LEN = 256 # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

print(f'Loading model - {model_path}')
model = RWKV(model=model_path, strategy=strategy)
pipeline = PIPELINE(model, "ChatRWKV/v2/20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187
END_OF_LINE_DOUBLE = 535
# pipeline = PIPELINE(model, "cl100k_base")
# END_OF_TEXT = 100257
# END_OF_LINE = 198

model_tokens = []
model_state = None

AVOID_REPEAT = '，：？！'
AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    out[END_OF_LINE] += newline_adj # adjust \n probability

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

# Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end
def fix_tokens(tokens):
    if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens

#@title Inference Setup {"display-mode": "form"}
#@markdown Inference properties:
temp = 0.5 #@param {"type": "number"}
top_p = 0.7 #@param {"type": "number"}
presence_penalty = 0.2 #@param {"type": "number"}
frequency_penalty = 0.2 #@param {"type": "number"}
# Run inference
from prompt_toolkit import prompt

PROMPT_FILE = 'ChatRWKV/v2/prompt/default/English-2.py'

def load_prompt(PROMPT_FILE):
    variables = {}
    with open(PROMPT_FILE, 'rb') as file:
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    user, bot, interface, init_prompt = variables['user'], variables['bot'], variables['interface'], variables['init_prompt']
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
    return user, bot, interface, init_prompt

user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
save_all_stat('', 'chat_init', out)
gc.collect()
torch.cuda.empty_cache()

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

def on_message(message):
    global model_tokens, model_state, user, bot, interface, init_prompt

    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()

    x_temp = temp
    x_top_p = top_p
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    msg = msg.strip()

    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    # use '+prompt {path}' to load a new prompt
    elif msg[:8].lower() == '+prompt ':
        print("Loading prompt...")
        try:
            PROMPT_FILE = msg[8:].strip()
            user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
            out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
            save_all_stat(srv, 'chat', out)
            print("Prompt set up.")
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Path error.")

    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:3].lower() == '+i ':
            msg = msg[3:].strip().replace('\r\n','\n').replace('\n\n','\n')
            new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg}

# Response:
'''
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')

            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+++':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '++':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        occurrence = {}
        for i in range(FREE_GEN_LEN+100):
            for n in occurrence:
                out[n] -= (presence_penalty + occurrence[n] * frequency_penalty)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token == END_OF_TEXT:
                break
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            if msg[:4].lower() == '+qa ':# or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
        print('\n')
        # send_msg = pipeline.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            msg = msg.strip().replace('\r\n','\n').replace('\n\n','\n')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25) # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (presence_penalty + occurrence[n] * frequency_penalty)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            # if token == END_OF_TEXT:
            #     break
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            out = run_rnn([token], newline_adj=newline_adj)
            out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1

            send_msg = pipeline.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break

        save_all_stat(srv, 'chat', out)

#@title Chat {"display-mode": "form"}

#@markdown Running this cell will start the chat. Simply type your message in the input

#@markdown Commands:
#@markdown - `+` to get an alternate chat reply
#@markdown - `+reset` to reset the chat
#@markdown - `+gen YOUR PROMPT` for a free single-round generation with any prompt
#@markdown - `+i YOUR INSTRUCT` for a free single-round generation with any instruct
#@markdown - `+++` to continue the last free generation (only for `+gen` / `+i`)
#@markdown - `++` to retry the last free generation (only for `+gen` / `+i`)

#@markdown Remember to `+reset` periodically to clean up the bot's memory.

while True:
    msg = input("Bob: ")
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Error: please say something')
