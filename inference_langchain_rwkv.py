!pip install langchain rwkv ninja

!git clone https://github.com/BlinkDL/RWKV-LM

#@title Select/Download Model { display-mode: "form" }
import urllib
import os
#@markdown Select the model you'd like to use:
model_file = "/content/drive/MyDrive/rwkv-12.pth" #@param {type:"string"}
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

#@title Load Model
import os
os.environ["RWKV_CUDA_ON"] = '1'
os.environ["RWKV_JIT_ON"] = '1'

from langchain.llms import RWKV

strategy = "cuda fp16i8 *20 -> cuda fp16" #@param {"type":"string"}
model = RWKV(model=model_path, strategy=strategy, tokens_path="RWKV-LM/RWKV-v4/20B_tokenizer.json")

#@title Chain
#@markdown A simple chain example. You first create the instruction template, and feed in your prompt as the instruction variable.

from langchain.prompts import PromptTemplate
task = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# Instruction:
{instruction}

# Response:
"""
instruction = "Function to get the count of unique values in a column" #@param {type:"string"}

prompt = PromptTemplate(
    input_variables=["instruction"],
    template=task,
)

from langchain.chains import LLMChain
chain = LLMChain(llm=model, prompt=prompt)

print(chain.run(instruction))

#@markdown Documentation —
#@markdown [PromptTemplate](https://python.langchain.com/en/latest/modules/prompts/prompt_templates/examples/prompt_serialization.html),
#@markdown [LLMChain](https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html)
