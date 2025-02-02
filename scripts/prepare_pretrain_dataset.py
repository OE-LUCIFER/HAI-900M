import subprocess
import sys
from typing import Optional, Union, Iterator
from functools import partial

def install_requirements():
    try:
        import datasets
        import litdata
        import litgpt
    except ImportError:
        print("Installing requirements.in...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.in"])
        print("Finished installing requirements.in")

install_requirements()

from datasets import load_dataset
from litdata import optimize, TokensLoader
from litgpt.tokenizer import Tokenizer
from litdata import StreamingDataset


def batch_dict_iterator(path: str,
                        name: Optional[str]=None,
                        data_dir: Optional[str]=None,
                        data_files: Optional[str]=None,
                        keep_in_memory: bool=False,
                        revision: Optional[str]=None,
                        split: str='train',
                        num_proc: Optional[int]=None,
                        format: Optional[str]=None) -> Iterator[str]:
    assert isinstance(format, str) or callable(format)

    dataset = load_dataset(path=path,
                           name=name,
                           data_dir=data_dir,
                           data_files=data_files,
                           keep_in_memory=keep_in_memory,
                           revision=revision,
                           split=split,
                           trust_remote_code=True,
                           num_proc=num_proc)

    if callable(format):
        for row in dataset:
            text = format(row)
            yield text
    else:
        for row in dataset:
            text = format.format(**row)
            yield text


def batch_iterator(dataset_config: Union[list, dict]):
    if isinstance(dataset_config, dict):
        for text in batch_dict_iterator(**dataset_config):
            yield text
    elif isinstance(dataset_config, list):
        for dc in dataset_config:
            for text in batch_dict_iterator(**dc):
                yield text
    else:
        raise ValueError('')


def tokenize_fn(dataset_config: Union[dict, list], tokenizer: Optional[Tokenizer]=None):
    assert isinstance(dataset_config, (dict, list))

    for text in batch_iterator(dataset_config):
        text_ids = tokenizer.encode(text, bos=False, eos=True)
        yield text_ids


# Constants
EOS_TOKEN = '</s>'

# Alpaca-style prompt template
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{0}

### Input:
{1}

### Response:
{2}"""

datasets_configs = [
    #
    # Emotional Intelligence Datasets
    #
    # EmotionalIntelligence-75k dataset
    {'path': 'OEvortex/EmotionalIntelligence-75k', 
     'split': 'train', 
     'format': lambda n: PROMPT_TEMPLATE.format(n['instruction'], n.get('input', ''), n['output']) + EOS_TOKEN
    },
    
    # Med-emo dataset for medical emotional intelligence
    {'path': 'OEvortex/Med-emo', 
     'split': 'train', 
     'format': lambda n: PROMPT_TEMPLATE.format(n['instruction'], n.get('input', ''), n['output']) + EOS_TOKEN
    },
    
    # HelpingAI2.5 English emotions dataset
    {'path': 'OEvortex/HelpingAI2.5-English-openemotions', 
     'split': 'train', 
     'format': lambda n: PROMPT_TEMPLATE.format(n['instruction'], n.get('input', ''), n['output']) + EOS_TOKEN
    },
    
    # HelpingAI2.5 Hinglish emotions dataset
    {'path': 'OEvortex/HelpingAI2.5-hinglish-openemotions', 
     'split': 'train', 
     'format': lambda n: PROMPT_TEMPLATE.format(n['instruction'], n.get('input', ''), n['output']) + EOS_TOKEN
    },

    #
    # general knowledge
    #
    # 3.18 GB, 1,010,500 - paper says that extracted is 6GB
    *[
        {'path': 'JeanKaddour/minipile', 'split': f'train[{i}%:{i + 5}%]', 'format': lambda n: n['text']}
        for i in range(0, 100, 5)
    ],
    {'path': 'JeanKaddour/minipile', 'split': 'validation', 'format': lambda n: n['text']},
    {'path': 'JeanKaddour/minipile', 'split': 'test', 'format': lambda n: n['text']},
    # 135 MB, 1,795
    {'path': 'open-phi/textbooks', 'format': lambda n: n['markdown']},
    # 631 MB, 111,048
    {'path': 'open-phi/programming_books_llama', 'format': lambda n: n['markdown']},

    #
    # multilingual text
    #
    ## 138 MB, 205,568
    {'path': 'CohereForAI/aya_dataset', 'format': lambda n: n['inputs']},
    {'path': 'CohereForAI/aya_dataset', 'format': lambda n: n['targets']},
    [
        # 193 MB, 1,141,967
        {'path': 'xu-song/cc100-samples', 'name': name, 'split': 'train', 'format': lambda n: n['text']}
        for name in [
            'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'bn_rom', 'br',
            'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es',
            'et', 'eu', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl',
            'gn', 'gu', 'ha', 'he', 'hi', 'hi_rom', 'hr', 'ht', 'hu',
            'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km',
            'kn', 'ko', 'ku', 'ky', 'la', 'lg', 'li', 'ln', 'lo', 'lt',
            'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'my_zaw',
            'ne', 'nl', 'no', 'ns', 'om', 'or', 'pa', 'pl', 'ps', 'pt',
            'qu', 'rm', 'ro', 'ru', 'sa', 'si', 'sc', 'sd', 'sk', 'sl',
            'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'ta_rom',
            'te', 'te_rom', 'th', 'tl', 'tn', 'tr', 'ug', 'uk', 'ur',
            'ur_rom', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo',
            'zh-Hans', 'zh-Hant', 'zu',
        ]
    ],
    *[
        # ~3 GB, 4,976,850
        # {'path': 'saillab/taco-datasets', 'data_dir': name, 'split': 'train', 'format': '{instruction} {input} {output}'}
        {'path': 'saillab/taco-datasets', 'data_dir': name, 'split': 'train', 'format': lambda n: n['output']}
        for name in [
            # 'multilingual-instruction-tuning-dataset /multilingual-alpaca-52k-gpt-4',
            'multilingual-instruction-tuning-dataset /multilinugal-dolly-15k',
        ]
    ],

    #
    # general knowledge
    #
    ## ~17.6 GB, ~6.41M rows
    # [
    #     {'path': 'wikimedia/wikipedia', 'name': '20231101.en', 'split': f'train[{i}%:{i + 20}%]', 'format': lambda n: n['text']}
    #     for i in range(0, 100, 20)
    # ],
    ## 2.89 GB, 430,000, English September of 2017
    # [
    #     {'path': 'jordiclive/wikipedia-summary-dataset', 'split': f'train[{i}%:{i + 20}%]', 'format': lambda n: n['summary']}
    #     for i in range(0, 100, 20)
    # ],
    # 65.1 MB, 7,819
    {'path': 'Sketched33/Cities_Wikipedia_Information', 'format': lambda n: n['wikipedia_content']},

    #
    # misc
    #
    # 472 KB, 5,034
    {'path': 'badrex/llm-emoji-dataset', 'format': '{character} {unicode} {short description} {tags} {LLM description}'},

    #
    # math
    #
    ## 2.87 GB, 552,000 - images/text - we use only latex text, top 10%
    # {'path': 'OleehyO/latex-formulas', 'data_dir': 'cleaned_formulas', 'split': 'train[:10%]', 'format': lambda n: n['latex_formula']},
    ## 12.2 MB, 500,000
    # {'path': 'fblgit/simple-math', 'revision': 'refs/convert/parquet', 'split': 'train+test', 'format': '{instruction} = {output}'},
    ## 125 MB, 1,000,000
    # {'path': 'Gusarich/math-expressions-1m', 'revision': 'refs/convert/parquet', 'split': 'train', 'format': '{expression} = {result}'},
    ## 3.49 GB, 22,259,474
    # [
    #     {'path': 'AtlasUnified/atlas-math-sets', 'split': f'train[{i}%:{i + 20}%]+validation+test', 'format': '{instruction} . {output}'}
    #     for i in range(0, 100, 20)
    # ],
    ## 9.05 GB, 2,583,257 - unsafe
    # [
    #     {'path': 'gair-prox/open-web-math-pro', 'split': f'train[{i}%:{i + 20}%]', 'format': lambda n: n['text']}
    #     for i in range(0, 100, 20)
    # ],
    ## 12.6 GB, 21,972,791 - we use 1M subset - 639 MB, 1,000,000
    # [
    #     {'path': 'nvidia/OpenMathInstruct-2', 'split': f'train_1M[{i}%:{i + 20}%]', 'format': '{problem} {generated_solution} {expected_answer}'}
    #     for i in range(0, 100, 20)
    # ],

    #
    # stem
    #
    ## 1.44 GB, 63,357
    # [
    #     {'path': 'neuralwork/arxiver', 'split': f'train[{i}%:{i + 20}%]', 'format': lambda n: n['markdown']}
    #     for i in range(0, 100, 20)
    # ],

    #
    # code
    #
    # [
    #     # 1.73 GB, 541,041
    #     {'path': 'bigcode/the-stack-smol-xl', 'data_dir': f'data/{name}', 'format': lambda n: n['content']}
    #     for name in [
    #         # 'batchfile' - unsafe
    #         # 'powershell' - unsafe
    #         'ada', 'agda', 'alloy', 'antlr', 'applescript', 'assembly',
    #         'augeas', 'awk', 'bison', 'bluespec', 'c',
    #         'c++', 'c-sharp', 'clojure', 'cmake', 'coffeescript', 'common-lisp',
    #         'css', 'cuda', 'dart', 'dockerfile', 'elixir',
    #         'elm', 'emacs-lisp','erlang', 'f-sharp', 'fortran', 'glsl', 'go',
    #         'groovy', 'haskell','html', 'idris', 'isabelle', 'java',
    #         'java-server-pages', 'javascript', 'julia', 'kotlin', 'lean',
    #         'literate-agda', 'literate-coffeescript', 'literate-haskell',
    #         'lua', 'makefile', 'maple', 'markdown', 'mathematica', 'matlab',
    #         'ocaml', 'pascal', 'perl', 'php', 'prolog',
    #         'protocol-buffer', 'python', 'r', 'racket', 'restructuredtext',
    #         'rmarkdown', 'ruby', 'rust', 'sas', 'scala', 'scheme',
    #         'shell', 'smalltalk', 'solidity', 'sparql', 'sql', 'stan',
    #         'standard-ml', 'stata', 'systemverilog', 'tcl', 'tcsh', 'tex',
    #         'thrift', 'typescript', 'verilog', 'vhdl', 'visual-basic', 'xslt',
    #         'yacc', 'zig',
    #     ]
    # ],
    [
        # 102 MB, 8,700
        {'path': 'bigcode/the-stack-smol-xs', 'data_dir': f'data/{name}', 'format': lambda n: n['content']}
        for name in [
            'batchfile',
            'powershell',
            'ada', 'agda', 'alloy', 'antlr', 'applescript', 'assembly',
            'augeas', 'awk', 'bison', 'bluespec', 'c',
            'c++', 'c-sharp', 'clojure', 'cmake', 'coffeescript', 'common-lisp',
            'css', 'cuda', 'dart', 'dockerfile', 'elixir',
            'elm', 'emacs-lisp','erlang', 'f-sharp', 'fortran', 'glsl', 'go',
            'groovy', 'haskell','html', 'idris', 'isabelle', 'java',
            'java-server-pages', 'javascript', 'julia', 'kotlin', 'lean',
            'literate-agda', 'literate-coffeescript', 'literate-haskell',
            'lua', 'makefile', 'maple', 'markdown', 'mathematica', 'matlab',
            'ocaml', 'pascal', 'perl', 'php', 'prolog',
            'protocol-buffer', 'python', 'r', 'racket', 'restructuredtext',
            'rmarkdown', 'ruby', 'rust', 'sas', 'scala', 'scheme',
            'shell', 'smalltalk', 'solidity', 'sparql', 'sql', 'stan',
            'standard-ml', 'stata', 'systemverilog', 'tcl', 'tcsh', 'tex',
            'thrift', 'typescript', 'verilog', 'vhdl', 'visual-basic', 'xslt',
            'yacc', 'zig',
        ]
    ],
    ## 1.62 GB, 1,632,309
    # {'path': 'nampdn-ai/tiny-codes', 'format': lambda n: n['response']},
    ## 7.81 GB, ~2,804,025
    # [
    #     {'path': 'rombodawg/code_bagel_hermes-2.5', 'split': f'train[{i}%:{i + 20}%]', 'format': '{input} {output}'}
    #     for i in range(0, 100, 20)
    # ],
    ## 6.61 GB, ~2,646,394
    # [
    #     {'path': 'rombodawg/code_bagel', 'split': f'train[{i}%:{i + 20}%]', 'format': '{input} {output}'}
    #     for i in range(0, 100, 20)
    # ],
]

outputs = optimize(
    fn=partial(tokenize_fn, tokenizer=Tokenizer('..')),
    inputs=datasets_configs,
    output_dir='../pretrain-data/',
    # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
    chunk_size=(2049 * 8000), # 2048 + 1
    num_workers=32,
    reorder_files=False,

    # NOTE: this is only available in newver versions of litdata which current version of litgpt does not use
    #
    # This is important to inform LitData that we are encoding contiguous 1D array (tokens).
    # LitData skips storing metadata for each sample e.g all the tokens are concatenated to form one large tensor.
    # item_loader=TokensLoader(block_size=8193),
)

#
# total number of chunks
#
dataset = StreamingDataset(
  input_dir='../pretrain-data/',
  item_loader=TokensLoader(block_size=2049), # 2048 + 1
)

print(len(dataset))
