import subprocess
import sys
from typing import Optional, Union, Callable, Iterator
from collections.abc import Collection
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
from cognition_dataset import self_cognition_messages

# Constants for HAI's communication style
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
HAI_ROLES = {
    'system': 'system',
    'user': 'user',
    'human': 'user',
    'assistant': 'assistant',  # Keep assistant role consistent
    'gpt': 'assistant',
    'AI': 'assistant',
    'hai': 'assistant', # Keep HAI as the assistant
}

# Message formatting template for ChatML style
HAI_MESSAGE_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"

def format_hai_message(role: str, content: str) -> str:
    """Format a message in ChatML style with appropriate markers."""
    return HAI_MESSAGE_TEMPLATE.format(
        role=HAI_ROLES.get(role.lower(), 'assistant'),
        content=content
    )

def add_hai_style(content: str) -> str:
    """Add HAI's Gen Z style to responses where appropriate."""
    if 'hai' in content.lower() or 'assistant' in content.lower():
        # Add emojis and Gen Z style for HAI's responses
        content = content.replace('I am', "I'm")
        content = content.replace('cannot', "can't")
        content = content.replace('will not', "won't")
        if not any(emoji in content for emoji in ['🔥', '💯', '💪', '📈']):
            content += ' 💯'
    return content

def batch_dict_iterator(path: Optional[str]=None,
                        name: Optional[str]=None,
                        data: Optional[Collection]=None,
                        data_dir: Optional[str]=None,
                        data_files: Optional[str]=None,
                        keep_in_memory: bool=False,
                        revision: Optional[str]=None,
                        split: str='train',
                        num_proc: Optional[int]=None,
                        field: Optional[str]=None,
                        transform: Optional[Callable]=None) -> Iterator[str]:
    """Iterator for processing datasets in HAI's format."""
    if path and not data:
        data = load_dataset(path=path,
                            name=name,
                            data_dir=data_dir,
                            data_files=data_files,
                            keep_in_memory=keep_in_memory,
                            revision=revision,
                            split=split,
                            trust_remote_code=True,
                            num_proc=num_proc)

    if data and field:
        data = data[field]

    if transform:
        data = [transform(n) for n in data]

    for n in data:
        text: list[str] = []
        for m in n:
            role = m.get('role', 'assistant')  # Default to assistant role
            content = add_hai_style(m.get('content', ''))
            formatted_msg = format_hai_message(role, content)
            text.append(formatted_msg)

        text = ''.join(text) #join without \n since each message contains its own
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
        text_ids = tokenizer.encode(text, bos=False, eos=True) # No Bos token here since it already in CHATML format
        yield text_ids


datasets_configs = [
    #
    # HAI's Self-Cognition
    #
    {'path': None, 'field': None, 'data': self_cognition_messages, 'transform': lambda r: [
        {'role': 'user', 'content': r['input']},
        {'role': 'assistant', 'content': r['output']},  # Using HAI's role
    ]},

    #
    # Emotional Intelligence Datasets
    #
    # EmotionalIntelligence-75k dataset
    {'path': 'OEvortex/EmotionalIntelligence-75k', 
     'split': 'train', 
     'transform': lambda r: [
        {'role': 'user', 'content': r['instruction']},
        {'role': 'assistant', 'content': add_hai_style(r['output'])},
    ]},
    
    # Med-emo dataset
    {'path': 'OEvortex/Med-emo', 
     'split': 'train', 
     'transform': lambda r: [
        {'role': 'user', 'content': r['instruction']},
        {'role': 'assistant', 'content': add_hai_style(r['output'])},
    ]},
    
    # HelpingAI2.5 English emotions
    {'path': 'OEvortex/HelpingAI2.5-English-openemotions', 
     'split': 'train', 
     'transform': lambda r: [
        {'role': 'user', 'content': r['instruction']},
        {'role': 'assistant', 'content': add_hai_style(r['output'])},
    ]},
    
    # HelpingAI2.5 Hinglish emotions
    {'path': 'OEvortex/HelpingAI2.5-hinglish-openemotions', 
     'split': 'train', 
     'transform': lambda r: [
        {'role': 'user', 'content': r['instruction']},
        {'role': 'assistant', 'content': add_hai_style(r['output'])},
    ]},

    #
    # general instructs
    #
    # arcee-ai/The-Tome - 4.58 GB, 1,752,473
    # - arcee-ai/infini-instruct-top-500k (BAAI/Infinity-Instruct)
    # - TIGER-Lab/WebInstructSub (top-500k) - IGNORE
    # - jondurbin/airoboros-3.2
    # - gardner/glaive-function-calling-v2-sharegpt
    # - arcee-ai/reasoning-sharegpt (SkunkworksAI/reasoning-0.01)
    # - arcee-ai/self-instruct-sharegpt (bigcode/self-oss-instruct-sc2-exec-filter-50k)
    # - cognitivecomputations/ultrainteract_trajectories_sharegpt
    # - cognitivecomputations/SystemChat-2.0
    # - arcee-ai/qwen2-72b-magpie-en
    [
        {'path': 'arcee-ai/The-Tome', 'split': f'train[{i}%:{i + 20}%]', 'field': 'conversations', 'transform': lambda msgs: [
            {'role': HAI_ROLES[m['from']], 'content': m['value']}
            for m in msgs
        ]}
        for i in range(0, 100, 20)
    ],
    # rombodawg/Everything_Instruct_Multilingual - 2.48 GB, 5,808,694
    # Science:
    #     antiven0m/physical-reasoning-dpoScience
    #     LawalAfeez/science-dataset
    # Social media:
    #     Kyle1668/AG-Tweets
    #     euclaise/reddit-instruct-curated
    # General Knowledge:
    #     NousResearch/CharacterCodex_Characters
    #     jstet/quotes-500k_Famous_Quotes
    #     FronkonGames/steam-games-dataset_Video_Games
    #     totuta_youtube_subs_howto100M_HowTo
    # Multi-lingual:
    #     Amani27/massive_translation_dataset
    #     udmurtNLP/udmurt-russian-english-labse
    #     grosenthal/latin_english
    #     msarmi9/korean-english-multitarget-ted-talks-task
    #     HaiderSultanArc/MT-Urdu-English_Translate
    #     Garsa3112/ChineseEnglishTranslationDataset
    # Cooking:
    #     andrewsiah/se_cooking_preference_sft
    #     Hieu-Phamkaggle/food_recipes
    # Writing:
    #     shahules786/PoetryFoundationData
    #     euclaise/writingprompts
    #     qwedsacf/ivypanda-essaysEssay
    # Medicine:
    #     keivalya/MedQuad-MedicalQnADataset
    #     nuvocare/MSD
    # History:
    #     ambrosfitz10k/history_data_v4
    # Law:
    #     dzunggg/legal-qa-v1
    # Role-Play:
    #     roleplay4/fun_CoupleRP
    #     Undi95andrijdavid/roleplay-conversation-sharegpt
    # News:
    #     RealTimeData/bbc_news_alltime
    # Coding: (rombodawg/code_bagel)
    #     layoric/tiny-codes-alpaca
    #     glaiveai/glaive-code-assistant-v3
    #     ajibawa-2023/Code-290k-ShareGPT
    #     chargoddard/commitpack-ft-instruct-rated
    #     iamtarun/code_instructions_120k_alpaca
    #     ise-uiuc/Magicoder-Evol-Instruct-110K
    #     cognitivecomputations/dolphin-coder
    #     nickrosh/Evol-Instruct-Code-80k-v1
    #     coseal/CodeUltraFeedback_binarized
    #     CyberNative/Code_Vulnerability_Security_DPO
    # Math: (rombodawg/code_bagel)
    #     TIGER-Lab/MathInstruct
    # Function calling: (rombodawg/code_bagel)
    #     glaiveai/glaive-function-calling-v2
    # General Instruct: (rombodawg/OpenHermes-2.5-Uncensored)
    #     teknium/OpenHermes-2.5
    [
        {'path': 'rombodawg/Everything_Instruct_Multilingual', 'split': f'train[{i}%:{i + 20}%]', 'transform': lambda r: [
            {'role': 'system', 'content': r['instruction']},
            {'role': 'user', 'content': r['input']},
            {'role': 'assistant', 'content': r['output']},
        ]}
        for i in range(0, 100, 20)
    ],

    # mlabonne/open-perfectblend - 1.48 GB, 1,420,909
    #   meta-math/MetaMathQA 	395,000
    #   openbmb/UltraInteract_sft 	288,579
    #   HuggingFaceH4/ultrachat_200k 	207,865
    #   microsoft/orca-math-word-problems-200k 	200,035
    #   HuggingFaceH4/ultrafeedback_binarized 	187,405
    #   theblackcat102/evol-codealpaca-v1 	111,272
    #   Post-training-Data-Flywheel/AutoIF-instruct-61k 	61,492
    #   mlabonne/lmsys-arena-human-preference-55k-sharegpt 	57,362
    [
        {'path': 'mlabonne/open-perfectblend', 'split': f'train[{i}%:{i + 20}%]', 'field': 'conversations', 'transform': lambda msgs: [
            {'role': HAI_ROLES[m['from']], 'content': m['value']}
            for m in msgs
        ]}
        for i in range(0, 100, 20)
    ],

    #
    # math
    #
    ## 6.07 GB, 11,402,286
    # [
    #     {'path': 'ai2-adapt-dev/openmath-2-math', 'split': f'train[{i}%:{i + 10}%]', 'field': 'messages'}
    #     for i in range(0, 100, 10)
    # ],
    # 912 MB, 2,570,505
    [
        {'path': 'ai2-adapt-dev/openmath-2-gsm8k', 'split': f'train[{i}%:{i + 10}%]', 'field': 'messages'}
        for i in range(0, 100, 10)
    ],

    #
    # tool/function calling
    #
    # 65.7 MB, 11,578
    {'path': 'NousResearch/hermes-function-calling-v1', 'field': 'conversations', 'transform': lambda msgs: [
        {'role': HAI_ROLES[m['from']], 'content': m['value']}
        for m in msgs
    ]},

    #
    # agent
    #
    # 1.51 GB, 485,874
    [
        {'path': 'arcee-ai/agent-data', 'split': f'train[{i}%:{i + 20}%]', 'field': 'conversations', 'transform': lambda msgs: [
            {'role': HAI_ROLES[m['from']], 'content': m['value']}
            for m in msgs
        ]}
        for i in range(0, 100, 20)
    ],

    #
    # general reasoning
    #
    [
        # 10.8 MB, 15,770
        # {'path': 'AtlasUnified/Atlas-Reasoning', 'data_files': 'reasoning.csv', 'format': '{Prompt} {Step-by-step reasoning} {Solution}'},
        {'path': 'AtlasUnified/Atlas-Reasoning', 'data_files': 'reasoning.csv', 'transform': lambda r: [
            {'role': 'user', 'content': r['Prompt']},
            {'role': 'assistant', 'content': r['Step-by-step reasoning'] + '\n' + r['Solution']},
        ]},
    ],

    #
    # math reasoning
    #
    [
        # 8.99 MB, 6,914
        # {'path': 'thesven/gsm8k-reasoning', 'format': '{question} {generation} {answer} {short_answer}'},
        {'path': 'thesven/gsm8k-reasoning', 'transform': lambda r: [
            {'role': 'user', 'content': r['question']},
            {'role': 'assistant', 'content': r['generation'] + '\n' + r['answer'] + '\n' + r['short_answer']},
        ]},

        # 1.79 MB, 3,963
        # {'path': 'AlgorithmicResearchGroup/math_reasoning_autoformalization_track', 'format': '{informal_statement} {informal_proof} {formal_proof}'},
        {'path': 'AlgorithmicResearchGroup/math_reasoning_autoformalization_track', 'transform': lambda r: [
            {'role': 'user', 'content': r['informal_statement']},
            {'role': 'assistant', 'content': r['informal_proof'] + '\n' + r['formal_proof']},
        ]},

        # 307 MB, 19,944
        # {'path': 'KingNish/reasoning-base-20k', 'format': '{user} {reasoning} {assistant}'},
        {'path': 'KingNish/reasoning-base-20k', 'transform': lambda r: [
            {'role': 'user', 'content': r['user']},
            {'role': 'assistant', 'content': r['reasoning'] + '\n' + r['assistant']},
        ]},

        # 9.45 MB, 10,000
        # {'path': 'Aarushhh/math-reasoning-10k', 'format': '{problem} {plan} {solution}'},
        {'path': 'Aarushhh/math-reasoning-10k', 'transform': lambda r: [
            {'role': 'user', 'content': r['problem']},
            {'role': 'assistant', 'content': r['plan'] + '\n' + r['solution']},
        ]},
    ],

    #
    # code reasoning
    #
    [
        # 56.4 MB, 29,857
        # {'path': 'SkunkworksAI/reasoning-0.01', 'format': '{instruction} {reasoning} {output}'},
        {'path': 'SkunkworksAI/reasoning-0.01', 'transform': lambda r: [
            {'role': 'user', 'content': r['instruction']},
            {'role': 'assistant', 'content': r['reasoning'] + '\n' + r['output']},
        ]},

        # 368 MB, 150,000
        # {'path': 'Magpie-Align/Magpie-Reasoning-150K', 'format': '{instruction} {response}'},
        {'path': 'Magpie-Align/Magpie-Reasoning-150K', 'transform': lambda r: [
            {'role': 'user', 'content': r['instruction']},
            {'role': 'assistant', 'content': r['response']},
        ]},
    ],

    #
    # reflection
    #
    [
        # 4.17 MB, 1,000
        {'path': 'dvilasuero/reflection-v1-gpt-4o-judge', 'transform': lambda r: [
            {'role': 'system', 'content': r['system']},
            {'role': 'user', 'content': r['prompt']},
            {'role': 'assistant', 'content': r['response']},
        ]},
        # 12.4 MB, 3,000
        {'path': 'dvilasuero/reflection-v1-openai-o-mini-judge', 'transform': lambda r: [
            {'role': 'system', 'content': r['system']},
            {'role': 'user', 'content': r['prompt']},
            {'role': 'assistant', 'content': r['response']},
        ]},
        # 70.8 MB, 36,549
        {'path': 'dvilasuero/reflection-v1-final-dedup', 'transform': lambda r: [
            {'role': 'system', 'content': r['system']},
            {'role': 'user', 'content': r['prompt']},
            {'role': 'assistant', 'content': r['response']},
        ]},
        # 30.6 MB, 25,391
        {'path': 'flozi00/reflection-qwen2.5-72b-260924', 'transform': lambda r: [
            r['system'][0],
            {'role': 'user', 'content': r['input']},
            {'role': 'assistant', 'content': r['reflection'] + '\n' + r['output']},
        ]},
        # 26.8 MB, 23,164
        {'path': 'gretelai/synthetic-gsm8k-reflection-405b', 'split': 'train+test', 'transform': lambda r: [
            {'role': 'user', 'content': r['question']},
            {'role': 'assistant', 'content': r['answer_with_tags']},
        ]},
    ],
]

outputs = optimize(
    fn=partial(tokenize_fn, tokenizer=Tokenizer('..')),
    inputs=datasets_configs,
    output_dir='../contrain-data/',
    # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
    chunk_size=(1024 * 16000),
    num_workers=32,
)

#
# total number of chunks
#
dataset = StreamingDataset(
  input_dir='../contrain-data/',
  item_loader=TokensLoader(block_size=1024),
)

print(len(dataset))