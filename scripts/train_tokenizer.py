import gc
import re

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


#
# datasets
#
def batch_iterator():
    # Emotional Intelligence dataset
    try:
        dataset = load_dataset('Abhaykoul/test', split='train')
        for item in dataset:
            if 'text' in item:
                yield item['text']
            if 'conversation' in item:
                yield item['conversation']
            if 'response' in item:
                yield item['response']
    except Exception as e:
        print(f"Warning: Error loading emotional dataset: {e}")

    # Gen Z style text dataset
    try:
        dataset = load_dataset('JeanKaddour/minipile', split='train')
        for text in dataset['text']:
            # Add Gen Z style markers and emojis
            text = text.replace("!", " 🔥!")
            text = text.replace("?", " 💯?")
            yield text
    except Exception as e:
        print(f"Warning: Error loading minipile dataset: {e}")

    del dataset
    gc.collect()

    # Emotional keywords and slang
    gen_z_tokens = [
        "fr fr", "no cap", "bestie", "fam", "vibing", "lit", "fire",
        "based", "bussin", "sheesh", "bet", "slay", "periodt", "tea",
        "lowkey", "highkey", "hits different", "rent free", "living rent free",
        "main character", "vibe check", "understood the assignment",
        "it's giving", "caught in 4k", "respectfully", "let's gooo",
        "real one", "straight facts", "big mood", "whole mood",
        "energy", "toxic", "sus", "cap", "no cap", "facts", "mood",
        "flex", "glow up", "hits different", "living my best life",
        "rent free", "slaps", "stan", "tea", "understood the assignment",
        "vibe", "we love to see it", "what's the tea", "you ate that",
        "period", "purr", "slay", "sis", "snapped", "spill the tea",
        "stay pressed", "that's the tweet", "this ain't it chief",
        "we move", "word", "yeet", "you did what needed to be done",
        "main character energy", "it's the ___ for me",
    ]

    for token in gen_z_tokens:
        yield token

    # Emotional expressions
    emojis = ["🔥", "💯", "💪", "🤝", "⚡", "👑", "📈", "🎯", "🐾", "🌟", 
              "🧠", "👂", "🗣️", "❤️", "🙌", "✨", "💫", "🌈", "🎮", "🚀",
              "😤", "😎", "🥺", "😌", "😩", "😳", "🥴", "😈", "🤔", "😊"]
    
    for emoji in emojis:
        yield emoji

    # code
    dataset = load_dataset('bigcode/programming-languages-keywords', split='train')

    for row in dataset:
        for n in row['keywords']:
            yield n

    del dataset
    gc.collect()

    # code
    dataset = (
        load_dataset('bigcode/the-stack-smol-xs', data_dir=f'data/{name}', split='train', trust_remote_code=True)
        for name in [
            # 'batchfile' - unsafe
            # 'powershell' - unsafe
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
    )

    for d in dataset:
        for text in d['content']:
            yield text

    del dataset
    gc.collect()

    # math
    dataset = load_dataset('OleehyO/latex-formulas', 'cleaned_formulas', split='train')

    for text in dataset['latex_formula']:
        yield text

    del dataset
    gc.collect()

    # text
    dataset = (
        load_dataset('saillab/taco-datasets', data_dir=data_dir, split='train')
        for data_dir in [
            'multilingual-instruction-tuning-dataset /multilingual-alpaca-52k-gpt-4',
            'multilingual-instruction-tuning-dataset /multilinugal-dolly-15k',
        ]
    )

    for d in dataset:
        for row in d:
            for n in row:
                yield row['instruction'] + '\n' + row['input'] + '\n' + row['output']

    del dataset
    gc.collect()

    # text
    dataset = (
        load_dataset('xu-song/cc100-samples', lang, split='train')
        for lang in [
            'en', 'hr', 'sr', 'ru',
            'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'bn_rom', 'br',
            'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'eo', 'es',
            'et', 'eu', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl',
            'gn', 'gu', 'ha', 'he', 'hi', 'hi_rom', 'ht', 'hu',
            'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km',
            'kn', 'ko', 'ku', 'ky', 'la', 'lg', 'li', 'ln', 'lo', 'lt',
            'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'my_zaw',
            'ne', 'nl', 'no', 'ns', 'om', 'or', 'pa', 'pl', 'ps', 'pt',
            'qu', 'rm', 'ro', 'sa', 'si', 'sc', 'sd', 'sk', 'sl',
            'so', 'sq', 'ss', 'su', 'sv', 'sw', 'ta', 'ta_rom',
            'te', 'te_rom', 'th', 'tl', 'tn', 'tr', 'ug', 'uk', 'ur',
            'ur_rom', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo',
            'zh-Hans', 'zh-Hant', 'zu',
        ]
    )

    for d in dataset:
        for text in d['text']:
            yield text

    del dataset
    gc.collect()


#
# special_tokens
#
# ChatML style special tokens
bos_token = '<s>'
eos_token = '</s>'
unk_token = '<unk>'
pad_token = '</s>' # Make pad token equal to eos_token for this use case

special_tokens = [
    bos_token,
    eos_token,
    unk_token,
    pad_token,
    
    # Core HAI Identity Tokens
    '<|hai|>',           # HAI's identity marker
    '<|human|>',         # Human speaker marker
    '<|im_start|>',   # Start of conversation
    '<|im_end|>',     # End of conversation
    
    # Emotional Intelligence Markers
    # Primary emotions
    '<|joy|>',
    '<|sadness|>',
    '<|anger|>',
    '<|fear|>',
    '<|love|>',
    '<|surprise|>',
    
    # Support and empathy
    '<|empathy|>',
    '<|comfort|>',
    '<|validate|>',
    '<|encourage|>',
    '<|support|>',
    '<|understand|>',
    
    # Gen Z Communication Style
    '<|casual|>',        # Casual tone marker
    '<|slang|>',         # Gen Z slang indicator
    '<|vibe_check|>',    # Vibe assessment
    '<|fr_fr|>',         # "For real for real"
    '<|no_cap|>',        # "No cap" (truth)
    '<|based|>',         # "Based" (agreement)
    
    # Emoji and Expression Markers
    '<|emoji_happy|>',   # 😊
    '<|emoji_sad|>',     # 😢
    '<|emoji_love|>',    # ❤️
    '<|emoji_hype|>',    # 🔥
    '<|emoji_think|>',   # 🤔
    '<|emoji_vibe|>',    # ✨
    
    # Conversation Flow
    '<|thinking|>',      # Internal thought process
    '<|responding|>',    # Preparing response
    '<|clarifying|>',    # Asking for clarification
    '<|reflecting|>',    # Reflecting on conversation
    
    # Technical Assistance
    '<|code_start|>',    # Start of code block
    '<|code_end|>',      # End of code block
    '🤔',       # Explanation marker
    '<|example|>',       # Example marker
    '<|solution|>',      # Solution marker
    
    # Personality Traits
    '<|friendly|>',
    '<|helpful|>',
    '<|creative|>',
    '<|authentic|>',
    '<|respectful|>',
    
    # Context Markers
    '<|personal|>',      # Personal context
    '<|academic|>',      # Academic context
    '<|technical|>',     # Technical context
    '<|social|>',        # Social context
    
    # Function and Tool Integration
    '<|tool_start|>',
    '<|tool_end|>',
    '<|function_call|>',
    '<|function_return|>',
    
    # JSON Schema Essential Tokens
    '"type"',
    '"properties"',
    '"required"',
    '"description"',
    '"items"',
    '"enum"',
    '"string"',
    '"number"',
    '"boolean"',
    '"array"',
    '"object"'
]

# Add byte-level tokens for enhanced character handling
for i in range(256):
    special_tokens.append(f'<|byte_{i:02x}|>')  # More descriptive byte token format

# Reserve tokens for future emotional states and expressions
for i in range(32):
    special_tokens.append(f'<|emotion_{i}|>')
for i in range(32):
    special_tokens.append(f'<|expression_{i}|>')

#
# Tokenizer Configuration
#
def create_tokenizer():
    """Create and configure the tokenizer with optimal settings for emotional intelligence."""
    # Initialize BPE tokenizer with byte fallback for handling unknown characters
    bpe = BPE(
        unk_token=unk_token,  # Use unk_token var
        byte_fallback=True,       # Fallback to bytes for unknown chars
        dropout=0.1              # Add dropout for better generalization
    )
    
    tokenizer = Tokenizer(bpe)
    
    # Configure normalizer for consistent text handling
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Strip(),                # Remove leading/trailing whitespace
        normalizers.Replace(r'\s+', ' '),   # Normalize whitespace
        normalizers.NFKC()                  # Unicode normalization
    ])
    
    # Enhanced pre-tokenizer configuration
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=True,
            use_regex=True
        ),
        pre_tokenizers.Digits(individual_digits=True),  # Split digits for better number handling
        pre_tokenizers.Punctuation()                    # Handle punctuation properly
    ])
    
    # Post-processor for adding special tokens and handling pairs
    tokenizer.post_processor = processors.Sequence([
        processors.ByteLevel(
            add_prefix_space=True,
            trim_offsets=False,
            use_regex=True
        ),
        processors.TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A {eos_token} $B {eos_token}",
            special_tokens=[
                (bos_token, 0),
                (eos_token, 1),
            ],
        ),
    ])
    
    # Configure decoder for proper text reconstruction
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(
            add_prefix_space=True,
            trim_offsets=True,
            use_regex=True
        ),
        decoders.Replace(r'\s+', ' ')  # Clean up whitespace in decoded text
    ])
    
    return tokenizer

# Create and configure tokenizer
tokenizer = create_tokenizer()

# Configure trainer with optimal parameters
trainer = BpeTrainer(
    vocab_size=32768,          # 32K vocab size
    min_frequency=2,           # Lower frequency threshold for better coverage
    special_tokens=special_tokens,
    max_token_length=24,       # Increased for better handling of emojis and expressions
    show_progress=True,        # Show training progress
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # Initialize with byte-level alphabet
)

# Train tokenizer
print("Training tokenizer...")
tokenizer.train_from_iterator(batch_iterator(), trainer)

# Save tokenizer files
print("Saving tokenizer...")
tokenizer.save('../tokenizer.json')
tokenizer.model.save('../')

# Enhanced chat template with emotional intelligence support, now in ChatML format
CHAT_TEMPLATE = """{% for message in messages %}
    {{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\\n'}}
{% endif %}"""

# Create fast tokenizer with enhanced configuration
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    chat_template=CHAT_TEMPLATE,
    bos_token=bos_token,
    eos_token=eos_token,
    unk_token=unk_token,
    pad_token=pad_token,
    mask_token='<|mask|>',
    clean_up_tokenization_spaces=False,  # Set to False for consistent behavior
    additional_special_tokens=special_tokens,
    # Enable truncation and padding
    truncation=True,
    padding=True,
    max_length=2048,
    use_default_system_prompt = False
)

# Save the fast tokenizer
print("Saving fast tokenizer...")
fast_tokenizer.save_pretrained('../')
print("Tokenizer training complete!")


# Print out the config and tokenizer json to show the current config to see it is now chatml config
print(f"Tokenizer Config: {fast_tokenizer.special_tokens_map}")
print(f"Tokenizer json: {fast_tokenizer.to_json()}")