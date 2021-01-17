import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

device = "cpu"
max_len = 256

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
	"""
	loading pretrained model from hugging face
	"""
    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    model = model.to(device)
    return model

def load_tokenizer():
	"""
	loading tokenizer from hugging face
	"""
    tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')
    return tokenizer

def generate(sentence,num_sentences):
	"""
	Input: Sentence, number of sentences
	Output: A dictionary with key as the input sentence and a list of paraphrased sentences as value
	"""
    result = {}
    final_outputs =[]
    model = load_model()
    tokenizer = load_tokenizer()
    text =  "paraphrase: " + sentence + " </s>"
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=120,
    top_p=0.98,
    early_stopping=True,
    num_return_sequences=num_sentences
    )
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    result[sentence] = final_outputs
    return result
