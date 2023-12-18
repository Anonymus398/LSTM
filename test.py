import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration 
tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B',eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
model

#Prefix <LM>
lm_text='<LM>Принялся Кутузов рассказывать свою историю как он сюда попал. Началось'
input_ids=torch.tensor([tokenizer.encode(lm_text)])
outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,early_stopping=True)
print(tokenizer.decode(outputs[0][1:]))

# print result: с того, что он был в армии, служил в артиллерии</s>.

#Prefix <SC1>
lm_text=str(input("Введите любой текст"))
input_ids=torch.tensor([tokenizer.encode(lm_text)])
outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,early_stopping=True)
print(tokenizer.decode(outputs[0][1:]))
