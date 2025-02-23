# %% [markdown]
# # **Homework 7 - Bert (Question Answering)**
# 
# If you have any questions, feel free to email us at ntu-ml-2023spring-ta@googlegroups.com
# 
# 
# 
# Slide:    [Link](https://docs.google.com/presentation/d/15lGUmT8NpLGtoxRllRWCJyQEjhR1Idcei63YHsDckPE/edit#slide=id.g21fff4e9af6_0_13)　Kaggle: [Link](https://www.kaggle.com/competitions/ml2023spring-hw7/host/sandbox-submissions)　Data: [Link](https://drive.google.com/file/d/1YU9KZFhQqW92Lw9nNtuUPg0-8uyxluZ7/view?usp=sharing)
# 
# 
# 

# %% [markdown]
# # Prerequisites

# %% [markdown]
# ## Download Dataset

# %%
#!nvidia-smi

# %%
# download link 1
# !gdown --id '1TjoBdNlGBhP_J9C66MOY7ILIrydm7ZCS' --output hw7_data.zip

# download link 2 (if above link failed)
# !gdown --id '1YU9KZFhQqW92Lw9nNtuUPg0-8uyxluZ7' --output hw7_data.zip

# download link 3 (if above link failed)
#!gdown --id '1k2BfGrvhk8QRnr9Xvb04oPIKDr1uWFpa' --output hw7_data.zip

#!unzip -o hw7_data.zip

# For this HW, K80 < P4 < T4 < P100 <= T4(fp16) < V100
#!nvidia-smi

# %% [markdown]
# ## Install packages
# 
# Documentation for the toolkit: 
# *   https://huggingface.co/transformers/
# *   https://huggingface.co/docs/accelerate/index
# 
# 

# %%
# You are allowed to change version of transformers or use other toolkits
#!pip install transformers==4.26.1
#!pip install accelerate==0.16.0

# %% [markdown]
# # Kaggle (Fine-tuning)

# %% [markdown]
# ## Task description
# - Chinese Extractive Question Answering
#   - Input: Paragraph + Question
#   - Output: Answer
# 
# - Objective: Learn how to fine tune a pretrained model on downstream task using transformers
# 
# - Todo
#     - Fine tune a pretrained chinese BERT model
#     - Change hyperparameters (e.g. doc_stride)
#     - Apply linear learning rate decay
#     - Try other pretrained models
#     - Improve preprocessing
#     - Improve postprocessing
# - Training tips
#     - Automatic mixed precision
#     - Gradient accumulation
#     - Ensemble
# 
# - Estimated training time (tesla t4 with automatic mixed precision enabled)
#     - Simple: 8mins
#     - Medium: 8mins
#     - Strong: 25mins
#     - Boss: 2hrs
#   

# %% [markdown]
# ## Import Packages

# %%
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW

from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]       

import random
from torch.utils.data import random_split
class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 60
        self.max_paragraph_len = 150
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 5

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn
        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])
            #print(answer_end_token)
            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2
            ratio=random.random()*0.8
            bias=int(self.max_paragraph_len*ratio)
            bias=max(bias,(answer_end_token -answer_start_token)//2+1)

            #ratio=1

            paragraph_start = max(0, min(mid - bias, len(tokenized_paragraph) - self.max_paragraph_len))
            
            
            #paragraph_start = max(0, min(mid - self.max_paragraph_len // ratio, len(tokenized_paragraph) - self.max_paragraph_len))
            
            
            paragraph_end = paragraph_start + self.max_paragraph_len
            '''
            print("======")
            print(answer_start_token,answer_end_token)
            print(paragraph_start,paragraph_end)
            print("========")
            '''

            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            

            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token,self.questions[idx],paragraph_start,paragraph_end

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask
def evaluate(data, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob and start_index<end_index:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            '''
            if start_index>end_index:
                temp=end_index
                end_index=start_index
                start_index=temp
            '''
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
'''

def evaluate(data, output, doc_stride=150, paragraph=None, paragraph_tokenized=None):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    # index in the whole tokens (not just relative to window)
    entire_start_index = 0
    entire_end_index = 0
    
    for k in range(num_of_windows):
        #print('window',k)
        # Obtain answer by choosing the most probable start position / end position
        mask = data[1][0][k].bool() &  data[2][0][k].bool() # token type & attention mask
        masked_output_start = torch.masked_select(output.start_logits[k], mask)[:-1] # -1 is [SEP]
        start_prob, start_index = torch.max(masked_output_start, dim=0)
        #masked_output_end = torch.masked_select(output.end_logits[k], mask)[start_index:-1] # -1 is [SEP]
        masked_output_end = torch.masked_select(output.end_logits[k], mask)[:-1] # -1 is [SEP]
        end_prob, end_index = torch.max(masked_output_end, dim=0)
        #end_index += start_index 
        

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        masked_data = torch.masked_select(data[0][0][k], mask)[:-1] # -1 is [SEP]

        # Replace answer if calculated probability is larger than previous windows
        if (prob > max_prob) and (end_index - start_index <= 30) and (end_index > start_index):
            max_prob = prob
            entire_start_index = start_index.item() + doc_stride * k
            entire_end_index = end_index.item() + doc_stride * k
            #print('entire_start_index',entire_start_index)
            #print('entire_end_index',entire_end_index)
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(masked_data[start_index : end_index + 1])
            # Remove spaces in answer (e.g. "大 金" --> "大金")
            answer = answer.replace('✔', ' ').replace('✦','\u200b').replace('☺','\u200e').replace('☆','\u3000').replace('●','#').replace(' ','')
def evaluate(data, output, doc_stride=150, paragraph=None, paragraph_tokenized=None):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    # index in the whole tokens (not just relative to window)
    entire_start_index = 0
    entire_end_index = 0
    
    for k in range(num_of_windows):
        #print('window',k)
        # Obtain answer by choosing the most probable start position / end position
        mask = data[1][0][k].bool() &  data[2][0][k].bool() # token type & attention mask
        masked_output_start = torch.masked_select(output.start_logits[k], mask.to('cuda'))[:-1] # -1 is [SEP]
        start_prob, start_index = torch.max(masked_output_start, dim=0)
        #masked_output_end = torch.masked_select(output.end_logits[k], mask)[start_index:-1] # -1 is [SEP]
        masked_output_end = torch.masked_select(output.end_logits[k], mask.to('cuda'))[:-1] # -1 is [SEP]
        end_prob, end_index = torch.max(masked_output_end, dim=0)
        #end_index += start_index 
        

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        masked_data = torch.masked_select(data[0][0][k], mask)[:-1] # -1 is [SEP]

        # Replace answer if calculated probability is larger than previous windows
        if (prob > max_prob) and (end_index - start_index <= 30) and (end_index > start_index):
            max_prob = prob
            entire_start_index = start_index.item() + doc_stride * k
            entire_end_index = end_index.item() + doc_stride * k
            #print('entire_start_index',entire_start_index)
            #print('entire_end_index',entire_end_index)
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(masked_data[start_index : end_index + 1])
            # Remove spaces in answer (e.g. "大 金" --> "大金")
            answer = answer.replace('✔', ' ').replace('✦','\u200b').replace('☺','\u200e').replace('☆','\u3000').replace('●','#').replace(' ','')
    return answer
'''
        
same_seeds(2)
# %% [markdown]
# ## Load Model and Tokenizer
# 
# 
# 
# 
#  

# %%
from transformers import (
  AutoTokenizer,
  AutoModelForQuestionAnswering,
  BertForQuestionAnswering,
)
models=["hfl/chinese-macbert-large","Langboat/mengzi-bert-base","freedomking/mc-bert",
        "hfl/chinese-lert-large","nghuyong/ernie-3.0-base-zh","alibaba-pai/pai-ckbert-large-zh",
        "hfl/chinese-pert-large","hfl/chinese-roberta-wwm-ext"]
#models=["hfl/chinese-lert-large","nghuyong/ernie-3.0-base-zh","alibaba-pai/pai-ckbert-large-zh","hfl/chinese-pert-large"]
#models=["hfl/chinese-roberta-wwm-ext"]
models=["hfl/chinese-macbert-large"]
models=["luhua/chinese_pretrain_mrc_macbert_large"]
models=["hfl/chinese-macbert-large","Langboat/mengzi-bert-base","freedomking/mc-bert",
        "hfl/chinese-lert-large","nghuyong/ernie-3.0-base-zh","alibaba-pai/pai-ckbert-large-zh",
        "hfl/chinese-pert-large","hfl/chinese-roberta-wwm-ext","luhua/chinese_pretrain_mrc_macbert_large"]
for i in range(len(models)):
    print(models[i])
    num_epoch = 2
    local=1
    model_name="hfl/chinese-macbert-large"
    model_name="ShannonAI/ChineseBERT-largexxx"
    model_name="IDEA-CCNL/Erlangshen-ZEN2-668M-Chinesexxx not work"
    model_name="Langboat/mengzi-bert-base"
    model_name="Langboat/bloom-6b4-zh !!too big"
    model_name="freedomking/mc-bert"
    model_name="weiweishi/roc-bert-base-zhxx"
    model_name="hfl/chinese-lert-large"
    model_name="hfl/chinese-pert-large"
    model_name="IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece xx"
    model_name="nghuyong/ernie-3.0-base-zh"
    model_name="alibaba-pai/pai-ckbert-large-zh"
    model_name=models[i]
    #model_name="hfl/chinese-pert-large"
    #model_name_local="/saved_model/hfl/chinese-pert-large"
    #model_name_local="/saved_model/chinese-lert-large"
    model_name_local="saved_model/"+models[i]
    #model_name_local="/saved_model/chinese-lert-large"
    
    if local:
        model = BertForQuestionAnswering.from_pretrained(model_name_local, local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    # You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)

    # %%
    '''
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    '''

    # %%


    questions, paragraphs = read_data("hw7_train.json")
    l=int(len(questions)*0.99)
    train_questions, train_paragraphs = questions[:l],paragraphs
    #print(len(train_questions),len(train_paragraphs))
    dev_questions, dev_paragraphs = questions[l:],paragraphs
    #dev_questions, dev_paragraphs = read_data("hw7_dev.json")
    test_questions, test_paragraphs = read_data("hw7_test.json")
    print("Adding tokens......")
    print(len(test_paragraphs))

    c=[]
    for i,t in enumerate(test_paragraphs):
        if i%50==0:
            print(i,"/",len(test_paragraphs))
        for tt in t:
            try:
                if tokenizer.tokenize(tt)[0]=="[UNK]":
                    c.append(tt)
            except:
                pass
    print(c)
    tokenizer.add_tokens(c)

    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))
    #print(c)
 
    
    print("Adding done!")
    # %% [markdown]
    # ## Tokenize Data

    # %%
    # Tokenize questions and paragraphs separately
    # 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__ 

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=True)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=True)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=True) 

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=True)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=True)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=True)

    # You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model

    # %% [markdown]
    # ## Dataset

    # %%




    train_set= QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set= QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    #print(len(train_set))
    #print(len(dev_set))
    #print(train_set[3])
    #train_set,dev_set=random_split(train_set,[l,l2],generator=torch.Generator().manual_seed(42))


    #dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
    print()

    # %% [markdown]
    # ## Function for Evaluation

    # %%


    # %%
    #! rm text.txt


    # %% [markdown]
    # ## Training

    # %%
    from accelerate import Accelerator
    from transformers import get_linear_schedule_with_warmup

    # hyperparameters
    validation = True
    logging_step = 100
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_batch_size = 8
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
    
    fp16_training = True
    if fp16_training:    
        accelerator = Accelerator(mixed_precision="fp16")
    else:
        accelerator = Accelerator()

    # Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 


    # %% [markdown]
    # ## Testing

    # %%
    #print(paragraphs[2])

    # %%
    print("Evaluating Test Set ...")

    result = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                        attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(data, output))
    model_name=model_name.replace("/","_")
    result_file = "result/result_"+model_name+".csv"
    with open(result_file, 'w') as f:	
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
        # Replace commas in answers with empty strings (since csv is separated by comma)
        # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")

# %%



# %% [markdown]
# # GradeScope - Question 2 (In-context learning)

# %% [markdown]
# ### In-context learning
# The example prompt is :
# ```
# 請從最後一篇的文章中找出最後一個問題的答案：
# 文章：<文章1 內容>
# 問題：<問題1 敘述>
# 答案：<答案1>
# ...
# 文章：<文章n 內容>
# 問題：<問題n 敘述>
# 答案：
# ```
'''
# %%
import torch
import random  
import numpy as np

# To avoid CUDA_OUT_OF_MEMORY
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Fix random seed for reproducibility
def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
same_seeds(2)

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

# You can try model with different size
# When using Colab or Kaggle, models with more than 2 billions parameters may 
# run out of memory
tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-1.7B")
model = AutoModelForCausalLM.from_pretrained("facebook/xglm-1.7B")

# %%
# To clean model output. If you try different prompts, you may have to fix 
# this function on your own
def clean_text(text):
    # Note: When you use unilingual model, the colon may become fullwidth
    text = text.split("答案:")[-1]
    text = text.split(" ")[0]
    return text

# %%
import random
import json

with open("hw7_in-context-learning-examples.json", "r") as f: 
    test = json.load(f)

# K-shot learning 
# Give model K examples to make it achieve better accuracy 
# Note: (1) When K >= 4, CUDA_OUT_OFF_MEMORY may occur.
#       (2) The maximum input length of XGLM is 2048
K = 2

question_ids = [qa["id"] for qa in test["questions"]]

with open("in-context-learning-result.txt", "w") as f:
    print("ID,Ground-Truth,Prediction", file = f)
    with torch.no_grad():
        for idx, qa in enumerate(test["questions"]):
            # You can try different prompts
            prompt = "請從最後一篇的文章中找出最後一個問題的答案\n"
            exist_question_indexs = [question_ids.index(qa["id"])]

            # K-shot learning: give the model K examples with answers
            for i in range(K):
                question_index = question_ids.index(qa["id"])
                while(question_index in exist_question_indexs): 
                    question_index = random.randint(0, len(question_ids) - 1)
                exist_question_indexs.append(question_index)    
                paragraph_id = test["questions"][question_index]["paragraph_id"]
                prompt += f'文章：{test["paragraphs"][paragraph_id]}\n'
                prompt += f'問題：{test["questions"][question_index]["question_text"]}\n'
                prompt += f'答案：{test["questions"][question_index]["answer_text"]}\n'

            # The final one question without answer
            paragraph_id = qa["paragraph_id"]
            prompt += f'文章：{test["paragraphs"][paragraph_id]}\n'
            prompt += f'問題：{qa["question_text"]}\n'
            prompt += f'答案：'
            
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt") 
            sample = model.generate(**inputs, max_new_tokens = 20)
            text = tokenizer.decode(sample[0], skip_special_tokens=True)

            # Note: You can delete this line to see what will happen
            text = clean_text(text)
            
            print(prompt)
            print(f'正確答案: {qa["answer_text"]}')
            print(f'模型輸出: {text}')
            print()

            print(f"{idx},{qa['answer_text']},{text}", file = f)

# %%



'''