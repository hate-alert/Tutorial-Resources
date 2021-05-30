from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import numpy as np
import torch
from .model import *
from .utils import *

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    segmenter="twitter", 
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)



class modelPredRationale():
    def __init__(self, model_path = 'bert-base-uncased', device = None):
        self.device = device
        self.model_path=model_path
        self.model = Model_Rational_Label.from_pretrained(model_path,output_attentions = True,output_hidden_states = False).to(self.device)
        if torch.cuda.is_available():
            self.model.cuda()  
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast = False) 
        
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    def tokenize(self, sentences, padding = True, max_len = 128):
        input_ids, attention_masks, token_type_ids = [], [], []
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return {'input_ids': input_ids, 'attention_masks': attention_masks}
    
    def process_data(self, sentences_list):
        sentences = []
        sentence_lengths = []
        for sentence in sentences_list:
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
            sentence_lengths.append(len(self.tokenizer.encode(sentence)))
        inputs = self.tokenize(sentences)
        tokenized_sentences = [self.tokenizer.convert_ids_to_tokens(ele) for ele in  inputs['input_ids']]

        return self.get_dataloader(inputs), sentence_lengths, tokenized_sentences
    
    def get_dataloader(self, inputs):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'])
        sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=32)
    
    
    def return_rationales(self, sentences_list):
        """Input: should be a list of sentences"""
        """Output: probablity values"""
        device = self.device

        test_dataloader,sentence_lengths, tokenized_sentences=self.process_data(sentences_list)

        print("Running eval on test data...")
        labels_list=[]
        rationale_list=[]
        rationale_logit_list = []
        # Evaluate data 
        for step,batch in enumerate(test_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
      
            label_logits, rationale_logits = self.model(b_input_ids, b_input_mask)
                    
            label_logits = label_logits.detach().cpu().numpy()
            rationale_logits = rationale_logits.detach().cpu().numpy()
            
            final_logits=[]
            final_rationales=[]
            for i in range(label_logits.shape[0]):
                final_logits.append(softmax(label_logits[i]))
                final_rationales.append([ele[1] for ele in rationale_logits[i]])                
            labels_list+=final_logits
            rationale_list+=final_rationales
              
        attention_vectors = []
        for idx, rationales in enumerate(rationale_list):
            attention_vector = softmax(rationales[:sentence_lengths[idx]])
            attention_vector = list(attention_vector) + [0]*(128-len(list(attention_vector)))
            attention_vectors.append(attention_vector)
        
        tokens_sentence=[]
        for idx, tokenized in enumerate(tokenized_sentences):
            tokenized = tokenized[:sentence_lengths[idx]]
            tokens_sentence.append(tokenized)

        return np.array(labels_list), np.array(attention_vectors), tokens_sentence 



class modelPred():
    def __init__(self, language='english', device=device):
        self.__modelDict ={
        'arabic':"Hate-speech-CNERG/dehatebert-mono-arabic",
        'english': "Hate-speech-CNERG/dehatebert-mono-english",
        'french': "Hate-speech-CNERG/dehatebert-mono-english",
        'german': "Hate-speech-CNERG/dehatebert-mono-german",
        'indonesian': "Hate-speech-CNERG/dehatebert-mono-indonesian",
        'polish': "Hate-speech-CNERG/dehatebert-mono-polish",
        'portugese': "Hate-speech-CNERG/dehatebert-mono-portugese",
        'italian': "Hate-speech-CNERG/dehatebert-mono-italian",
        'spanish': "Hate-speech-CNERG/dehatebert-mono-spanish",
        'kannada': "Hate-speech-CNERG/deoffxlmr-mono-kannada",
        'malyalam': "Hate-speech-CNERG/deoffxlmr-mono-malyalam",
        'tamil': "Hate-speech-CNERG/deoffxlmr-mono-tamil",
        }
        self.device = device
        self.model_path=self.__modelDict[language]
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        # if(model_name=='xlmr'):
        #     self.model = XLMRobertaForSequenceClassification.from_pretrained(self.model_path,output_attentions = True,output_hidden_states = False).to(self.device)
        # elif(model_name=='bert'):
        #     self.model = BertForSequenceClassification.from_pretrained(self.model_path,output_attentions = True,output_hidden_states = False).to(self.device)
        self.model.cuda()  
        self.model.eval() 
        
    def preprocess_func(self, text):
        new_text = re.sub('@\w+', '@user',text)
        new_text = new_text.replace("\r\n\'",' ').replace("\n",' ')
        new_text = re.sub(r"http\S+", "", new_text)
        new_text = new_text.replace('&amp;', '&')
        return new_text
    
    def tokenize(self, sentences, padding = True, max_len = 128):
        input_ids, attention_masks, token_type_ids = [], [], []
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return {'input_ids': input_ids, 'attention_masks': attention_masks}
    
    def process_data(self, sentences_list):
        sentences = []
        for sentence in sentences_list:
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
        inputs = self.tokenize(sentences)
        return self.get_dataloader(inputs)
    
    def get_dataloader(self, inputs):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'])
        sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=32)
    
    def return_probab(self, sentences_list):
        """Input: should be a list of sentences"""
        """Output: probablity values"""
        device = self.device

        test_dataloader=self.process_data(sentences_list)

        print("Running eval on test data...")
        labels_list=[]
        sentence_lengths = [len(self.tokenizer.encode(sentence)) for sentence in  sentences_list]
        # Evaluate data 
        for step,batch in enumerate(test_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
      
            label_logits = self.model(b_input_ids, b_input_mask).logits        
            label_logits = label_logits.detach().cpu().numpy()
            
            final_logits=[]
            for i in range(label_logits.shape[0]):
                final_logits.append(softmax(label_logits[i]))
            labels_list+=final_logits
            
        return np.array(labels_list)