#这是用来对第三阶段数据集进行数据预处理的
from transformers import BertTokenizer
from bert_crf import BertCRFForTokenClassification
import torch
from tqdm import tqdm
import jsonlines
import os
import json
import warnings
import random
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class BertCrfTagging(object):
    def __init__(self,
                 data_path='/content/driver/My\ Drive/cail2023/train_query.json',
                 save_path='/content/driver/My\ Drive/cail2023/preprocess/query.json',
                 checkpoint='SCR-Preprocess/checkpoint-1900',
                 max_length=100,
                 batch_size=16,
                 device=torch.device('cpu'),
                 mode='query'):
        self.device = device
        self.model = BertCRFForTokenClassification.from_pretrained(checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-Chinese')
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.save_path = save_path
        self.mode = mode
        print('using {} device'.format(device))

    def process(self):
        if self.mode == 'query':
            data = list(jsonlines.open(self.data_path))
            data = self.generate_batch(data)
            with jsonlines.open(os.path.join(self.save_path), 'w') as f:
                for i, batch in enumerate(tqdm(data, desc='writing {} file'.format(self.mode))):
                    data[i] = self.add_inputs(batch)
                    for d in data[i]:
                        jsonlines.Writer.write(f, d)
        else:
            
                data = []
                files = []
                for file in os.listdir(os.path.join(self.data_path)):
                    data.append(json.load(open(os.path.join(self.data_path, file), encoding='utf-8')))
                    files.append(file)

                data = self.generate_batch(data)
                files = self.generate_batch(files)

                for i, batch in enumerate(data):
                    data[i] = self.add_inputs(batch)
                    for j, d in enumerate(data[i]):
                        with open(os.path.join(self.save_path, files[i][j]), 'w', encoding='utf-8') as f:
                            json.dump(d, f, ensure_ascii=False)

    def generate_batch(self, input_data):
        batches = []
        for idx in range(len(input_data)//self.batch_size+1):
            sub = input_data[idx*self.batch_size:(idx+1)*self.batch_size]
            if sub:
                batches.append(sub)

        return batches

    def add_inputs(self, batch):
        facts = []
        for b in batch:
            if 'query' in b:
                facts.append(b['fact'])
            else:
                str1=b["qw"]
                substr2="本院认为"
                substr="现已审理终结"
                index=str1.find(substr)
                str1=str1[index+6:]
                index2=str1.find(substr2)
                str1=str1[:index2]
                facts.append(str1)
                b['fact_part']=str1
                del b['qw']
                del b['result']
                del b['reason']
                del b['fact']
                del b['charge']
                del b['article']
        # dynamic padding
        # inputs = self.tokenizer.batch_encode_plus(facts, max_length=512)
        # max_length = min(max([len(ipt) for ipt in inputs['input_ids']]), 512)
        max_length = 512
        inputs = self.tokenizer.batch_encode_plus(facts,
                                                  max_length=max_length,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  return_tensors='pt')

        # shift tensors to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        # forward pass
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, pad_token_label_id=-100)
            bio_predictions = outputs[1]

            pad_mask = (bio_predictions != -100).long().to(self.device)
            bio_predictions = bio_predictions * pad_mask

            evt_predictions = (bio_predictions + 1) // 2

        inputs['event_type_ids'] = evt_predictions
        for i, b in enumerate(batch):
            input_ids = torch.masked_select(inputs['input_ids'][i], inputs['attention_mask'][i].bool())  # remove <PAD>
            input_ids = input_ids[1:-1]                           # remove the [CLS] and [SEP] tokens
            input_ids = input_ids[0: self.max_length].tolist()    # do truncation to 100 or 409

            event_type_ids = torch.masked_select(inputs['event_type_ids'][i], inputs['attention_mask'][i].bool())
            event_type_ids = event_type_ids[1:-1]
            event_type_ids = event_type_ids[0: self.max_length].tolist()

            if 0 in input_ids:
                print('shit')
            batch[i]['inputs'] = {'input_ids': input_ids,
                                  'event_type_ids': event_type_ids}
        return batch


if __name__ == "__main__":
    random.seed(42)

    os.makedirs('./output_data-supervised/query', exist_ok=True)
    os.makedirs('./output_data-supervised/candidates', exist_ok=True)

     #process query data
    model = BertCrfTagging(data_path='SCR-Preprocess/CAIL_stage_3/query.json',
                          save_path='SCR-Preprocess/output_data-supervised/query.json',
                          max_length=100,
                         batch_size=50,
                        mode='query',
                        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
    model.process()

    # process candidates data
    model = BertCrfTagging(data_path='SCR-Preprocess/CAIL_stage_3/corpus',
                           save_path='SCR-Preprocess/output_data-supervised/candidates',
                           max_length=409,
                           batch_size=50,
                           mode='candidate',
                           device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
    model.process()