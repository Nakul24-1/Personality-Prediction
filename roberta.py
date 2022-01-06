import tensorflow as tf
from transformers import RobertaConfig,TFRobertaForSequenceClassification
from transformers import RobertaTokenizer
import numpy as np
from sklearn import preprocessing
import tensorflow_datasets as tfds
import pandas as pd



configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base",use_fast=False)
max_length = 400
batch_size = 16

config = RobertaConfig.from_json_file('app/Models/NEU/config.json')
model = TFRobertaForSequenceClassification.from_pretrained('app/Models/NEU/tf_model.h5', config=config)

config2 = RobertaConfig.from_json_file('app/Models/AGR/config.json')
model2 = TFRobertaForSequenceClassification.from_pretrained('app/Models/AGR/tf_model.h5', config=config2)
config3 = RobertaConfig.from_json_file('app/Models/cCon/config.json')
model3 = TFRobertaForSequenceClassification.from_pretrained('app/Models/cCon/tf_model.h5', config=config3)
config4 = RobertaConfig.from_json_file('app/Models/EXT/config.json')
model4 = TFRobertaForSequenceClassification.from_pretrained('app/Models/EXT/tf_model.h5', config=config4)
config5 = RobertaConfig.from_json_file('app/Models/cOPN/config.json')
model5 = TFRobertaForSequenceClassification.from_pretrained('app/Models/cOPN/tf_model.h5', config=config5)

def send_val():
  batch_size = 16
  
  comment_dataframe = pd.read_csv('input_essay.csv',header = None,names=['Username','Essay'])

  comment_dataframe['NEU'] = 0
  comment_dataframe['AGR'] = 0
  comment_dataframe['CON'] = 0
  comment_dataframe['EXT'] = 0
  comment_dataframe['OPN'] = 0

  #print(comment_dataframe)

  comment_dataframe=comment_dataframe.dropna()
  submission_sentences_modified = tf.data.Dataset.from_tensor_slices((comment_dataframe['Essay'],
                                                                            comment_dataframe['NEU']))
  ds_submission_encoded = encode_examples(submission_sentences_modified).batch(batch_size)
  
  
  submission_pre = tf.nn.softmax(model.predict(ds_submission_encoded))
  submission_pre2 = tf.nn.softmax(model2.predict(ds_submission_encoded))
  submission_pre3 = tf.nn.softmax(model3.predict(ds_submission_encoded))
  submission_pre4 = tf.nn.softmax(model4.predict(ds_submission_encoded))
  submission_pre5 = tf.nn.softmax(model5.predict(ds_submission_encoded))

  submission_pre_argmax = tf.math.argmax(submission_pre, axis=1)

  #print(np.array(submission_pre[0])[:,0])
  #print(np.array(submission_pre2[0])[:,0])
  #print(np.array(submission_pre3[0])[:,0])
  #print(np.array(submission_pre4[0])[:,0])
  #print(np.array(submission_pre5[0])[:,0])


  comment_dataframe['NEU'] = (np.array(submission_pre[0])[:,1])
  comment_dataframe['AGR'] = (np.array(submission_pre2[0])[:,1])
  comment_dataframe['CON'] = (np.array(submission_pre3[0])[:,1])
  comment_dataframe['EXT'] = (np.array(submission_pre4[0])[:,1])
  comment_dataframe['OPN'] = (np.array(submission_pre5[0])[:,1])


  #newdf=comment_dataframe[['Name','Neurotism','NEU']]
  newdf=comment_dataframe[['Username','NEU','AGR','CON','EXT','OPN']]

  #x = np.array(newdf['Neurotism']) #returns a numpy array
  #newdf.iloc[:,1:-1] = newdf.iloc[:,1:-1].apply(lambda x: (x-30)/ (120-30), axis=0)
  print(newdf)


  return (newdf)


def convert_example_to_feature(review):
  return roberta_tokenizer.encode_plus(review,
                                       add_special_tokens=True,
                                       max_length=max_length,
                                       pad_to_max_length=True,
                                       return_attention_mask=True,
  )

def map_example_to_dict(input_ids, attention_masks, label):
    return {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
           }, label

def encode_examples(ds, limit=-1):
    # Prepare Input list
    input_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)

    for review, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(review.decode())
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list,
                                               attention_mask_list,
                                               label_list)).map(map_example_to_dict)
