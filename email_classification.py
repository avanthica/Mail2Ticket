import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
import pickle


df = pd.read_excel('subject_data_new.xlsx')
df.head()

df['encoded_issue_type'].value_counts()

df['text'] = df['Subject'].astype(str) + " " + df['Body'].astype(str)

# Preprocess the text (cleaning)
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    # text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

df['text'] = df['text'].apply(clean_text)
print(df['text'])

# In[5]:
#balancing of data code

# from imblearn.over_sampling import RandomOverSampler
# import pandas as pd

# # Example: Assuming `df` is your dataset with 'text' and 'encoded_issue_type'
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(df[['text']], df['encoded_issue_type'])

# df = pd.DataFrame({'text': X_resampled['text'], 'encoded_issue_type': y_resampled})
# df

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# token = tokenizer.encode_plus(
#     df['text'].iloc[0],
#     max_length=256,
#     truncation=True,
#     padding='max_length',
#     add_special_tokens=True,
#     return_tensors='tf'
# )

# token.input_ids

X_input_ids = np.zeros((len(df), 256))
X_attn_masks = np.zeros((len(df), 256))

def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['text'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks

X_input_ids, X_attn_masks = generate_training_data(df, X_input_ids, X_attn_masks, tokenizer)

labels = np.zeros((len(df), 18))
print(labels.shape)

# one-hot encoded target tensor
labels[np.arange(len(df)), df['encoded_issue_type'].values] = 1 


print(labels)

# creating a data pipeline using tensorflow dataset utility, creates batches of data for easy loading...
dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))
dataset.take(1) # one sample data

def SentimentDatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, labels


dataset = dataset.map(SentimentDatasetMapFunction) # converting to required format for tensorflow dataset

dataset.take(1)

dataset = dataset.shuffle(10000).batch(16, drop_remainder=True) # batch size, drop any left out tensor

dataset.take(1)

p = 0.8
train_size = int((len(df)//16)*p) # for each 16 batch of data we will have len(df)//16 samples, take 80% of that for train.
print(train_size)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

#print(train_dataset)


# ### **Model**

from transformers import TFBertModel

model = TFBertModel.from_pretrained('bert-base-uncased') # bert base model with pretrained weights

# defining 2 input layers for input_ids and attn_masks
input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

bert_embds = model.bert(input_ids, attention_mask=attn_masks)[1] # 0 -> activation layer (3D), 1 -> pooled output layer (2D)
intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(18, activation='softmax', name='output_layer')(intermediate_layer) # softmax -> calcs probs of classes

email_cls = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output_layer)
email_cls.summary()


optim = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

email_cls.compile(optimizer=optim, loss=loss_func, metrics=[acc])

hist = email_cls.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20
)

tokenizer.save_pretrained("bert_model_tokenizer")
email_cls.save('email_classification_new_data_20')

with open("email_cls_model.pkl", "wb") as f:
    pickle.dump("email_cls", f)



# import pandas as pd
# import numpy as np
# import re
# from tqdm.auto import tqdm
# import tensorflow as tf
# from transformers import BertTokenizer


# ### **Prediction**

# In[ ]:


# email_cls = tf.keras.models.load_model('email_classification_new_data')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def prepare_data(input_text, tokenizer):
#     token = tokenizer.encode_plus(
#         input_text,
#         max_length=256,
#         truncation=True,
#         padding='max_length',
#         add_special_tokens=True,
#         return_tensors='tf'
#     )
#     return {
#         'input_ids': tf.cast(token.input_ids, tf.float64),
#         'attention_mask': tf.cast(token.attention_mask, tf.float64)
#     }

# def make_prediction(model, processed_data, classes=['Bug','CS Hardware - Server','CS Hardware - Spectrophotometer','Data - Colour Category','Data - Fibre Type','Data - New Triangle Code','Data - Preferred Triangles','Data - Shadecard','Data - Standards','Data Manipulation','IT Support','Improvement','Interface','Licence','Other','Question','Reporting','Update']):
#     probs = model.predict(processed_data)[0]
#     return probs,classes[np.argmax(probs)]


# In[33]:


# import re

# def extract_subject_body(email_text):
#     # Match subject - case-insensitive for "Sub:", "SUB:", "Subject:"
#     subject_match = re.search(r"(?i)(?:Sub|Subject):\s*(.*?)(?=\s*(?:Body|BODY):|$)", email_text, re.DOTALL)
#     subject = subject_match.group(1).strip() if subject_match else "No Subject"

#     # Match body - case-insensitive for "Body:", "BODY:", "body:"
#     body_match = re.search(r"(?i)Body:\s*(.*)", email_text, re.DOTALL)
#     body = body_match.group(1).strip() if body_match else "No Body"

#     return subject, body


# In[34]:


# def clean_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\n', ' ', text)  # Remove newlines
#     text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
#     # text = re.sub(r'\d+', '', text)  # Remove numbers
#     return text


# In[51]:

#single instance prediction
# description = input('Enter description: ')
# subject, body = extract_subject_body(description)
# print(subject)
# print(body)
# text = subject+ " " + body
# print(text)
# text = clean_text(text)
# processed_data = prepare_data(text, tokenizer)
# probs, result = make_prediction(email_cls, processed_data=processed_data)
# print(np.argmax(probs))
# print(f"Predicted issue type: {result}")
