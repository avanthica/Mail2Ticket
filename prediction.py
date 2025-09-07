from flask import Flask, request, jsonify 
from transformers import BertTokenizer
import tensorflow as tf
import numpy as np
import re
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Hide all logs except critical errors
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide INFO & WARNING logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU checks (Forces CPU mode)
import warnings
warnings.filterwarnings("ignore")  # Ignore Python warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)  # Suppress TensorFlow warnings completely

model = tf.keras.models.load_model("email_classification_new_data_20", compile=False)
tokenizer = BertTokenizer.from_pretrained("bert_model_tokenizer")

app = Flask(__name__)

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes=['Bug','CS Hardware - Server','CS Hardware - Spectrophotometer','Data - Colour Category','Data - Fibre Type','Data - New Triangle Code','Data - Preferred Triangles','Data - Shadecard','Data - Standards','Data Manipulation','IT Support','Improvement','Interface','Licence','Other','Question','Reporting','Update']):
    probs = model.predict(processed_data)[0]
    return probs,classes[np.argmax(probs)]



def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    # text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

@app.route('/')
def home():
    return "Flask API is running! Use /predict to classify emails.", 200

@app.route('/predict',methods=['POST'])
def predict():
    # try:
        data = request.get_json()
        print(data)
        
        if not data:
            return jsonify({"error": "Invalid request. Ensure JSON data contains 'email_text'"}), 400

        subject = data['Subject']
        body = data['Email']
        text = subject+ " " + body
        text = clean_text(text)
        processed_data = prepare_data(text, tokenizer)
        probs, result = make_prediction(model, processed_data=processed_data)
        print(f"Predicted issue type: {result}")
        return jsonify({
            "subject": subject,
            "body": body,
            "predicted_issue_type": result
        })
    



if __name__ == '__main__':
    app.run(port = 3000, debug = True)
