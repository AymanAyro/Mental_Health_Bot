# **Project Title: Mental Health Chatbot - Pandora**

## **Objective**:
The goal of this project is to develop a mental health chatbot named Pandora that can interact with users and provide relevant responses based on their queries. The chatbot utilizes natural language processing (NLP) techniques to understand user inputs and classify them into specific intents, allowing it to deliver contextually appropriate responses.

##### Check It Out!: [Mental Health Bot Streamlit App ](https://mentalhealthbot-jt65ytr62h6hyt6zwiqklm.streamlit.app/)

## **Usage Instructions**:
- Run the Application: To launch the chatbot application, navigate to the project directory and use the following command in the terminal:
 
`streamlit run app.py`
- Installing Required Packages: Install the required Python libraries listed in requirements.txt by running:
  
`pip install -r requirements.txt`

Ensure all dependencies like Transformers, Azure Text Analytics, Streamlit, PyTorch, and MLflow are correctly installed for the app to function.

## **Credits**:
- The intent dataset used for training the chatbot is available on Kaggle:
 [Kaggle Dataset for Mental Health Chatbot](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data))
  
### **Project Overview**:

1. ***Data Preparation***:

- Load the intents from a `JSON` file (intents)json) that contains various user inputs and corresponding responses)
- Extract patterns (user inputs) and responses, preparing them for further processing.

2. ***Text Preprocessing***:

- Utilize the `nltk` library to tokenize and clean the text data. This involves converting the text to lowercase, removing punctuation, and eliminating stopwords.
- Additionally, employ the `spaCy` library for lemmatization, ensuring that words are reduced to their base forms to enhance the model's understanding of the language.
  
3. ***Data Analysis and Visualization***:

- Analyze the distribution of patterns across different intents and visualize this distribution using bar charts.
  
 ![image](https://github.com/user-attachments/assets/9eb358ac-7fa7-40e8-90b4-953616cb1a33)

- Create a word cloud from all user patterns to highlight the most frequently used terms.

![image](https://github.com/user-attachments/assets/eceffe84-8666-4449-8969-19944188167e)


- Examine the lengths of the user inputs and visualize the distribution of sentence lengths with histograms.

  ![image](https://github.com/user-attachments/assets/d76fd2f9-3a42-49cd-bab2-79ffd13e7976)

- Checking Token Frequency

![image](https://github.com/user-attachments/assets/730daa87-b887-4756-a302-e5acb995979f)

4. ***Model Preparation***:

- Convert the textual data into numeric labels for model training, mapping each intent to a unique identifier.
- Split the data into training and validation sets to evaluate model performance.

```
# Prepare dataset
texts = []
labels = []
label_to_id = {}
id_to_label = {}
for idx, intent in enumerate(intents['intents']):
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])
    label_to_id[intent['tag']] = idx
    id_to_label[idx] = intent['tag']

# Convert labels to numeric form
numeric_labels = [label_to_id[label] for label in labels]

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, numeric_labels, test_size=0.2, random_state=42)

# Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

train_encodings = tokenize_data(train_texts)
val_encodings = tokenize_data(val_texts)
```

5. ***Tokenization with BERT***:

- Leverage the BERT tokenizer to prepare the text data for model input. This process involves padding, truncation, and returning tensors suitable for the model.

6. ***Dataset Creation***:

- Define a custom dataset class to facilitate loading the training and validation data into the `PyTorch` framework.

7. ***Model Training***:

- Instantiate a BERT model specifically designed for sequence classification.
- Hyperparameters Used:
-- Batch Size: 128
-- Optimizer: AdamW
-- Number of Epochs: 200
-- Learning Rate: 5e-5
-- Weight Decay: 0.001
-- Learning Rate Scheduler: Cosine with minimum learning rate of 1e-6
- Utilize the Trainer class from the transformers library to fine-tune the BERT model on the training dataset.
```
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=200,
    learning_rate = 5e-5,
    lr_scheduler_type = 'cosine_with_min_lr',
    lr_scheduler_kwargs = {'min_lr': 1e-6},
    weight_decay=0.001,
)
```
- Evaluation Loss: The model achieved an evaluation loss of *2.016* at the end of training.
  ![image](https://github.com/user-attachments/assets/ff10ae26-71cb-435a-8e41-3ed3b9272479)

8. ***Intent Prediction***:

- Implement a function to predict user intent by tokenizing the input text and using the trained model to generate predictions.
Convert the predicted numeric label back to the corresponding string tag.
```
def predict_intent(text):
    # Tokenize the input text and move it to the same device as the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Perform the forward pass with all tensors on the same device
    outputs = model(**inputs)
    
    # Get the logits and apply softmax
    probs = outputs.logits.softmax(dim=1)
    
    # Get the predicted label (numeric)
    predicted_label = probs.argmax().item()
    
    # Convert the numeric label back to the string tag
    predicted_intent = id_to_label[predicted_label]
    
    return predicted_intent
```

9. ***Response Generation***:

- Create a response function that retrieves an appropriate response from the intents based on the predicted user intent.
```
def get_response(intent):
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            return intent_data['responses'][0]  # Pick a response
    return "Sorry, I don't understand."
```

10. ***Chatbot Interaction***:

- Develop an interactive loop for user engagement where users can input their queries and receive responses from the chatbot.
- Allow users to exit the chat gracefully by typing 'quit'.
```
def mental_health_chatbot():
    print("Hello, I'm Pandora, your mental health assistant. How can I help you today? (Type 'quit' to exit)\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        print("\nUser Input:", user_input)
        # Check if the user wants to quit
        if user_input.lower() == 'quit':
            print("Pandora: Goodbye! Take care.")
            break
        
        # Predict the intent of the user input
        intent = predict_intent(user_input)
        print(f"\nPredicted intent: {intent}")  # Debugging purposes, you can remove this line if not needed
        
        # Get a response based on the predicted intent
        response = get_response(intent)
        
        # Display the chatbot response
        print(f"\nPandora: {response}")
```
![image](https://github.com/user-attachments/assets/c1a8a686-bbd8-4723-a7e5-36f205931a28)

# **Implementation**:

## **Framework and Libraries**:

- Utilizes `Streamlit` for the user interface.
- Employs `Transformers` library for the `BERT` model.
- Integrates `Azure Text Analytics` for sentiment analysis.
- Uses `PyTorch` for model handling and inference.
- Incorporates `MLflow` for tracking experiments and logging interactions.
  
## **Initialization**:

- Sets up Azure Text Analytics credentials and client.
- Defines paths for saving model and tokenizer files.
  
## **File Management**:

- Checks for local existence of model and tokenizer files.
- Downloads necessary files from Google Drive if not present.
  
## **Model and Tokenizer Loading**:

- Loads the pre-trained BERT model and tokenizer.
- Transfers the model to the appropriate device (CPU/GPU).
  
## **Intent Management**:

- Loads intents from a JSON file to extract user intents and responses.
- Prepares mappings between labels and intents.
  
## **Sentiment Analysis**:

- Implements a function to analyze sentiment using Azure Text Analytics.
![image](https://github.com/user-attachments/assets/001f6d65-ee8a-44df-8ee9-2028ab60e338)

## **Intent Prediction**:

-Defines a function to predict user intent by tokenizing input and utilizing the BERT model.

## Response Generation:

-Creates a function to generate responses based on predicted intent and sentiment.

## **User Interaction**:

- Initializes the `Streamlit` app with a title and description.
- Manages conversation history using Streamlit session state.
- Processes user input for sentiment analysis and intent prediction.
- Generates and displays bot responses.
  
## **Logging and Tracking**:

- Uses `MLflow` to log parameters and model interactions for tracking performance and improvements.
  
  ![image](https://github.com/user-attachments/assets/613cdd34-a9a6-4638-a18b-e5121e9791e7)

  ![image](https://github.com/user-attachments/assets/c471c060-f1af-42d6-807e-5597bc624ce3)
  
  ![image](https://github.com/user-attachments/assets/adbf563b-6f3f-410a-bc62-f460fcf0f4f1)

  ![image](https://github.com/user-attachments/assets/5a844887-2cf6-48ce-96b7-01a79960017d)



## **Display Conversation**:

- Presents the conversation history, showing both user messages and bot responses along with intents.
![image](https://github.com/user-attachments/assets/b0eb137b-3c8f-4daf-b2be-99ea55fb35e7)

![image](https://github.com/user-attachments/assets/a08fc2ec-55da-4077-a29f-6b2fb4e6aac8)

# **Future Work**:

1. ***Utilizing More Data***:

- One of the key areas for improvement is expanding the dataset by incorporating additional data sources. This can include gathering more conversational data relevant to mental health, as well as more diverse user inputs. Larger datasets will allow the model to generalize better across different conversation contexts and improve response accuracy.
- Leveraging data augmentation techniques for text, such as paraphrasing or syntactic variations, could also help increase the training data without collecting entirely new data.
  
2. ***Fine-Tuning the Model***:

- Further fine-tuning the BERT model on domain-specific mental health datasets can improve the chatbot's understanding of sensitive conversations. Fine-tuning can also enhance the model's ability to detect subtle differences in user intent and sentiment.
- Implementing a hyperparameter search for the best model configurations (such as learning rate, batch size, etc.) could lead to more optimal performance and reduce the overall evaluation loss.

3. ***Adding More Features***:

- Context Awareness: Improving the bot’s ability to maintain a multi-turn, context-aware conversation. Currently, the chatbot processes each user input individually, but enhancing the context handling will allow for more coherent conversations across multiple exchanges.
- Emotion Recognition: Adding facial expression analysis (using computer vision) or voice sentiment analysis to further refine the user's emotional state could help deliver even more personalized responses.
- Recommendation System: Including personalized recommendations for mental health resources or exercises based on detected emotions and intent could be another powerful feature.
- Multi-Language Support: Extending the chatbot’s capabilities to handle conversations in multiple languages, allowing it to cater to a broader audience globally.

# **Conclusion**:
- By utilizing advanced NLP techniques and pre-trained models like BERT, Pandora is capable of understanding user queries related to mental health and providing helpful responses. This project not only demonstrates the application of machine learning in real-world scenarios but also addresses the significant need for accessible mental health support. This implementation provides a comprehensive framework for creating a mental health chatbot that leverages advanced NLP techniques, ensuring a supportive and interactive experience for users.
