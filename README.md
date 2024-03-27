# IITK-at-SemEval-2024-Task-10: Emotion-Discovery-and-Reasoning-its-Flip-in-Conversation-EDiReF
- The following repository contains the source code for our submission in the Task 10 of SemEval-2024

## Task Overview
- The Task consists of Emotion Recognition in Conversation (ERC) and Identifying the parts of the conversation that caused the emotion of a speaker to change (EFR).
- Sub-Task 1 was ERC in Hindi-English Code-Mixed Conversations
- Sub-Task 2 was EFR in Hindi-English Code-Mixed Conversations
- Sub-Task 3 was EFR in English Conversations

## Our Approach
This paper presents our approach for the SemEval-2024 Task 10: Emotion Discovery and Reasoning its Flip in Conversations. We propose a transformer-based speaker-centric model for the Emotion Flip Reasoning (EFR) task and a masked-memory network along with a speaker participation vector for the Emotion Recognition in Conversations (ERC) task. We propose a Probable Trigger Zone, which is more likely to contain the utterances causing the emotion of a speaker to flip. In EFR, sub-task 3, the proposed approach archives a 5.9 (F1 score) improvement over the provided task baseline. The ablation study results highlight the significance of various design choices in the proposed method.

## Repository Structure
- `Data/`
    - Consisits of the training, validation and test data for all the three subtasks
- `Dataloaders/`
    - Consists of the code to convert data into pickle files which the model uses
- `Inference/`
    - Consists of codes utilised for inference phase of models
- `Pickles/`
    - Consists of the utterance embedding pickle files we utilised for our submission
- `Training/`
    - Consists of codes for training the models
- `Utility/`
    - `combine_predictions.ipynb`
        - To combine the predictions for all three tasks into the competition excepected format
    - `hing_bert_embeddings.ipynb`
        - To compute utterance embeddings for Hindi-English Code Mixed Utterances
        - Produces `sent2emb.pickle`
    - `erc_data_modify.ipynb`
        - Splits the data into smaller conversations for the ERC Task
    - `hypothesis.ipynb`
        - Transforms the predicitons of the model for the EFR task accoring to the hypothesis we have proposed in the paper
    - `json_to_csv.ipynb`
        - Prunes and modifies the data according to specifications and converts it to a form expected by dataloaders
    - `voyage_embeddings.ipynb`
        - To compute utterance embeddings for English Utterances
        - Produces `sent2emb.pickle`

## ERC Model - Training and Inference
- Split the data using `erc_data_modify.ipynb`
- Convert the new data into csv format using `json_to_csv.ipynb`
- Compute the utterance embeddings
- Generate pickle files following instructions given in Dataloaders
- Train the model
- For prediction, similar to training, generate the pickle files by setting both train and val paths with corresponding test paths
- Run the inference scripts

## EFR Model - Training and Inference
- Convert it into csv format using `json_to_csv.ipynb`
- Compute the utterance embeddings
- Generate pickle files following instructions given in Dataloaders
- Train the model
- For prediction, similar to training, generate the pickle files by setting both train and val paths with corresponding test paths
- Run the inference scripts
- Apply the hypothesis on the predictions using `hypothesis.ipynb`

## Our Models: 
- The models trained and used for submission can be found below
- https://drive.google.com/drive/folders/1QQxarnm4z52BTwd0sBFnZsVfLNRFj8Ed?usp=sharing
## Reference
https://github.com/LCS2-IIITD/Emotion-Flip-Reasoning
