# Open-Book Question Answering 

This repository contains code related to experiments conducted within the framework of an internship at Sinch (Sweden) focused on open book question answering systems. 

### Internship topic abstract
This internship explores the development of an open-book question answering (QA) system, a well studied task in natural language processing (NLP). The study focuses on fine-tuning large language models, comparing encoder transformers like BERT and DeBERTa with generative decoder-only transformers like GPT2 and Llama2-7B. The models undergo supervised fine-tuning on SQuAD v1 and v2 datasets, with an investigation into direct preference optimization (DPO). Specific scores and threshold selection are applied to control the generation of incorrect answers. The comparative study concludes with real Sinch data, evaluating the models' performance on conversation API documentation.

### Usage
The repository is organized into three subdirectories: extractive_qa, generative_qa, and inference. Please refer to the requirements.in file for all necessary dependencies. The code utilizes the Hugging Face transformers libraries for implementation.

#### extractive_qa 
Inside the "extractive_qa" directory, you can find subfolders containing code for fine-tuning and validating on SQuAD v1 and v2 datasets, as well as validating on the trivia_qa dataset for the extractive QA model. Experiments are intended to be run using the "main.py" files in the "src" subdirectories. All experiment hyperparameters can be modified in the "config.json" file within the "configs" subdirectories.

###### Command
```
cd extractive_qa/*****_experiments/src
python main.py
```

###### SQuAD v1, v2 hyperparameters

| Hyperparameter Name | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| experiment_name     | Defines the experiment's name, used to select the output directory|
| dataset_id          | Hugging Face dataset ID                                           |
| model_id            | Hugging Face model ID                                             |
| device              | Specifies the device for running experiments                      |
| batch_size          | Determines the training batch size                                |
| epochs              | Specifies the number of training epochs                           |
| lr                  | Sets the learning rate                                            |
| max_length          | Sets the size of the input sequence                               |
| stride              | Defines the stride used when truncating examples that are too long|
| max_answer_length   | Controls the maximum size of a predicted answer in terms of tokens|
| n_best              | Determines the number of candidate answers                        |
| seed                | Sets the random seed for reproducibility                          |

###### Trivia_qa hyperparameters

| Hyperparameter name | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| experiment_name     | name of the experiment, use to select the output directory        |
| dataset_id          | ID of the Hugging Face dataset                                    |
| dataset_config      | Dataset configuration                                             |
| dataset_json        | Sets the device on which the experiments will run                 |
| tokenizer_id        | ID of the Hugging Face tokenizer                                  |
| model_id            | ID of the Hugging Face model                                      |
| device              | Specifies the device for running experiments                      |
| stride              | Defines the stride used when truncating examples that are too long|
| max_length          | Sets the size of the input sequence                               |
| batch_size          | Sets the training batch size                                      |
| n_best              | Determines the number of candidate answers                        |
| max_answer_length   | Controls the maximum size of a predicted answer in terms of tokens|
| seed                | Sets the random seed for reproducibility                          |

#### generative_qa 
In the "generative_qa" folder, you can find all the code related to fine-tuning (SFT, DPO) and validation of decoder-only large language models only on SQuAD v1 and v2 datasets. Similarly, the code is intended to be executed via the "main.py" file located in the "src" subfolder, and hyperparameters can be adjusted in the "config.json" file located in the "configs" subfolder.

###### Command
```
cd generative_qa/src
python main.py
```

###### Hyperparameters

| Hyperparameter name | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| experiment_name     | Name of the experiment, used to specify the output directory      |
| dataset_id          | ID of the Hugging Face dataset                                    |
| tokenizer_id        | ID of the Hugging Face tokenizer                                  |
| model_id            | ID of the model (can be either a local or Hugging Face checkpoint)|
| device              | Specifies the device for running experiments                      |
| epochs              | Number of training epochs                                         |
| max_steps           | Maximum number of training steps (Overrides epochs if different from -1)|
| Qlora               | Set to "True" to use Qlora, "False" otherwise                     |
| train               | Finetuning method ("SFT", "DPO", "None")                          |
| val                 | Set to "True" to perform validation, "False" otherwise            |
| few_shot            | Set to "True" to perform few-shot learning, "False" otherwise     |
| K                   | Size of the few-shot training dataset                             |
| lr                  | Learning rate for training                                        |
| beta                | Beta parameter for DPO                                            |
| stride              | Defines the stride used when truncating examples that are too long|
| max_length          | Sets the size of the input sequence                               |
| block_size          | Block size for language modeling (SFT)                            |
| batch_size          | Sets the training batch size                                      |
| max_answer_length   | Controls the maximum size of a predicted answer in terms of tokens|
| seed                | Sets the random seed for reproducibility                          |

#### inference
In the "inference" folder, you can find the code related to inference for extractive and generative models. In the "src" subfolder, a notebook (extractive_inference.ipynb) is provided for inference of extractive models. The inference code for generative models is intended to be run via the "generative_inference.py" file, and the hyperparameters related to this configuration can be adjusted in the "config.json" file in the "configs" subfolder.

###### Command
```
cd inference/src
python generative_inference.py
```

###### Generative inference hyperparameters

| Hyperparameter Name | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| experiment_name     | Name of the experiment, used to select the output directory       |
| tokenizer_id        | ID of the Hugging Face tokenizer                                  |
| model_id            | ID of the model (can be either a local or Hugging Face checkpoint)|
| device              | Specifies the device for running experiments                      |
| Qlora               | Set to "True" to use Qlora, "False" otherwise                     |
| no_ans_threshold    | Threshold on the "no answer" score to determine whether the answer provided is correct or not|
| ans_threshold       | Threshold on the answer score to determine whether the answer provided is correct or not|
| stride              | Defines the stride used when truncating examples that are too long|
| max_length          | Sets the size of the input sequence                               |
| max_answer_length   | Controls the maximum size of a predicted answer in terms of tokens|
| seed                | Sets the random seed for reproducibility                          |
