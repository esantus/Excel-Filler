# Excel Filler: a Word+Char Convolutional Neural Network for Filling Missing Fields in Excel Files


## About

*Excel Filler* is a Word+Char Convolutional Neural Network that accurately classifies text or predict numbers in one or more input columns of an Excel file and writes the predicted class and numbers in the output columns. Both input and output columns can be indicated by the user.


## Compatibility and Future Development

Please notice that the code works well with any versions of pytorch.


## What to use it for?

Suppose you have an Excel file with two or more textual or number columns. One or more of these columns are fully filled, while one or more of them are only partially filled.

An example (see: example.xlsx) can be an Excel file containing *Region*, *City* and *Country*. Suppose that for *Region* and *City* you have thousands of filled rows, while for *Country* you only have few hundred filled ones. Given this situation, you can train *Excel Filler* to learn the association between the existing combinations of *Region*, *City* and *Country*, and --- on this basis --- predict the countries of the remaining *Region*-*City* pairs.

The insertion of character embedding is meant to allow users to apply *Excel Filler* also on non-word fields (e.g. bar-codes).

The input columns can be both characters or numbers, and the user needs to specify which columns are words and which columns are numbers. Similarly, the output can also be either words or numbers. Notice that depending on whether user wants classification or regression, the user needs to specify the loss function to be cross-entropy or MSE.

Because the system will learn on the existing *Region*-*City*-*Country* combinations, it is important to notice that at the prediction time it will infer the new combinations (i.e. it will classify the *Region*-*City* pairs) only on the basis of what it has experienced during training. This means, in other words, that it will classify the *Region*-*City* pairs only according to any of the *Countr*ies that it has seen during training time.


## How does it work?

*Excel Filler* consists in a Word+Char Convolutional Neural Network (CNN), which is a neural technique that is simultaneously fast to train and highly accurate. When appropriately tuned, this system can predict --- for any input column(s) --- a class, among those observed during training.

The system loads the input and output columns indicated by the user from an Excel file. It processes the contained text and runs a machine learning method to learn the existing combinations and predict the missing ones.

As any other machine learning method, *Excel Filler* needs to go through training, validation and testing, before it can be actually used for predictions. You can switch these modalities by simply calling the program in the following way:

```
python main.py --mode train --excel_file excel_path --embedding_file embedding_path --input_columns col1,col2 --output_columns col3,col4

python main.py --mode train --excel_file excel_path --embedding_file embedding_path --input_columns col1,col2 --output_columns col3,col4

python main.py --mode train --excel_file excel_path --embedding_file embedding_path --input_columns col1,col2 --output_columns col3,col4
```

While all input columns are processed together, the system loops among the output columns. The loop includes all the three modes (i.e. train, test, predict). In the 'predict' mode, for each output column it will generate an excel file containing all the existing columns plus a new column with the predictions. The file names clearly describe the predicted column and the source file.

For more information about Convolutional Neural Network, please read [this nice article](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/) from Adit Deshpande.


More details on how to properly configure the command:

**With only word input and output**:
If all the input and output columns are words, then the user can simply specify the input_columns and output_columns.
```
python3 main.py --mode train --excel_file output.xlsx --input_columns sent --output_columns lab
```


**With both word input and number input, but words output**:
In this case, the user needs to add the number input by using **--input_numbers col**
```
python3 main.py --mode train --excel_file output.xlsx --input_columns sent --input_numbers sth --output_columns lab
```


**With just number input**:
In this case, the user needs to specify the input columns using **--input_numbers a,b,c,d**
```
python3 main.py --mode train --excel_file iris.xlsx --input_numbers a,b,c,d --output_columns e
```
**On the Iris dataset, we are able to get an accuracy of 1.0.

**With just number input, and the goal being Regressor**:
In this case, the user needs to specify the loss to be MSE by using **--objective mse**
```
python3 main.py --mode predict --excel_file boston.xlsx --input_numbers DIS,RAD,TAX,PTRATIO,B --output_columns target --objective mse
```
**On the Boston dataset, we are able to get a MSE error of 0.6**.

Accuracies tested on toy dataset:

For classification task, on the Iris dataset, we are able to get an accuracy of $1.0$.
For regression task, on the Boston dataset, we are ablt to get an MSE loss of $0.6$.





## Configuration

The system includes a large range of hyper parameters that the user might want to set. Those that might need more attention are mode, gpu, class_balance, char, epochs, batch_size, char_emb_dims.

```
>>> python main.py --help

Usage: main.py [options]

Options:
  -h, --help            show this help message and exit
  --mode=MODE           save the mode (train, test, predict)
  -d, --debug           if True, print debug information
  -g, --gpu             if True, use the GPU
  -b, --class_balance   if True, use class balance
  -r, --char            if True, use character embeddings too
  -f EXCEL_FILE, --excel_file=EXCEL_FILE
                        excel file to be used for Training/Prediciton
  -e EMBEDDING_FILE, --embedding_file=EMBEDDING_FILE
                        embedding file to be used
  -i INPUT_COLUMNS, --input_columns=INPUT_COLUMNS
                        input columns in format: x1,x2...,xN
  -o OUTPUT_COLUMNS, --output_columns=OUTPUT_COLUMNS
                        output columns in format: x1,x2...,xN
  -s MODEL_PATH, --model_path=MODEL_PATH
                        folder in which the model is (going to be) saved
  -n MODEL_NAME, --model_name=MODEL_NAME
                        model name as it is (going to be) saved
  -m TUNING_METRIC, --tuning_metric=TUNING_METRIC
                        tuning metric
  -j OBJECTIVE, --objective=OBJECTIVE
                        objective function
  --init_lr=INIT_LR     save the initial learning rate
  --epochs=EPOCHS       save the number of epochs
  --batch_size=BATCH_SIZE
                        save the batch size
  --patience=PATIENCE   save the patience before cutting the learning rate
  --emb_dims=EMB_DIMS   save the embedding dimension
  --char_emb_dims=CHAR_EMB_DIMS
                        save the char embedding dimension
  --hidden_dims=HIDDEN_DIMS
                        save the number of hidden dimensions for TextCNN
  --num_layers=NUM_LAYERS
                        save the number of layers
  --dropout=DROPOUT     save the dropout probability
  --weight_decay=WEIGHT_DECAY
                        save the weight decay
  --filter_num=FILTER_NUM
                        save the number of filters
  --filters=FILTERS     save the list of filters in format x1,x2...,xN
  --num_class=NUM_CLASS
                        save the number of classes in the output
  --max_words=MAX_WORDS
                        save the maximum number of words to use from the input
  --max_chars=MAX_CHARS
                        save the maximum number of chars to use for every word
  --train_size=TRAIN_SIZE
                        save the relative size of the training set
  --dev_size=DEV_SIZE   save the relative size of the dev set
  --test_size=TEST_SIZE
                        save the relative size of the test set
  --num_workers=NUM_WORKERS
                        save the number of workers
```



