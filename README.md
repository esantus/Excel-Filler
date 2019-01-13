# Excel Filler

Excel Filler is a neural classifier that accurately associates a class to one or more input columns in an Excel file.


## What to use it for?

Suppose you have an Excel file containing one or more columns (for example, region and city), and you need to classify each row in these columns according to a specific type (for example, country). Let's suppose that for some of these rows you already have the type, but you are missing it for many others.

Excel Filler helps you to easily and accurately fill the missing types.


## How does it work?

Excel Filler consists in a word+char neural network, which is fast to train and particularly accurate. With the appropriate tuning, this technology can predict the type of any input column (or combination of columns) for which it was trained.

The system loads an Excel file. The user has to indicate which columns represent the input and which columns represent the output. After observing the existing input-output combinations, Excel Filler learn how to predict the most appropriate output for each given input.


## How to use it

In its easier setting, it is sufficient to call the program as:

```
python main.py
```
