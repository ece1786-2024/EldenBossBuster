# Prepare kaggle data for RAG

The data (folder eldenringScrap) is a Kaggle dataset url: https://www.kaggle.com/datasets/pedroaltobelli/ultimate-elden-ring-with-shadow-of-the-erdtree-dlc/data

### Package management

```
conda create -n elden
conda activate elden
conda install pandas
```

### Process method

For table with numerical data like this one:

name, weight, requirements, attack
kanata, 6, {'Dex': 10}, 999

We want to put the numbers into text context so that llms can read

Step1 melt, transform the table to long form:

name, variable, value
kanata, weight, 6
kanata, requirements, {'Dex': 10}
kanata, attack, 999

This format makes it easier to create a sentence for each value.
Step2 to text
```
'The ' + df['variable'] + ' of ' + df['name'] + ' is ' + df['value']
```
kanata, weight, 6 --> The weight of kanata is 6
