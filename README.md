**Intellect Design Arena**

**Internship Documentation**

**By**

**Kalp Shah**

**Time period: 15-th-May to 14-th-July (2 months)**

**Mentor: Nitish Michael**

**TOPICS:**

1. **Character embedding and sentence classification**
2. **Coulmn Classifier**
3. **TAPAS**
4. **RWKV**

**1.Sentence classification using Character Level embedding:**

Character-level embeddings are an example of a text representation used in natural language processing (NLP) tasks. Character-level embeddings portray each character in a word as a vector instead of word-level embeddings, which represent words as vectors in a high-dimensional space. By doing this, we can handle words that aren't in our lexicon and record the morphology of words.

Each character must initially be represented as a vector in order to generate a character-level embedding for a word. A number of methods, including one-hot encoding and pre-trained character embeddings, can be used to accomplish this. Once each character has a vector representation, we can join all of the characters to make a single vector representation of the word. The character vectors can be combined in a variety of ways, including concatenation, averaging, or using a weighted sum.

On the input side, they dramatically increase the vocabulary our models can handle and show resilience in the face of spelling mistakes and rare words. On the output side, character models are computationally cheaper due to their small vocabulary size. This attribute makes training techniques (such as co-training a language model) feasible and fast even under a constrained budget.

Once we have character-level embeddings for each word in a phrase, we can use those embeddings to feed predictions into a classification model. The model could be a transformer-based model like BERT, a recurrent neural network (RNN), or a convolutional neural network (CNN).

**Two Models built using character level embedding and CNN:**

**a. Sentiment Analysis:**

We used the IMBD movies review dataset in this model to build the character-level embedding model.

- Dataset has a review for each movie and the corresponding sentiment as Label.
- Used 50 words from each sample as input for training the model.
- Build a 1D CNN model for predicting the samples with character embedding.
- Used Pytorch framework for building the model.
- Achieved F1 scores of 0.63 and 0.43 on positive and negative sentiment respectively.

**b. Names and Languages Dataset.**

- Data set contains names in different 18 languages.
- Build the model for predicting the language of the name entered.
- Build a 2D CNN model using character embedding.
- Achieved f1 score of 40%.

**2. Column Classifier:**

About the Dataset:

- Dataset contains one master dataset and several other dataset.
- Task is to predict the column name with the similar meaning.
- Master dataset have four column names, which are:

- Sno
- Names
- Date
- Job
- Created the dataset with the similar column names:

Created Dataset sample:

dataset = [
 ('ID', 'sno'),
 ('Reference Number', 'sno'),
 ('Index', 'sno'),
 ('Code', 'sno'),
 ('Identifier', 'sno'),
 ('Catalog Number', 'sno'),

 ('Title', 'names'),
 ('Full Name', 'names'),
 ('Last Name', 'names'),
 ('Given Name', 'names'),
 ('Alias', 'names'),
 ('Nickname', 'names'),


 ('Event Date', 'date'),
 ('Transaction Date', 'date'),
 ('Effective Date', 'date'),
 ('Due Date', 'date'),
 ('Creation Date', 'date'),

 ('Position', 'job'),
 ('Role', 'job'),
 ('Title', 'job'),
 ('Occupation', 'job'),
 ('Profession', 'job'),
 ('Specialization', 'job'),
 ]

- Build a simple CNN model for this task which predicts the name of the column of the master dataset when the column name from another dataset is given as input.

3. **TAPAS:**

Introduction:

TAPAS extends the BERT architecture to encode the question jointly along with tabular data structure, resulting in a model that can then point directly to the answer. Instead of creating a model that works only for a single style of table, this approach results in a model that can be applied to tables from a wide range of domains.

TAPAS Model:

Our model's architecture is based on BERT's encoder with additional positional embeddings used to encode tabular structure (visualized in Figure 2). We flatten the table into a sequence of words, split words into word pieces (tokens) and concatenate the question tokens before the table tokens.

We use different kinds of positional embeddings:

- Position ID is the index of the token in the flattened sequence (same as in BERT).
- Segment ID takes two possible values: 0 for the question, and 1 for the table header and cells.
- Column / Row ID is the index of the column/row that this token appears in, or 0 if the token is a part of the question.
- Rank ID if column values can be parsed as floats or dates, we sort them accordingly and assign an embedding based on their numeric rank (0 for not comparable, 1 for the smallest item, i + 1 for an item with rank i). This can assist the model when processing questions that involve superlatives, as word pieces may not represent numbers informatively
- Previous Answer given a conversational setup where the current question might refer to the previous question or its answers, we add a special embedding that marks whether a cell token was the answer to the previous question (1 if the token's cell was an answer or 0 otherwise).

![](RackMultipart20230713-1-u3ab5a_html_7505dd8641d0dd6c.png)

Different Formats of Datasets for TAPAS:

Basically, there are 3 different ways in which one can fine-tune TapasForQuestionAnswering, corresponding to the different datasets on which Tapas was fine-tuned:

SQA: if you're interested in asking follow-up questions related to a table, in a conversational set-up. For example if you first ask "what's the name of the first actor?" then you can ask a follow-up question such as "how old is he?". Here, questions do not involve any aggregation (all questions are cell selection questions).

WTQ: if you're not interested in asking questions in a conversational set-up, but rather just asking questions related to a table, which might involve aggregation, such as counting a number of rows, summing up cell values or averaging cell values. You can then for example ask "what's the total number of goals Cristiano Ronaldo made in his career?". This case is also called weak supervision, since the model itself must learn the appropriate aggregation operator (SUM/COUNT/AVERAGE/NONE) given only the answer to the question as supervision.

WikiSQL-supervised: this dataset is based on WikiSQL with the model being given the ground truth aggregation operator during training. This is also called strong supervision. Here, learning the appropriate aggregation operator is much easier.

To summarize:

| **Task** | **Example dataset** | **Description** |
| --- | --- | --- |
| Conversational | SQA | Conversational, only cell selection questions |
| --- | --- | --- |
| Weak supervision for aggregation | WTQ | Questions might involve aggregation, and the model must learn this given only the answer as supervision |
| Strong supervision for aggregation | WikiSQL-supervised | Questions might involve aggregation, and the model must learn this given the gold aggregation operator |

Dataset:

data = {"sno":["1","2"], "birthdate":["02/03/96","09/03/67"],"name": ["jack", "Ruby",], "job": ["carpenter", "professor"]}

Dataset Converted Into SQA Format:

queries = [
"What is the sno of jack?",
"What is the sno of Ruby?",
"What is the birthdate of jack?",
"What is the birth of Ruby?",
"What is the job of jack?",
"What is the job of Ruby?",
"Give the name with the carpenter job?",
"Give the name with the professor job?",

 ]
 answer\_coordinates = [[(0, 0)],[(1, 0)],[(0, 1)],[(1, 1)],[(0, 2)],[(1, 2)],[(0, 3)],[(1, 3)]]
 answer\_text = [["1"], ["2"], ["02/03/96"],["09/03/67"], ["jack"], ["Ruby"], ["carpenter"], ["professor"]]

**Cell Selection:**

The classification layer selects the subset of the cell from the table according to requirements. Depending on the selected aggregation operator, these cells can be the final answer or the input used to compute the final answer. Cells are modeled as independent Bernoulli variables. First, we compute the logit for a token using a linear layer on top of its last hidden vector. Cell logits are then computed as the average over logits of tokens in that cell. The output of the layer is the probability p(c)s to select cell c.

![](RackMultipart20230713-1-u3ab5a_html_942af99d4b2b4bef.png)

**Aggregation operator prediction:**

Semantic parsing tasks require discrete reasoning over the table, such as summing numbers or counting cells. To handle these cases without producing logical forms, TAPAS outputs a subset of the table cells together with an optional aggregation operator. The aggregation operator describes an operation to be applied to the selected cells, such as SUM, COUNT, AVERAGE or NONE. The operator is selected by a linear layer followed by a softmax on top of the final hidden vector of the first token (the special [CLS] token). We denote this layer as pa(op), where op is some aggregation operator.

**Inference:**

We predict the most likely aggregation operator together with a subset of the cells (using the cell selection layer). We select all table cells with a probability of larger than 0.5 to predict a discrete cell selection. These predictions are then executed against the table to retrieve the answer, by applying the predicted aggregation over the selected cells.

data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
'Age': ["56", "45", "59"],
'Number of movies': ["87", "53", "69"],
'Date of birth': ["7 february 1967", "10 june 1996", "28 november 1967"]}
 queries = ["How many movies has George Clooney played in?", "How old is he?", "What's his date of birth?"]

**Output:**

How many movies has George Clooney played in?"

Predicted answer: 53

"How old is he?"

Predicted answer: 45

"What's his date of birth?"

Predicted answer: 7 february 1967

**Limitations:**

1. TAPAS handles single tables as context, which are able to fit in memory. Thus, TAPAS model would fail to capture very large tables, or databases that contain multiple tables.
2. Although TAPAS can parse compositional structures, its expressivity is limited to a form of aggregation over a subset of table cells. Thus, structures with multiple aggregations such as "number of actors with an average rating higher than 4" could not be handled correctly.

**4. RWKV:**

**Receptance Weighted Key Value (RWKV**) model, a novel architecture that effectively combines the strengths of RNNs and Transformers while circumventing key drawbacks. RWKV is carefully designed to alleviate the memory bottleneck and quadratic scaling associated with Transformers with a more efficient linear scaling, while still preserving the rich, expressive properties that make the Transformer a dominant architecture in the field. One of the defining characteristics of RWKV is its ability to offer parallelized training and robust scalability, similar to Transformers.

The Implementation of linear atten"Ion 'n RWKV is carried out without approximation, which offers a considerable improvement in efficiency and enhances scalability.

The RWKV architecture derives its name from the four primary model elements used in the time-mixing and channel-mixing blocks:

• **R** : Receptance vector acting as the acceptance of past information.

• **W** : Weight is the positional weight decay vector. A trainable model parameter.

• **K** : The key is a vector analogous to K in traditional attention.

• **V** : Value is a vector analogous to V in traditional attention.

**Fine-Tuned the RWKV 169M model on four datasets:**

1. **IMBD movies review dataset**

Finetuned the model on imbd movies review dataset, which provides information or reviews about the movie asked as an output.

1. **Pandas Dataset**

Created the Pandas dataset and Finetune the 169-M RWKV model using the same dataset.

**Sample of the created dataset:**

"Function to get the mode of all values in a column
def get\_mode\_of\_all\_values\_in\_a\_column(df, column\_name):
return df[column\_name].mode() # break
Function to get the top n rows of a dataframe
def get\_top\_n\_rows(df, n):
return df.head(n) # break
Function to get the bottom n rows of a dataframe
def get\_bottom\_n\_rows(df, n):
return df.tail(n) # break
Function to rename a column in a dataframe
def rename\_column(df, current\_name, new\_name):
 df.rename(columns={current\_name: new\_name}, inplace=True)
return df # break
Function to fill missing values in a column with a specified value
def fill\_missing\_values(df, column\_name, value):
 df[column\_name].fillna(value, inplace=True)
return df # break
Function to drop rows with missing values in a specific column
def drop\_rows\_with\_missing\_values(df, column\_name):
return df.dropna(subset=[column\_name]) # break"

**Inference:**

input\_text = "Function to fill missing values in a column with a specified value"

#OUTPUT

"Generated text: Function to fill missing values in a column with a specified value
def fill\_missing\_values(df, column\_name, value):
df[column\_name].fillna(value, inplace=True)
return df"

**4. Employees dataset:**

Finetuned the RWKV 169-M model on employes dataset for sequence generation task

**Created Dataset Sample:**

" "serial number 17 18 19 \<start\>"sno\<stop\>
"name David Olivia Sophia \<start\>"name\<stop\>
"birthdate 1986 1993 1991 \<start\>"birthdate\<stop\>
"job Architect Designer Engineer \<start\>"job\<stop\>
"serial number 21 22 23 \<start\>"sno\<stop\>
"name Ava Benjamin Charlotte \<start\>"name\<stop\>
"birthdate 1997 1988 1994 \<start\>"birthdate\<stop\>
"job Lawyer Engineer Teacher \<start\>"job\<stop\>
"serial number 25 26 27 \<start\>"sno\<stop\>
"name Emma James Noah \<start\>"name\<stop\>
"birthdate 1995 1992 1999 \<start\>"birthdate\<stop\>
"job Doctor Architect Engineer \<start\>"job\<stop\>"

**Inference:**

input\_text = "name Sophia Noah Emma \<start\>"
#OUTPUT
"Column Name: sno"

**References:**

[**https://aclanthology.org/D14-1181/**](https://aclanthology.org/D14-1181/)

[**https://cs.stanford.edu/~diyiy/research.html**](https://cs.stanford.edu/~diyiy/research.html)

[**Implementation on Text Classification Using Bag of Words Model**](https://deliverypdf.ssrn.com/delivery.php?ID=359103099101004115116100010071116065023050001054093024010084017068124126077088077027031029003043109007047089110095066104071011052052059034038026113083024091122066118046054062026019072067127126119124115088102071091110122002094076094098007002114026024008&EXT=pdf&INDEX=TRUE)

[**https://arxiv.org/ftp/arxiv/papers/1806/1806.06407.pdf**](https://arxiv.org/ftp/arxiv/papers/1806/1806.06407.pdf)

[**https://huggingface.co/docs/transformers/model\_doc/tapas**](https://huggingface.co/docs/transformers/model_doc/tapas)

[**https://arxiv.org/abs/2004.02349**](https://arxiv.org/abs/2004.02349)

[**https://github.com/BlinkDL/RWKV-LM**](https://github.com/BlinkDL/RWKV-LM)

[**https://huggingface.co/docs/transformers/main/model\_doc/rwkv**](https://huggingface.co/docs/transformers/main/model_doc/rwkv)
