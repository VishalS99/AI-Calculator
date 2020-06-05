# The Simple Calculator

<p>A very basic calculator that understands handwritten math expressions and writtens the answer.</p>
<p>The software translates basic handwritten arithematic equations to machine readable ones and evaluates them.</p>

## Pipeline

<ol>
    <li>Preprocess image</li>
    <li>Segment te equation into individual characters.</li>
    <li>Predict each segmented character using a custom CNN</li>
    <li>Evaluate the final expression</li>
</ol>

## CNN model

A basic CNN with 3 sets of CNN layers and 2 Dense layers having a total of approximately 410,000 parameters.

<img src="./Details/summary.png" alt="Summary- CNN">

- The model was trained with 7315 training images and 1252 validation images (Thanks to [clarence zhao](https://www.kaggle.com/clarencezhao) for the [dataset](https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset))

- The model achieved a training accuracy of 97% with a training loss of 11%, and validation accuracy of 98% with 9% validation loss.

<img src="./Details/stats.png" alt="Metrics- CNN">
