# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier designed to predict whether an individual's annual income exceeds $50,000 USD based on various census features. It was developed using the scikit-learn library and is part of a scalable machine learning pipeline that includes automated data processing and API deployment via FastAPI.

## Intended Use

The intended use of this model is for educational and research purposes to study demographic income trends within the United States. It is a binary classification tool meant for developers and data scientists. This model is not intended for use in actual financial or credit-scoring applications, as the underlying data is outdated.

## Training Data

The model was trained on the UCI Census Income Dataset (also known as the "Adult" dataset). The data contains approximately 32,561 entries with features such as age, education level, occupation, and marital status. The training set was derived using an 80/20 split of the original data. Categorical features were pre-processed using One-Hot Encoding, and the target label ("salary") was transformed using a Label Binarizer.

## Evaluation Data

The evaluation was conducted on a held-out test set comprising 20% of the original data. This test set was not seen by the model during the training phase. The test data underwent the same pre-processing transformations as the training set to ensure the validity of the results.

## Metrics

The model performance was evaluated using three primary metrics: Precision, Recall, and the F1 Score. Based on the most recent training run, the model achieved the following performance on the test set:

Precision: 0.7419

Recall: 0.6384

F1 Score: 0.6863

## Ethical Considerations

The dataset used for this model contains sensitive demographic attributes, including race and sex. An analysis of model performance across these categories reveals significant disparities in predictive accuracy. For instance, the model achieves a higher F1 score of 0.6997 for individuals identified as Male, compared to an F1 score of 0.6015 for individuals identified as Female. Additionally, performance varies across racial groups, with the Black demographic group showing an F1 score of 0.6667, while the White demographic group shows 0.6850. These discrepancies suggest that the model may carry historical biases present in the 1994 Census data. Users should be aware that relying on this model for decision-making could reinforce existing socioeconomic inequalities, and it should be used with caution when applied to underrepresented groups.

## Caveats and Recommendations

The primary caveat is that this model is based on 1994 Census data, which does not reflect current economic conditions, inflation, or modern job markets. It is recommended that this model be used primarily as a baseline or for educational purposes. For a production-level tool, more modern datasets and extensive hyperparameter tuning would be necessary to improve the recall and fairness across all demographic groups.
