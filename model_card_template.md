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

The model was evaluated using Precision, Recall, and the F1 Score. On the global test set, the model achieved a Precision of 0.7419, a Recall of 0.6384, and an F1 Score of 0.6863. Performance was also measured across data slices, revealing that the model performs best on the "United-States" demographic (F1: 0.6814) and significantly better on individuals with advanced degrees, such as Doctorates (F1: 0.8793), compared to those with lower education levels like 10th grade (F1: 0.2353).

## Ethical Considerations

The dataset used for this model contains sensitive demographic attributes, including race and sex. An analysis of model performance across these categories reveals significant disparities in predictive accuracy. For instance, the model achieves a higher F1 score of 0.6997 for individuals identified as Male, compared to an F1 score of 0.6015 for individuals identified as Female. Additionally, performance varies across racial groups, with the Black demographic group showing an F1 score of 0.6667, while the White demographic group shows 0.6850. These discrepancies suggest that the model may carry historical biases present in the 1994 Census data. Users should be aware that relying on this model for decision-making could reinforce existing socioeconomic inequalities, and it should be used with caution when applied to underrepresented groups.

## Caveats and Recommendations

A significant caveat is the model's reliance on 1994 Census data, which does not reflect modern economic realities or current income distributions. Additionally, the model shows a "Recall gap," meaning it is better at identifying people who make <=50K than those who make >50K. It is recommended that this model be used only as an educational baseline. For future iterations, I recommend performing hyperparameter tuning specifically to improve the Recall for underrepresented demographic slices and incorporating more recent socio-economic data.
