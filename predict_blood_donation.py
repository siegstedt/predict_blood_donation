#!/usr/bin/env python
# coding: utf-8

# ## 1. Inspecting transfusion.data file
# <p><img src="https://assets.datacamp.com/production/project_646/img/blood_donation.png" style="float: right;" alt="A pictogram of a blood bag with blood donation written in it" width="200"></p>
# <p>Blood transfusion saves lives - from replacing lost blood during major surgery or a serious injury to treating various illnesses and blood disorders. Ensuring that there's enough blood in supply whenever needed is a serious challenge for the health professionals. According to <a href="https://www.webmd.com/a-to-z-guides/blood-transfusion-what-to-know#1">WebMD</a>, "about 5 million Americans need a blood transfusion every year".</p>
# <p>Our dataset is from a mobile blood donation vehicle in Taiwan. The Blood Transfusion Service Center drives to different universities and collects blood as part of a blood drive. We want to predict whether or not a donor will give blood the next time the vehicle comes to campus.</p>
# <p>The data is stored in <code>datasets/transfusion.data</code> and it is structured according to RFMTC marketing model (a variation of RFM). We'll explore what that means later in this notebook. First, let's inspect the data.</p>

# In[53]:


# Print out the first 5 lines from the transfusion.data file
get_ipython().system('head -n 5 datasets/transfusion.data')


# In[54]:


get_ipython().run_cell_magic('nose', '', '\nlast_input = In[-2]\n\nimport re\ntry:\n    bash_cmd = re.search(r\'get_ipython\\(\\).system\\(\\\'(.*)\\\'\\)\', last_input).group(1)\nexcept AttributeError:\n    bash_cmd = \'\'\n\ndef test_head_command():\n    assert \'head\' in bash_cmd, \\\n        "Did you use \'head\' command?"\n    assert \'-n\' in bash_cmd, \\\n        "Did you use \'-n\' parameter?"\n    assert \'5\' in bash_cmd, \\\n        "Did you specify the correct number of lines to print?"')


# ## 2. Loading the blood donations data
# <p>We now know that we are working with a typical CSV file (i.e., the delimiter is <code>,</code>, etc.). We proceed to loading the data into memory.</p>

# In[55]:


# Import pandas
import pandas as pd

# Read in dataset
transfusion = pd.read_csv("datasets/transfusion.data")

# Print out the first rows of our dataset
transfusion.head()


# In[56]:


get_ipython().run_cell_magic('nose', '', '\nlast_output = _\n\ndef test_pandas_loaded():\n    assert \'pd\' in globals(), \\\n        "\'pd\' module not found. Please check your import statement."\n\ndef test_transfusion_loaded():\n    correct_transfusion = pd.read_csv("datasets/transfusion.data")\n    assert correct_transfusion.equals(transfusion), \\\n        "transfusion not loaded correctly."\n    \ndef test_head_output():\n    try:\n        assert "6000" in last_output.to_string()\n    except AttributeError:\n        assert False, \\\n            "Please use transfusion.head() as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "Hmm, the output of the cell is not what we expected. You should see 6000 in the first five rows of the transfusion DataFrame."')


# ## 3. Inspecting transfusion DataFrame
# <p>Let's briefly return to our discussion of RFM model. RFM stands for Recency, Frequency and Monetary Value and it is commonly used in marketing for identifying your best customers. In our case, our customers are blood donors.</p>
# <p>RFMTC is a variation of the RFM model. Below is a description of what each column means in our dataset:</p>
# <ul>
# <li>R (Recency - months since the last donation)</li>
# <li>F (Frequency - total number of donation)</li>
# <li>M (Monetary - total blood donated in c.c.)</li>
# <li>T (Time - months since the first donation)</li>
# <li>a binary variable representing whether he/she donated blood in March 2007 (1 stands for donating blood; 0 stands for not donating blood)</li>
# </ul>
# <p>It looks like every column in our DataFrame has the numeric type, which is exactly what we want when building a machine learning model. Let's verify our hypothesis.</p>

# In[57]:


# Print a concise summary of transfusion DataFrame
transfusion.info()


# In[58]:


get_ipython().run_cell_magic('nose', '', '\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_info_command():\n    assert \'transfusion\' in last_input, \\\n        "Expected transfusion variable in your input."\n    assert \'info\' in last_input, \\\n        "Did you use the correct method?"\n    assert \'print\' not in last_input, \\\n        "Please use transfusion.info() to inspect DataFrame\'s structure, not the display() or print() functions."')


# ## 4. Creating target column
# <p>We are aiming to predict the value in <code>whether he/she donated blood in March 2007</code> column. Let's rename this it to <code>target</code> so that it's more convenient to work with.</p>

# In[59]:


# Rename target column as 'target' for brevity 
transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)

# Print out the first 2 rows
transfusion.head(2)


# In[60]:


get_ipython().run_cell_magic('nose', '', '\nlast_output = _\n\ndef test_target_column_added():\n    assert \'target\' in transfusion.columns, \\\n        "\'target\' column not found in transfusion.columns"\n\ndef test_head_2_rows_only():\n    try:\n        assert last_output.shape[0] == 2\n    except AttributeError:\n        assert False, \\\n            "Please use transfusion.head(2) as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "Did you call \'head()\' method with the correct number of lines?"')


# ## 5. Checking target incidence
# <p>We want to predict whether or not the same donor will give blood the next time the vehicle comes to campus. The model for this is a binary classifier, meaning that there are only 2 possible outcomes:</p>
# <ul>
# <li><code>0</code> - the donor will not give blood</li>
# <li><code>1</code> - the donor will give blood</li>
# </ul>
# <p>Target incidence is defined as the number of cases of each individual target value in a dataset. That is, how many 0s in the target column compared to how many 1s? Target incidence gives us an idea of how balanced (or imbalanced) is our dataset.</p>

# In[61]:


# Print target incidence proportions, rounding output to 3 decimal places
transfusion.target.value_counts(normalize=True).round(3)


# In[62]:


get_ipython().run_cell_magic('nose', '', '\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\nlast_output = _\n\ndef test_command_syntax():\n    assert \'transfusion.target\' in last_input, \\\n        "Did you call \'value_counts()\' method on \'transfusion.target\' column?"\n    assert \'value_counts(normalize=True)\' in last_input, \\\n        "Did you use \'normalize=True\' parameter?"\n    assert \'round\' in last_input, \\\n        "Did you call \'round()\' method?"\n    assert \'round(3)\' in last_input, \\\n        "Did you call \'round()\' method with the correct argument?"\n    assert last_input.find(\'value\') < last_input.find(\'round\'), \\\n        "Did you chain \'value_counts()\' and \'round()\' methods in the correct order?"\n\ndef test_command_output():\n    try:\n        assert "0.762" in last_output.to_string()\n    except AttributeError:\n        assert False, \\\n            "Please use transfusion.target.value_counts(normalize=True).round(3) to inspect proportions, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "Hmm, the output of the cell is not what we expected. You should see 0.762 in your output."')


# ## 6. Splitting transfusion into train and test datasets
# <p>We'll now use <code>train_test_split()</code> method to split <code>transfusion</code> DataFrame.</p>
# <p>Target incidence informed us that in our dataset <code>0</code>s appear 76% of the time. We want to keep the same structure in train and test datasets, i.e., both datasets must have 0 target incidence of 76%. This is very easy to do using the <code>train_test_split()</code> method from the <code>scikit learn</code> library - all we need to do is specify the <code>stratify</code> parameter. In our case, we'll stratify on the <code>target</code> column.</p>

# In[63]:


# Import train_test_split method
from sklearn.model_selection import train_test_split

# Split transfusion DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion.target
)

# Print out the first 2 rows of X_train
X_train.head(2)


# In[64]:


get_ipython().run_cell_magic('nose', '', '\nlast_output = _\n\ndef test_train_test_split_loaded():\n    assert \'train_test_split\' in globals(), \\\n        "\'train_test_split\' function not found. Please check your import statement."\n\ndef test_X_train_created():\n    correct_X_train, _, _, _ = train_test_split(transfusion.drop(columns=\'target\'),\n                                                transfusion.target,\n                                                test_size=0.25,\n                                                random_state=42,\n                                                stratify=transfusion.target)\n    assert correct_X_train.equals(X_train), \\\n        "\'X_train\' not created correctly. Did you stratify on the correct column?"\n    \ndef test_head_output():\n    try:\n        assert "1750" in last_output.to_string()\n    except AttributeError:\n        assert False, \\\n            "Please use X_train.head(2) as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "Hmm, the output of the cell is not what we expected. You should see 1750 in the first 2 rows of the X_train DataFrame."')


# ## 7. Selecting model using TPOT
# <p><a href="https://github.com/EpistasisLab/tpot">TPOT</a> is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.</p>
# <p><img src="https://assets.datacamp.com/production/project_646/img/tpot-ml-pipeline.png" alt="TPOT Machine Learning Pipeline"></p>
# <p>TPOT will automatically explore hundreds of possible pipelines to find the best one for our dataset. Note, the outcome of this search will be a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">scikit-learn pipeline</a>, meaning it will include any pre-processing steps as well as the model.</p>
# <p>We are using TPOT to help us zero in on one model that we can then explore and optimize further.</p>

# In[65]:


# Import TPOTClassifier and roc_auc_score
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')


# In[66]:


get_ipython().run_cell_magic('nose', '', '\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_TPOTClassifier_loaded():\n    assert \'TPOTClassifier\' in globals(), \\\n        "\'TPOTClassifier\' class not found. Please check your import statement."\n    \ndef test_roc_auc_score_loaded():\n    assert \'roc_auc_score\' in globals(), \\\n        "\'roc_auc_score\' function not found. Please check your import statement."\n\ndef test_TPOTClassifier_instantiated():\n    assert isinstance(tpot, TPOTClassifier), \\\n        "\'tpot\' is not an instance of TPOTClassifier. Did you assign an instance of TPOTClassifier to \'tpot\' variable?"\n\ndef test_pipeline_steps_printed():\n    assert \'{idx}\' in last_input, \\\n        "Did you use {idx} variable in the f-string in the print statement?"\n    assert \'{transform}\' in last_input, \\\n        "Did you use {transform} variable in the f-string in the print statement?"')


# ## 8. Checking the variance
# <p>TPOT picked <code>LogisticRegression</code> as the best model for our dataset with no pre-processing steps, giving us the AUC score of 0.7850. This is a great starting point. Let's see if we can make it better.</p>
# <p>One of the assumptions for linear regression models is that the data and the features we are giving it are related in a linear fashion, or can be measured with a linear distance metric. If a feature in our dataset has a high variance that's an order of magnitude or more greater than the other features, this could impact the model's ability to learn from other features in the dataset.</p>
# <p>Correcting for high variance is called normalization. It is one of the possible transformations you do before training a model. Let's check the variance to see if such transformation is needed.</p>

# In[67]:


# X_train's variance, rounding the output to 3 decimal places
X_train.var().round(3)


# In[68]:


get_ipython().run_cell_magic('nose', '', '\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\nlast_output = _\n\ndef test_command_syntax():\n    assert \'X_train\' in last_input, \\\n        "Did you call \'var()\' method on \'X_train\' DataFrame?"\n    assert \'var\' in last_input, \\\n        "Did you call \'var()\' method?"\n    assert \'round(3)\' in last_input, \\\n        "Did you call \'round()\' method with the correct argument?"\n    assert last_input.find(\'var\') < last_input.find(\'round\'), \\\n        "Did you chain \'var()\' and \'round()\' methods in the correct order?"\n\ndef test_var_output():\n    try:\n        assert "2114363" in last_output.to_string()\n    except AttributeError:\n        assert False, \\\n            "Please use X_train.var().round(3) to inspect the variance, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "Hmm, the output of the cell is not what we expected. You should see 2114363 in your output."')


# ## 9. Log normalization
# <p><code>Monetary (c.c. blood)</code>'s variance is very high in comparison to any other column in the dataset. This means that, unless accounted for, this feature may get more weight by the model (i.e., be seen as more important) than any other feature.</p>
# <p>One way to correct for high variance is to use log normalization.</p>

# In[69]:


# Import numpy
import numpy as np

# Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = X_train_normed.var().idxmax(axis=1)

# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns=col_to_normalize, inplace=True)

# Check the variance for X_train_normed
X_train_normed.var().round(3)


# In[70]:


get_ipython().run_cell_magic('nose', '', '\nlast_output = _\n\ndef test_numpy_loaded():\n    assert \'np\' in globals(), \\\n        "\'np\' module not found. Please check your import statement."\n\ndef test_X_train_normed_created():\n    assert \'X_train_normed\' in globals(), \\\n        "\'X_train_normed\' DataFrame not found. Please check your variable assignment statement."\n\ndef test_col_to_normalize():\n    assert col_to_normalize == \'Monetary (c.c. blood)\', \\\n        "\'col_to_normalize\' is set to an incorrect column name."\n\ndef test_X_train_normed_log_normalized():\n    correct_X_train_normed = X_train.copy() \\\n        .assign(monetary_log = lambda x: np.log(x[\'Monetary (c.c. blood)\'])) \\\n        .drop(columns=\'Monetary (c.c. blood)\')\n    assert correct_X_train_normed.equals(X_train_normed), \\\n        "\'X_train_normed\' is incorrect. Are you \'col_to_normalize\' in the loop?"\n\ndef test_var_output():\n    try:\n        assert "611.147" in last_output.to_string()\n    except AttributeError:\n        assert False, \\\n            "Please use X_train_normed.var().round(3) as the last line of code in the cell to inspect the variance, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "Hmm, the output of the cell is not what we expected. You should see 611.147 in your output."')


# ## 10. Training the linear regression model
# <p>The variance looks much better now. Notice that now <code>Time (months)</code> has the largest variance, but it's not the <a href="https://en.wikipedia.org/wiki/Order_of_magnitude">orders of magnitude</a> higher than the rest of the variables, so we'll leave it as is.</p>
# <p>We are now ready to train the linear regression model.</p>

# In[71]:


# Importing modules
from sklearn import linear_model

# Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

# Train the model
logreg.fit(X_train_normed, y_train)

# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


# In[72]:


get_ipython().run_cell_magic('nose', '', '\ndef test_linear_model_loaded():\n    assert \'linear_model\' in globals(), \\\n        "\'linear_model\' module not found. Please check your import statement."\n    \ndef test_roc_auc_score_loaded():\n    assert \'roc_auc_score\' in globals(), \\\n        "\'roc_auc_score\' function not found. Please check your import statement."\n\ndef test_LogisticRegression_instantiated():\n    assert isinstance(logreg, linear_model.LogisticRegression), \\\n        ("\'logreg\' is not an instance of linear_model.LogisticRegression. "\n         "Did you assign an instance of linear_model.LogisticRegression to \'logreg\' variable?")\n\ndef test_model_fitted():\n    assert hasattr(logreg, \'coef_\'), \\\n        "Did you call \'fit()\' method on \'logreg\'?"\n\ndef test_logreg_auc_score():\n    assert \'{:.4f}\'.format(logreg_auc_score) == \'0.7891\', \\\n        "Hmm, the logreg_auc_score is not what we expected. You should see \'AUC score: 0.7891\' printed out."')


# ## 11. Conclusion
# <p>The demand for blood fluctuates throughout the year. As one <a href="https://www.kjrh.com/news/local-news/red-cross-in-blood-donation-crisis">prominent</a> example, blood donations slow down during busy holiday seasons. An accurate forecast for the future supply of blood allows for an appropriate action to be taken ahead of time and therefore saving more lives.</p>
# <p>In this notebook, we explored automatic model selection using TPOT and AUC score we got was 0.7850. This is better than simply choosing <code>0</code> all the time (the target incidence suggests that such a model would have 76% success rate). We then log normalized our training data and improved the AUC score by 0.5%. In the field of machine learning, even small improvements in accuracy can be important, depending on the purpose.</p>
# <p>Another benefit of using logistic regression model is that it is interpretable. We can analyze how much of the variance in the response variable (<code>target</code>) can be explained by other variables in our dataset.</p>

# In[73]:


# Importing itemgetter
from operator import itemgetter

# Sort models based on their AUC score from highest to lowest
sorted(
    [('tpot', tpot_auc_score.round(4)), ('logreg', logreg_auc_score.round(4))],
    key=itemgetter(1),
    reverse=True
)


# In[74]:


get_ipython().run_cell_magic('nose', '', '\nlast_output = _\n\ndef test_itemgetter_loaded():\n    assert \'itemgetter\' in globals(), \\\n        "\'itemgetter\' function not found. Please check your import statement."\n\ndef test_logreg_is_first_in_the_list():\n    assert last_output[0][0] == \'logreg\', \\\n        "Expected \'logreg\' to be first in the list."\n\ndef test_logreg_score():\n    assert round(last_output[0][1], 4) == 0.7891, \\\n        "Hmm, the output of the cell is not what we expected. You should see 0.7851 as \'logreg\' score in your output."')

