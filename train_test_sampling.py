import numpy as np
import pandas as pd

# define constant
INPUT_DIR = "./input/"
seed = 1

# read and drop the unnecessary columns, only keep the columns for two queries and similarity
train = pd.read_csv(INPUT_DIR+"train.csv")
train.drop(["id","qid1","qid2"], axis = 1, inplace=True)

# select only 50000 rows as train, 5000 as test (no dev set)
np.random.seed(seed)
sample_index = np.random.permutation(train.shape[0])[:5500]
X_train = train.ix[sample_index[:5000],0:2]
X_test = train.ix[sample_index[5000:],0:2]
Y_train = train.ix[sample_index[:5000],2].values
Y_test = train.ix[sample_index[5000:],2].values

# output to INPUT_DIR
X_train.to_csv(INPUT_DIR+"X_train.csv", index=False)
X_test.to_csv(INPUT_DIR+"X_test.csv", index=False)
np.save(INPUT_DIR+"Y_train.npy",Y_train)
np.save(INPUT_DIR+"Y_test.npy",Y_test)