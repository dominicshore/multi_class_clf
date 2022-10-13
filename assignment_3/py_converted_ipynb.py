# %%
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from random import randrange
sns.set(rc={'figure.figsize':(15,8)})

# %%
df = pd.read_csv("./data/abalone.data", names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])

# %%
raw_abs = df.copy()
raw_abs

# %%
raw_abs.loc[raw_abs['rings'].between(0, 7,inclusive='both'), 'ring_class'] = int(1)
raw_abs.loc[raw_abs['rings'].between(8, 10,inclusive='both'), 'ring_class'] = int(2)
raw_abs.loc[raw_abs['rings'].between(11, 15,inclusive='both'), 'ring_class'] = int(3)
raw_abs.loc[raw_abs['rings'] > 15, 'ring_class'] = int(4)
raw_abs['ring_class'] = raw_abs['ring_class'].astype(int)

# %%
raw_abs.drop(columns = "rings", axis=1, inplace=True)


# %%
raw_abs = raw_abs[(raw_abs['height']<0.4) & (raw_abs['height']>0.01)]

# %% [markdown]
# # Visualisations

# %% [markdown]
# ### Distribtion of ring class

# %%
raw_abs.head()

# %%
raw_abs['ring_class'].value_counts().plot(kind='barh', figsize=(8,6))
plt.ylabel("Ring-class")
plt.xlabel("Class count")
plt.grid(False)
plt.title("Abalone ring-class count", y=1.02, fontsize = 18);

# %%
raw_abs.hist()

# %%
heatmap = sns.heatmap(raw_abs.corr(), annot=True, cbar=False, vmin=-1., vmax=1., cmap=sns.cm.rocket)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)

# %% [markdown]
# # Building the model

# %%
num_pipeline = Pipeline([
    ('Nomalisation', MinMaxScaler()),
    ])

sex_pipeline = Pipeline([
    ('ord_encoder', OrdinalEncoder(categories=[['M', 'F', 'I']]))
])

ringClass_pipeline = Pipeline([
    ('ringClass_1Hot', OneHotEncoder())
])

num_arribs = list(raw_abs.drop(columns=["sex", "ring_class"]))

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_arribs),
    ("sex", sex_pipeline, ['sex']),
    ("ringClass_1Hot", ringClass_pipeline, ['ring_class'])
])



# %%
abs_prepared = pd.DataFrame(full_pipeline.fit_transform(raw_abs))
abs_prepared

# %%
X = abs_prepared.iloc[:,:-4]
y = abs_prepared.iloc[:,-4:]

# %%
model_results = []

# Single layer
def run_model(hidden_neurons, lr=0.01, experimental_runs = 10, hid_layers = 1):
    
    for expi_run in range(experimental_runs):
        
        random_seed = randrange(0,100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.4, random_state=random_seed)

        model = keras.models.Sequential([
        keras.layers.Dense(hidden_neurons, input_shape = (8,), activation = "relu"),
        # keras.layers.Dense(32, activation = "relu"),
        keras.layers.Dense(4, activation = "softmax")
        ])

        model.compile(loss = 'categorical_crossentropy',
                optimizer = keras.optimizers.SGD(learning_rate=lr),
                metrics = ['accuracy']
                )

        history = model.fit(X_train, y_train, epochs=100, verbose=0)

        mod_eval = model.evaluate(X_test, y_test)

        model_results.append([expi_run, hidden_neurons, lr, hid_layers, mod_eval[1]])

        # if ((expi_run>10) & (max(history.history['accuracy']) >= max(model_results['max_acc']))):
        #     pd.DataFrame(history.history).plot(figsize=(8,5))
        #     plt.grid(True)
        #     plt.gca().set_ylim(0,1)
        #     plt.title(str(hid_layers) + " layer network with" + str(hidden_neurons) +" hidden nurons. LR = "+ str(lr) + " with " + str(round(history.history['accuracy'][-1], 4)*100) + " % Acc.")
        #     plt.savefig(".\\images\\" + str(hid_layers) +"_layer_" + str(hidden_neurons) +"_hidden_nur" + str(lr) + "_lr.png", bbox_inches='tight')
        #     plt.close()

list(map(lambda x: run_model(hidden_neurons = x), [5,10,15,20]))

temp_results = pd.DataFrame(model_results, columns=["iter", 'hidden', 'lr', 'layers', 'test_acc'])
sing_lay_neuron_stats = temp_results.groupby(['hidden', 'lr', 'layers'])['test_acc'].describe()

# %%
sing_lay_neuron_stats[sing_lay_neuron_stats['mean']== sing_lay_neuron_stats['mean'].max()]

# %%


# %%
model_results = []

# Single layer
def run_model(hidden_neurons, lr=0.01, experimental_runs = 10, hid_layers = 1):
    
    for expi_run in range(experimental_runs):
        
        random_seed = randrange(0,100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.4, random_state=random_seed)

        model = keras.models.Sequential([
        keras.layers.Dense(hidden_neurons, input_shape = (8,), activation = "relu"),
        # keras.layers.Dense(32, activation = "relu"),
        keras.layers.Dense(4, activation = "softmax")
        ])

        model.compile(loss = 'categorical_crossentropy',
                optimizer = keras.optimizers.SGD(learning_rate=lr),
                metrics = ['accuracy']
                )

        history = model.fit(X_train, y_train, epochs=100, verbose=0)

        mod_eval = model.evaluate(X_test, y_test)

        model_results.append([expi_run, hidden_neurons, lr, hid_layers, mod_eval[1]])

        # if ((expi_run>10) & (max(history.history['accuracy']) >= max(model_results['max_acc']))):
        #     pd.DataFrame(history.history).plot(figsize=(8,5))
        #     plt.grid(True)
        #     plt.gca().set_ylim(0,1)
        #     plt.title(str(hid_layers) + " layer network with" + str(hidden_neurons) +" hidden nurons. LR = "+ str(lr) + " with " + str(round(history.history['accuracy'][-1], 4)*100) + " % Acc.")
        #     plt.savefig(".\\images\\" + str(hid_layers) +"_layer_" + str(hidden_neurons) +"_hidden_nur" + str(lr) + "_lr.png", bbox_inches='tight')
        #     plt.close()

list(map(lambda x: run_model(hidden_neurons = 20, lr = x), [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))


temp_results = pd.DataFrame(model_results, columns=["iter", 'hidden', 'lr', 'layers', 'test_acc'])
lr_results = temp_results.groupby(['hidden', 'lr', 'layers'])['test_acc'].describe()

# %%
lr_results[lr_results['mean']== lr_results['mean'].max()]

# %%
model_results = []

# Single layer
def run_model(hidden_neurons=20, lr=0.01, experimental_runs = 10, hid_layers = 2):
    
    for expi_run in range(experimental_runs):
        
        random_seed = randrange(0,100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.4, random_state=random_seed)

        model = keras.models.Sequential([
        keras.layers.Dense(hidden_neurons, input_shape = (8,), activation = "relu"),
        keras.layers.Dense(32, activation = "relu"),
        keras.layers.Dense(4, activation = "softmax")
        ])

        model.compile(loss = 'categorical_crossentropy',
                optimizer = keras.optimizers.SGD(learning_rate=lr),
                metrics = ['accuracy']
                )

        history = model.fit(X_train, y_train, epochs=100, verbose=0)

        mod_eval = model.evaluate(X_test, y_test)

        model_results.append([expi_run, hidden_neurons, lr, hid_layers, mod_eval[1]])

        # if ((expi_run>10) & (max(history.history['accuracy']) >= max(model_results['max_acc']))):
        #     pd.DataFrame(history.history).plot(figsize=(8,5))
        #     plt.grid(True)
        #     plt.gca().set_ylim(0,1)
        #     plt.title(str(hid_layers) + " layer network with" + str(hidden_neurons) +" hidden nurons. LR = "+ str(lr) + " with " + str(round(history.history['accuracy'][-1], 4)*100) + " % Acc.")
        #     plt.savefig(".\\images\\" + str(hid_layers) +"_layer_" + str(hidden_neurons) +"_hidden_nur" + str(lr) + "_lr.png", bbox_inches='tight')
        #     plt.close()

run_model()


temp_results = pd.DataFrame(model_results, columns=["iter", 'hidden', 'lr', 'layers', 'test_acc'])
two_lay_results = temp_results.groupby(['hidden', 'lr', 'layers'])['test_acc'].describe()
two_lay_results

# %%
model_results = []

# Single layer
def run_model(hidden_neurons=20, lr=0.01, experimental_runs = 10, hid_layers = 2):
    
    for expi_run in range(experimental_runs):
        
        random_seed = randrange(0,100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.4, random_state=random_seed)

        model = keras.models.Sequential([
        keras.layers.Dense(hidden_neurons, input_shape = (8,), activation = "relu"),
        keras.layers.Dense(32, activation = "relu"),
        keras.layers.Dense(4, activation = "softmax")
        ])

        model.compile(loss = 'categorical_crossentropy',
                optimizer = keras.optimizers.Adam(learning_rate=lr),
                metrics = ['accuracy']
                )

        history = model.fit(X_train, y_train, epochs=100, verbose=0)

        mod_eval = model.evaluate(X_test, y_test)

        model_results.append([expi_run, hidden_neurons, lr, hid_layers, mod_eval[1]])

        # if ((expi_run>10) & (max(history.history['accuracy']) >= max(model_results['max_acc']))):
        #     pd.DataFrame(history.history).plot(figsize=(8,5))
        #     plt.grid(True)
        #     plt.gca().set_ylim(0,1)
        #     plt.title(str(hid_layers) + " layer network with" + str(hidden_neurons) +" hidden nurons. LR = "+ str(lr) + " with " + str(round(history.history['accuracy'][-1], 4)*100) + " % Acc.")
        #     plt.savefig(".\\images\\" + str(hid_layers) +"_layer_" + str(hidden_neurons) +"_hidden_nur" + str(lr) + "_lr.png", bbox_inches='tight')
        #     plt.close()

run_model()


temp_results = pd.DataFrame(model_results, columns=["iter", 'hidden', 'lr', 'layers', 'test_acc'])
two_lay_adam_results = temp_results.groupby(['hidden', 'lr', 'layers'])['test_acc'].describe()
two_lay_adam_results


