from tensorflow.keras import layers as nn
import tensorflow as tf

class SimpleModelFactory(object):
    def __init__(self, input_dim, hidden_1=600, hidden_2=400, dropout_1=0.3, dropout_2=0.3, number_of_roles=34):
        self.input_dim=input_dim
        self.hidden_1=hidden_1
        self.hidden_2=hidden_2
        self.dropout_1=dropout_1
        self.dropout_2=dropout_2
        self.number_of_roles=number_of_roles
        
    def create(self):
        model = tf.keras.models.Sequential()
        model.add(nn.Dense(self.hidden_1, activation=tf.keras.activations.relu, input_dim=self.input_dim))
        model.add(nn.Dropout(rate=self.dropout_1))
        model.add(nn.BatchNormalization())
        model.add(nn.Dense(self.hidden_2))
        model.add(nn.BatchNormalization())
        model.add(nn.Activation(tf.keras.activations.relu))
        model.add(nn.Dropout(self.dropout_2))
        model.add(nn.Dense(self.number_of_roles))
        model.add(nn.BatchNormalization())
        model.add(nn.Activation(tf.keras.activations.softmax))


        return model

    def create_and_compile(self):
        model = self.create()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


class SparseModelFactory(object):
    def __init__(self, input_dims: list, hidden_categorical=400, hidden_embeddings=100, hidden_concat=400, dropout=0.3, number_of_roles=34):
        self.input_dims = input_dims
        self.hidden_categorical=hidden_categorical
        self.hidden_embeddings=hidden_embeddings
        self.hidden_concat=hidden_concat
        self.dropout=dropout
        self.number_of_roles=number_of_roles
        
    def create(self):
        input_categorical = nn.Input(shape=(self.input_dims[0],), name='input-categorical')
        input_pred = nn.Input(shape=(self.input_dims[1],), name='input-predicate-embeddings')
        input_arg = nn.Input(shape=(self.input_dims[2],), name='')

        categorical = nn.Dense(self.hidden_categorical)(input_categorical)
        categorical = nn.BatchNormalization()(categorical)
        categorical = nn.Activation(tf.keras.activations.relu)(categorical)

        pred = nn.Dense(self.hidden_embeddings)(input_pred)
        pred = nn.BatchNormalization()(pred)
        pred = nn.Activation(tf.keras.activations.relu)(pred)

        arg = nn.Dense(self.hidden_embeddings)(input_arg)
        arg = nn.BatchNormalization()(arg)
        arg = nn.Activation(tf.keras.activations.relu)(arg)

        concat = nn.Concatenate(axis=1)([pred, arg, categorical])
        concat = nn.Dropout(self.dropout)(concat)
        concat = nn.Dense(self.hidden_concat)(concat)
        concat = nn.BatchNormalization()(concat)
        concat = nn.Activation(tf.keras.activations.relu)(concat)

        final = nn.Dropout(self.dropout)(concat)
        final = nn.Dense(self.number_of_roles)(final)
        final = nn.BatchNormalization()(final)
        final = nn.Activation(tf.keras.activations.softmax)(final)

        return tf.keras.Model(inputs=[input_categorical, input_pred, input_arg], outputs=[final])

    def create_and_compile(self):
        model = self.create()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    