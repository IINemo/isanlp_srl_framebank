from tensorflow.keras import layers as nn
import tensorflow as tf

class SimpleModel(tf.keras.Model):
    def __init__(self, hidden_1=600, hidden_2=400, dropout_1=0.3, dropout_2=0.3, number_of_roles=34):
        super(SimpleModel, self).__init__(name='simple_model')
        self.fc1 = nn.Dense(units=hidden_1, activation=tf.nn.relu)
        self.dropout1 = nn.Dropout(rate=dropout_1)
        self.dropout2 = nn.Dropout(rate=dropout_2)
        self.fc2 = nn.Dense(units=hidden_2)
        self.bn1 = nn.BatchNormalization()
        self.fc3 = nn.Dense(units=number_of_roles)
        self.bn2 = nn.BatchNormalization()
        
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = tf.nn.relu(self.bn1(x))
        x = self.dropout2(x)
        x = self.bn2(self.fc3(x))
        return tf.nn.softmax(x)

class SparseModel(tf.keras.Model):
    def __init__(self, hidden_categorical=400, hidden_embeddings=100, hidden_final=400, dropout_1=0.3, dropout_2=0.3, number_of_roles=34):
        super(SparseModel, self).__init__(name="sparse_model")
        self.fc_categorical = nn.Dense(units=hidden_categorical)
        self.fc_pred_embedding = nn.Dense(units=hidden_embeddings)
        self.fc_arg_embedding = nn.Dense(units=hidden_embeddings)
        self.fc_concat = nn.Dense(units=hidden_final)
        self.fc_final = nn.Dense(units=number_of_roles)
        self.dropout1 = nn.Dropout(rate=dropout_1)
        self.dropout2 = nn.Dropout(rate=dropout_2)
        self.batch_norms = {f'bn_{i}': nn.BatchNormalization() for i in ['categorical', 'embed_pred', 'embed_arg', 'concat', 'final']}
        
    def call(self, inputs):
        embed_pred, embed_arg, categorical = inputs
        embed_pred = self.fc_pred_embedding(embed_pred)
        embed_pred = tf.nn.relu(self.batch_norms['bn_embed_pred'](embed_pred))
        
        embed_arg = self.fc_arg_embedding(embed_arg)
        embed_arg = tf.nn.relu(self.batch_norms['bn_embed_arg'](embed_arg))
        
        categorical = self.fc_categorical(categorical)
        categorical = tf.nn.relu(self.batch_norms['bn_categorical'](categorical))
        
        concat = tf.concat([embed_pred, embed_arg, categorical], axis=1)
        concat = self.dropout1(concat)
        concat = self.fc_concat(concat)
        concat = tf.nn.relu(self.batch_norms['bn_concat'](concat))
        
        final = self.dropout2(concat)
        final = self.fc_final(final)
        final = tf.nn.softmax(self.batch_norms['bn_final'](final))
        
        return final
    