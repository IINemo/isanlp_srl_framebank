from tensorflow.keras.layers import (
    Dense, LSTM, Convolution1D,
    Dropout, MaxPooling1D, BatchNormalization
)
from tensorflow.keras.layers import (
    Flatten, Input, TimeDistributed,
    Activation, RepeatVector, Permute,
    Lambda, Concatenate, Bidirectional, 
    Masking, concatenate, multiply
)
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, Model

class ModelZoo(object):
    
    @staticmethod
    def attention(input_shape, conv_size=200, lstm_size=80):
        _input = Input(shape = input_shape, dtype = 'float')

        conv = Convolution1D(filters=conv_size, 
                            kernel_size=2, 
                            padding='same', 
                            activation='relu')(_input)

        activations = LSTM(lstm_size, return_sequences=True)(conv)

        # compute importance for each step
        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)  
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(lstm_size)(attention)
        attention = Permute([2, 1])(attention)

            # apply the attention
        seq_repr = multiply([activations, attention])
        seq_repr = Lambda(lambda xin: K.sum(xin, axis=1))(seq_repr)
        seq_model = Model(inputs=[_input], outputs=[seq_repr])

        return seq_model
    
    @staticmethod
    def simple_model(input_shape, conv_filters=128, lstm_size=80,
                     dropout=0.1, hidden=60,
                     optimizer='adam', number_of_roles=34):
        
        model = Sequential()
        model.add(Convolution1D(filters=conv_filters, 
                            kernel_size=2, 
                            padding='same', 
                            activation='relu', 
                            input_shape = input_shape))

        model.add(LSTM(lstm_size))
        model.add(Dropout(dropout))
        model.add(Dense(hidden, activation='tanh'))
        model.add(Dense(number_of_roles, activation='softmax'))
        model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
        return model
    
    @staticmethod
    def simple_attentional_model(input_shape, conv_size=128,
                                 lstm_size=80, optimizer='adam',
                                 number_of_roles=34):
        
        
        model = Sequential()
        model.add(ModelZoo.attention(input_shape, conv_size, lstm_size))
        model.add(Dense(number_of_roles, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def plain_model(input_shape, hidden_1=600, hidden_2=400,
                    dropout_1=0.3, dropout_2=0.3, optimizer='adam',
                    number_of_roles=34):
        
        plain_model = Sequential()
        plain_model.add(Dense(hidden_1, 
                          #input_shape=(plain_features.shape[1],), 
                          input_shape = input_shape,
                          activation = 'relu'))
        plain_model.add(Dropout(dropout_1))
    
        plain_model.add(Dense(hidden_2))
        plain_model.add(BatchNormalization())
        plain_model.add(Activation('relu'))
        plain_model.add(Dropout(dropout_2))
    
        plain_model.add(Dense(number_of_roles))
        plain_model.add(BatchNormalization())
        plain_model.add(Activation('softmax'))
    
        plain_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
        return plain_model

    @staticmethod
    def sparse_model(categ_size, emb_size, 
                     hidden_plain=400, hidden_embed=100, 
                     hidden_2=400, dropout_1=0.3, dropout_2=0.3,
                     optimizer='adam', number_of_roles=34):
        
        input_plain = Input(shape=(categ_size,), name = 'input_categorical')
        input_pred_embed = Input(shape=(emb_size,), name = 'pred_embed')
        input_arg_embed = Input(shape=(emb_size,), name = 'arg_embed')
    
        plain = Dense(hidden_plain)(input_plain)
        plain = BatchNormalization()(plain)
        plain = Activation('relu')(plain)
    
        def embed_submodel(inpt):
            embed = Dense(hidden_embed)(inpt)
            embed = BatchNormalization()(embed)
            embed = Activation('relu')(embed)
            return embed
    
        embed_pred = embed_submodel(input_pred_embed)
        embed_arg = embed_submodel(input_arg_embed)
    
        final = Concatenate(axis = 1)([embed_pred, embed_arg, plain])
        final = Dropout(dropout_1)(final)
        final = Dense(hidden_2)(final)
        final = BatchNormalization()(final)
        final = Activation('relu')(final)
        final = Dropout(dropout_2)(final)
        final = Dense(number_of_roles)(final)
        final = BatchNormalization()(final)
        final = Activation('softmax')(final)
    
        model = Model([input_arg_embed, input_pred_embed, input_plain], final)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
        return model
    
    @staticmethod
    def graph_bidirectional_model(input_shape, plain_features_shape, conv_size=128,
                                  lstm_size=100, hidden_1=700,
                                  hidden_2=300, dropout_1=0.3,
                                  dropout_2=0.3, optimizer='adam', 
                                  number_of_roles=34):

        arg_context_model = Sequential()
        arg_context_model.add(
            Convolution1D(
                filters=conv_size, 
                kernel_size=2, 
                padding='same', 
                activation='relu',
                input_shape = input_shape
            )
        )
        arg_context_model.add(Bidirectional(LSTM(lstm_size), merge_mode = 'sum'))

        ###############################

        plain_model = Sequential()
        plain_model.add(Dense(hidden_1, 
                              input_shape=plain_features_shape, 
                              activation = 'relu'))

        ###############################

        final = Sequential()
        final.add(concatenate([arg_context_model, plain_model], axis=1))
        final.add(Dropout(dropout_1))

        final.add(Dense(hidden_2))
        final.add(BatchNormalization())
        final.add(Activation('relu'))
        final.add(Dropout(dropout_2))

        final.add(Dense(number_of_roles))
        final.add(BatchNormalization())
        final.add(Activation('softmax'))

        final.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return final

    def graph_attentional_model(arg_context_shape, pred_context_shape,
                                plain_features_shape, hidden_plain=800,
                                hidden_final=400, dropout_1=0.3, dropout_2=0.3,
                                number_of_roles=34, optimizer='adam'):

        arg_context_model = ModelZoo.attention(input_shape=arg_context_shape)
        pred_context_model = ModelZoo.attention(input_shape=pred_context_shape)

        ###############################

        plain_model = Sequential()
        plain_model.add(Dense(hidden_plain, 
                              input_shape=plain_features_shape, 
                              activation = 'relu'))


        ###############################

        final = Sequential()
        final.add(Concatenate(axis=1)([arg_context_model, pred_context_model, plain_model.output]))
        final.add(Dropout(dropout_1))

        #final.add(Dense(300, activation = 'relu'))
        final.add(Dense(hidden_final))
        final.add(BatchNormalization())
        final.add(Activation('relu'))
        final.add(Dropout(dropout_2))

        final.add(Dense(number_of_roles))
        final.add(BatchNormalization())
        final.add(Activation('softmax'))

        final.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return final