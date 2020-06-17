import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Dense, Dropout, SpatialDropout2D, LSTM, Input, BatchNormalization, Activation

def crnn(input_shape, conv_layers, lstm_units,
               kernel_size=3, pool_size=2, reg_rate=0.2,
               hidden_dense=100
               ) -> Model:
    """ Creates cnn model with specified number of convolutional layers

    Parameters
    ----------
    input_shape : 2 integer tuple
        Shape of the input for first convolutional layer
    conv_layers : list or array
        Vector of filters for convolutional layers. The number of layers
        depends on the length of the vector
    lstm_units : int
    	Number of units in LSTM layer
    kernel_size : int
        Kernel size for convolutional layers
    pool_size : int
        Window size for pooling layers
    reg_rate: float
        Regularization rate
    num_classes : int
        Number of classes for classification problem. 
        Needed for output layer
        
    Returns
    -------
    keras.models.Model
        
    """

    model_input = Input((None, input_shape[1]), name='input')
    layer = model_input
    for num_filters in conv_layers:
    # give name to the layers
	    layer = Conv1D(
	            filters=num_filters,
	            kernel_size=kernel_size
	        )(layer)
	    layer = BatchNormalization(momentum=0.9)(layer)
	    layer = Activation('relu')(layer)
	    layer = MaxPooling1D(pool_size)(layer)
	    layer = Dropout(reg_rate)(layer)
    
    ## LSTM Layer
    layer = LSTM(lstm_units, return_sequences=False)(layer)
    layer = Dropout(reg_rate)(layer)
    
    ## Dense Layer
    layer = Dense(hidden_dense, activation='relu')(layer)
    layer = Dropout(reg_rate)(layer)
    
    ## Softmax Output
    layer = Dense(num_classes)(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = Model(model_input, model_output)
    # model = Sequential()
    
        
    # model.add(Conv1D(conv_layers[0], kernel_size, activation='relu', input_shape=input_shape+(1,)))
    # model.add(MaxPooling1D(pool_size))
    # model.add(Dropout(reg_rate))
    # for index, filters in enumerate(conv_layers[1:]):
    #     model.add(Conv1D(filters, kernel_size, activation='relu'))
    #     model.add(MaxPooling1D(pool_size))
    #     model.add(Dropout(reg_rate))
    # model.add(Reshape())
    # model.add(LSTM(lstm_units, return_sequences=False))
    # model.add(Dropout(reg_rate))

    # #model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(units=num_classes, activation='softmax'))
    
    return model

def simple_cnn(input_shape, conv_layers,
               kernel_size=3, pool_size=2,
               regularization='dropout', reg_rate=0.2,
               hidden_dense=100,
               num_classes=5
               ) -> Model:
    """ Creates cnn model with specified number of convolutional layers

    Parameters
    ----------
    input_shape : 2 integer tuple
        Shape of the input for first convolutional layer
    conv_layers : list or array
        Vector of filters for convolutional layers. The number of layers
        depends on the length of the vector
    kernel_size : int
        Kernel size for convolutional layers
    pool_size : int
        Window size for pooling layers
    regularization : str
        Type of regularization layers: 'dropout' or 'spacial'
    reg_rate: float
        Regularization rate
    num_classes : int
        Number of classes for classification problem. 
        Needed for output layer
        
    Returns
    -------
    keras.models.Model
        
    """

    model = Sequential()
    
    if regularization == 'spatial':
        regularization_layer = SpatialDropout2D(reg_rate)
    else:
        regularization_layer = Dropout(reg_rate)
        
    model.add(Conv2D(conv_layers[0], kernel_size, activation='relu', input_shape=input_shape+(1,)))
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(reg_rate))
    for index, filters in enumerate(conv_layers[1:]):
        model.add(Conv2D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size))
        model.add(Dropout(reg_rate))
        
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    return model

