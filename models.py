import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout, SpatialDropout2D


def simple_cnn(input_shape, conv_layers,
               kernel_size=3, pool_size=2,
               regularization='dropout', reg_rate=0.2,
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
    model.add(regularization_layer)
    for index, filters in enumerate(conv_layers[1:]):
        model.add(Conv2D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size))
        model.add(regularization_layer)
        
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    return model