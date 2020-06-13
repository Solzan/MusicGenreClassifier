import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout, SpatialDropout2D


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

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
    
    
    EPOCHS = 100
    working_directory = os.path.dirname(os.getcwd())
    checkpoint_filepath = os.path.join(working_directory, "Checkpoint") 
    
    # Model weights are saved at the end of every epoch, if it's the best seen so far.
    # The model weights (that are considered the best) are loaded into the model.
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
    
    model_learningScheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model_earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        min_delta=0, 
        patience=10,        #stopping training after 10 not improved consecutive epochs 
        verbose=0, 
        mode='auto',
        baseline=None, 
        restore_best_weights=False)
    


    if regularization == 'spatial':
        regularization_layer = SpatialDropout2D(reg_rate)
    else:
        regularization_layer = Dropout(reg_rate)
        
    model.add(Conv2D(conv_layers[0], kernel_size, activation='relu', input_shape=input_shape+(1,)))
    model.add(MaxPooling2D(pool_size))
    model.add(regularization_layer)
    
    for index, filter in enumerate(conv_layers[1:]):
        model.add(Conv2D(conv_layers[index], kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size))
        model.add(regularization_layer)
        
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    
    model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback, model_learningScheduler_callback, model_earlyStopping_callback])
    model.load_weights(checkpoint_filepath)

    
    return model