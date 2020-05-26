import os
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping



def trainDataGenerator():
  train_data_dir = './ashish/train/'

      # Let's use some data augmentaiton 
  train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

  img_rows, img_cols = 224, 224 
  batch_size = 32

  train_generator = train_datagen.flow_from_directory(
          train_data_dir,
          target_size=(img_rows, img_cols),
          batch_size=batch_size,
          class_mode='categorical')

  return train_generator



def validDataGenerator():
  validation_data_dir = './ashish/validation/'

  validation_datagen = ImageDataGenerator(rescale=1./255)

  img_rows, img_cols = 224, 224 
  batch_size = 32

  validation_generator = validation_datagen.flow_from_directory(
          validation_data_dir,
          target_size=(img_rows, img_cols),
          batch_size=batch_size,
          class_mode='categorical')

  return validation_generator


def buidingModel(bottom_model, num_classes, num_layers):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    layerWidth = os.getenv('WIDTH').split(',')
    index = 0
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    while index < num_layers:
        top_model = Dense(int(layerWidth[index]),activation='relu')(top_model)
        index = index + 1
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model


def trainModel(train_generator, validation_generator):
  checkpoint = ModelCheckpoint("New_Face_recog.h5",
                         monitor="val_loss",
                         mode="min",
                         save_best_only = True,
                         verbose=1)

  earlystop = EarlyStopping(monitor = 'val_loss', 
                            min_delta = 0, 
                            patience = 3,
                            verbose = 1,
                            restore_best_weights = True)
     # we put our call backs into a callback list
  callbacks = [earlystop, checkpoint]

      # We use a very small learning rate 
  model.compile(loss = 'categorical_crossentropy',
            optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

     # Enter the number of training and validation samples here
  nb_train_samples = 1077
  nb_validation_samples = 223

      # We only train 5 EPOCHS 
  epochs = 1
  batch_size = 16

  history = model.fit_generator(
      train_generator,
      steps_per_epoch = nb_train_samples // batch_size,
      epochs = epochs,
      callbacks = callbacks,
      validation_data = validation_generator,
      validation_steps = nb_validation_samples // batch_size)

  return history



if __name__ == "__main__": 

  # MobileNet was designed to work on 224 x 224 pixel input images sizes
  img_rows, img_cols = 224, 224 

  # Re-loads the MobileNet model without the top or FC layers
  MobileNetV2 = MobileNetV2(weights = 'imagenet', 
                   include_top = False, 
                   input_shape = (img_rows, img_cols, 3))


  # Layers are set to trainable as True by default
  for layer in MobileNetV2.layers:
      layer.trainable = False

  num_classes = int(os.environ['OUTPUT'])

  FC_Head = buidingModel(MobileNetV2, num_classes, int(os.environ['START']))

  model = Model(inputs = MobileNetV2.input, outputs = FC_Head)
      
  print(model.summary())

  train_generator = trainDataGenerator()

  validation_generator = validDataGenerator()

  history = trainModel(train_generator, validation_generator)

  print(history.history.val_accuracy[0])

  accuracy_validity = int(round(history.history.val_accuracy[0],4) * 10000)
  
  os.system("python3 counter.py {}".format(accuracy_validity))