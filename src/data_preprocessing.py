train_direc = "/content/drive/My Drive/Datasets/dryFruitDataset/train"
test_direc = "/content/drive/My Drive/Datasets/dryFruitDataset/test"

train_datagen = ImageDataGenerator(validation_split=0.2)
train_batches = train_datagen.flow_from_directory(train_direc,
                                                  batch_size=32,
                                                  target_size=(244,244),
                                                  class_mode='categorical')
test_batches = ImageDataGenerator().flow_from_directory(test_direc , batch_size=7, target_size=(244,244))