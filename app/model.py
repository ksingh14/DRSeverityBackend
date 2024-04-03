from keras import applications, layers
from keras.models import Sequential
from keras.optimizers import Adam

efficient_net = applications.efficientnet.EfficientNetB5(
    weights='imagenet',
    include_top=False,
    input_shape=(512,512,3)
)

def build_model():
    model = Sequential()
    model.add(efficient_net)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(2048, activation='relu')) #added
    # model.add(layers.Dropout(0.5)) #added
    model.add(layers.Dense(5, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model