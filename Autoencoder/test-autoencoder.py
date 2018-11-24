import glob
import os, sys
import numpy
from PIL import Image
import keras
from keras.models import Sequential #-> red neuronal de capas
from keras.models import Model, load_model
from keras.layers import Reshape, UpSampling2D, Lambda, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, concatenate #-> fully conectded layer
from keras.optimizers import SGD #->  optimizador (un tipo de ellos) Scholastic gradient descent
from keras.utils import to_categorical
from keras.backend import tf as ktf
import matplotlib.pyplot as plt

#1 - entrenar
#1.1 - creo modelo
#1.2 - leo imágenes
#1.3 - fit

width = 256
height = 256
channels = 3
tam_imagenes = (width, height)

def entrenar(path, modelo, num_epochs=1):
    path_class = path
    train_imagenes = []
    for curr_image in next(os.walk(path_class))[2]:
        im = Image.open(os.path.join(path_class, curr_image))
        im = im.resize(tam_imagenes, Image.ANTIALIAS)
        imagen_array = numpy.asarray(im).astype(dtype='float32')
        if imagen_array.shape == (256, 256):
            imagen_array = numpy.expand_dims(imagen_array, axis=2)
            imagen_array = numpy.concatenate((imagen_array, imagen_array, imagen_array), axis=2)
        #NORMALIZAMOS
        normalizar = lambda x: x / 255.0
        vfunc = numpy.vectorize(normalizar)
        imagen_array_normalizada = vfunc(imagen_array)
        print(curr_image+' '+str(imagen_array_normalizada.shape))
        train_imagenes.append(imagen_array_normalizada)
    #plt.imshow(train_imagenes[2])
    #plt.show()
    #print(train_imagenes[2].shape)
    #ENTRENAMOS
    train_imagenes = numpy.array(train_imagenes)
    modelo.fit(train_imagenes, train_imagenes,
              batch_size=5,
              epochs=num_epochs,
              verbose=1)
    return modelo


def creaModeloSimple(input_shape=(256, 256, 3), encoding_dim=256):

    total_tam = input_shape[0]*input_shape[1]*input_shape[2]

    imagenEntrada = Input(shape=input_shape)
    aplana_1 = Flatten()(imagenEntrada)
    encoded = Dense(encoding_dim, activation='relu')(aplana_1)
    decoded = Dense(total_tam, activation='sigmoid')(encoded)
    decodifica_fin = Reshape(input_shape)(decoded)
    m = Model(inputs=[imagenEntrada], outputs=decodifica_fin)
    #m.summary()
    return m

def creaModelo(input_shape=(256, 256, 3)):
    imagenEntrada = Input(shape=input_shape)
    conv2d_1_3x3_a = Conv2D(6, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=input_shape)(imagenEntrada)
    conv2d_1_3x3_b = MaxPooling2D(pool_size=(2, 2))(conv2d_1_3x3_a)

    conv2d_2_3x3_a = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=input_shape)(imagenEntrada)
    conv2d_2_3x3_b = MaxPooling2D(pool_size=(2, 2))(conv2d_2_3x3_a)

    conv2d_3_3x3_a = Conv2D(6, kernel_size=(7, 7), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=input_shape)(imagenEntrada)
    conv2d_3_3x3_b = MaxPooling2D(pool_size=(2, 2))(conv2d_3_3x3_a)

    concatenate_1 = concatenate([conv2d_1_3x3_b, conv2d_2_3x3_b, conv2d_3_3x3_b], axis=3)

    concatenate_2 = Conv2D(6, kernel_size=(1, 1), strides=(1, 1), padding='same',
                     activation='relu',)(concatenate_1)
    concatenate_3 = MaxPooling2D(pool_size=(2, 2))(concatenate_2)
    aplana_1 = Flatten()(concatenate_3)
    #reduccion_1 = AveragePooling2D(pool_size=(2, 2))(aplana_1)
    reduccion_2 = Dense(64*64, name='dense_layer_2')(aplana_1)
    #reduccion_2 = Dense(96*96, name='dense_layer_2')(aplana_1)
    #ESTA ES LA PARTE DE ENCODING
    #Convierto el 96*96 en 96,96
    decodifica_1 = Reshape((64, 64, 1))(reduccion_2)
    decod_expand_1 = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(decodifica_1)
    #transformo llamando a la función de tensorflow
    decod_expand_2 = UpSampling2D(size=(4, 4))(decod_expand_1)
    #decod_expand_2 = Lambda(lambda image: ktf.image.resize_images(image, (input_shape[0], input_shape[1])))(decod_expand_1)
    resultado = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='sigmoid')(decod_expand_2)
   # decod_expand_2 = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(decod_expand_1)
    m = Model(inputs=[imagenEntrada], outputs=resultado)
    #m.summary()
    return m

def testea_imagen(path_img, modelo):
    path_imagen = path_img #'C:\\datasets\\cat-dogs\\perro.jpg'
    im = Image.open(path_imagen)
    im = im.resize(tam_imagenes, Image.ANTIALIAS)
    imagen_array = numpy.asarray(im).astype(dtype='float32')
    # NORMALIZAMOS
    normalizar = lambda x: x / 255.0
    vfunc = numpy.vectorize(normalizar)
    imagen_array_normalizada = vfunc(imagen_array)
    prediction = modelo.predict(numpy.array([imagen_array_normalizada]))
    max_value_prediction = numpy.amax(prediction[0])
    print(max_value_prediction)
    print("----")
    predicion_truncada = prediction[0]
    if max_value_prediction > 1.0:
        truncar = lambda x: min(x, 1.0)
        vfunc_truncar = numpy.vectorize(truncar)
        predicion_truncada = vfunc_truncar(prediction[0])
    #print(prediction[0])
    f, axarr = plt.subplots(2)
    axarr[0].imshow(imagen_array_normalizada)
    axarr[1].imshow(predicion_truncada)

    #plt.imshow(imagen_array_normalizada)
    #plt.imshow(prediction[0])
    plt.show()

def mas_entrenamientos(path_imagenes, path_nombre='modelo_chungo.h5', num_epochs=1):
    modelo = load_model(path_nombre)
    modelo = entrenar(path_imagenes, modelo, num_epochs=num_epochs)
    #La salvo
    modelo.save(path_nombre)

def prueba_imagenes(paths_imagenes, path_nombre='modelo_chungo.h5'):
    modelo = load_model(path_nombre)
    for i in paths_imagenes:
        testea_imagen(i, modelo)



def primer_entrenamiento(path, save_file='modelo_chungo.h5', num_epochs=1):
    modelo = creaModelo()

    modelo.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    modelo.summary()


    modelo = entrenar(path, modelo, num_epochs)
    '''
    path_imagen = 'C:\\datasets\\cat-dogs\\perro.jpg'
    im = Image.open(path_imagen)
    im = im.resize(tam_imagenes, Image.ANTIALIAS)
    imagen_array = numpy.asarray(im).astype(dtype='float32')
    # NORMALIZAMOS
    normalizar = lambda x: x / 255.0
    vfunc = numpy.vectorize(normalizar)
    imagen_array_normalizada = vfunc(imagen_array)
    prediction = modelo.predict(numpy.array([imagen_array_normalizada]))
    print(prediction)
    plt.imshow(prediction[0])
    plt.show()'''
    modelo.save(save_file)

def primer_entrenamiento_simple(path, save_file='modelo_chungo_simple.h5', num_epochs=1):
    modelo = creaModeloSimple()

    modelo.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    modelo.summary()


    modelo = entrenar(path, modelo, num_epochs)
    modelo.save(save_file)

def pruebas_1():
    print("Pruebas 1")
    path = 'C:\\datasets\\cat-dogs\\test_set\\cats'
    #path = 'C:\\datasets\\cat-dogs\\microcats'
    #primer_entrenamiento(path, save_file='modelo_chungo.h5', num_epochs=5)
    mas_entrenamientos(path, path_nombre='modelo_chungo.h5', num_epochs=2)
    paths = []
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4043.jpg')

    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4044.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4045.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4046.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4047.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4048.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4049.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4050.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4051.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4052.jpg')
    paths.append('C:\\datasets\\cat-dogs\\test_set\\cats\\cat.4053.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4026.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4026.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4001.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4003.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4005.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4007.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4028.jpg')
    paths.append('C:\\datasets\\cat-dogs\\microcats\\cat.4025.jpg')
    paths.append('C:\\datasets\\cat-dogs\\perro.jpg')

    prueba_imagenes(paths, path_nombre='modelo_chungo.h5')

def pruebas_2():
    print("Pruebas 2")
    #------------------------snoopy---------------
    path = 'C:\\datasets\\snoopy'
    #primer_entrenamiento(path, save_file='modelo_snoopy.h5', num_epochs=20)
    #mas_entrenamientos(path, path_nombre='modelo_snoopy.h5', num_epochs=25)

    paths = []
    paths.append('C:\\datasets\\snoopy\\image_0003.jpg')
    paths.append('C:\\datasets\\snoopy\\image_0012.jpg')
    paths.append('C:\\datasets\\snoopy\\image_0016.jpg')
    paths.append('C:\\datasets\\cat-dogs\\perro.jpg')
    paths.append('C:\\datasets\\cat-dogs\\saltamontes.jpg')
    paths.append('C:\\datasets\\cartoon1.jpg')
    paths.append('C:\\datasets\\cartoon2.jpg')
    paths.append('C:\\datasets\\cartoon3.jpg')
    paths.append('C:\\datasets\\cartoon4.jpg')
    prueba_imagenes(paths, path_nombre='modelo_snoopy.h5')

def pruebas_3():
    print("Pruebas 3")
    #------------------------snoopy---------------
    path = 'C:\\datasets\\snoopy'
    #primer_entrenamiento(path, save_file='modelo_snoopy_simple.h5', num_epochs=200)
    #mas_entrenamientos(path, path_nombre='modelo_snoopy_simple.h5', num_epochs=25)

    paths = []
    paths.append('C:\\datasets\\snoopy\\image_0003.jpg')
    paths.append('C:\\datasets\\snoopy\\image_0012.jpg')
    paths.append('C:\\datasets\\snoopy\\image_0016.jpg')
    paths.append('C:\\datasets\\cat-dogs\\perro.jpg')
    paths.append('C:\\datasets\\cat-dogs\\saltamontes.jpg')
    paths.append('C:\\datasets\\cartoon1.jpg')
    paths.append('C:\\datasets\\cartoon2.jpg')
    paths.append('C:\\datasets\\cartoon3.jpg')
    paths.append('C:\\datasets\\cartoon4.jpg')
    prueba_imagenes(paths, path_nombre='modelo_snoopy_simple.h5')

if __name__ == "__main__":
    primer_entrenamiento_simple('C:\\datasets\\snoopy', save_file='modelo_snoopy_simple.h5', num_epochs=50)
    #pruebas_1()
    #pruebas_2() #Snoopy
    pruebas_3() #Snoopy dense