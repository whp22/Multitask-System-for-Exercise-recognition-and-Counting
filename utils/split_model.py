from keras.models import Model

def split_model(full_model, counting_trainable):
    '''split model'''
    modelc = Model(full_model.input, full_model.output[0:1])
    modela = Model(full_model.input, full_model.output[1:])

    '''trainable'''
    for i in range(len(full_model.layers)):
        name = full_model.layers[i].name
        if counting_trainable:
            if 'counting' in name:
                full_model.layers[i].trainable = True
            else:
                full_model.layers[i].trainable = False
        else:
            if 'counting' in name:
                full_model.layers[i].trainable = False
            else:
                full_model.layers[i].trainable = True


    return [modelc, modela]