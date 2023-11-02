
from utils.layers import *
from models.resnet4counting import *
from models.resnet4action import *

def Resnet_block(x, kernel_size, strides=(1, 1), out_size=None,
        convtype='depthwise', shortcut_act=True,
        features_div=2, name=None):
    """(Separable) Residual Unit implementation.
    """
    assert convtype in ['depthwise', 'normal'], \
            'Invalid convtype ({}).'.format(convtype)

    num_filters = K.int_shape(x)[-1]
    if out_size is None:
        out_size = num_filters

    skip_conv = (num_filters != out_size) or (strides != (1, 1))

    if skip_conv:
        x = BatchNormalization(name=appstr(name, '_bn1'))(x)

    shortcut = x
    if skip_conv:
        if shortcut_act:
            shortcut = relu(shortcut, name=appstr(name, '_shortcut_act'))
        shortcut = Conv2D(out_size, (1, 1), strides=strides, padding='same',
                name=appstr(name, '_shortcut_conv'))(shortcut)

    if not skip_conv:
        x = BatchNormalization(name=appstr(name, '_bn1'))(x)
    x = relu(x, name=appstr(name, '_act1'))

    if convtype == 'depthwise':
        x = SeparableConv2D(out_size, kernel_size, strides=strides, padding='same',
                name=appstr(name, '_conv1'))(x)
    else:
        x = Conv2D(int(out_size / features_div), (1, 1), padding='same',
                name=appstr(name, '_conv1'))(x)
        middle_bn_name = appstr(name, '_bn2')
        x = BatchNormalization(name=middle_bn_name)(x)
        x = relu(x, name=appstr(name, '_act2'))
        x = Conv2D(out_size, kernel_size, strides=strides,padding='same',
                name=appstr(name, '_conv2'))(x)

    x = add([shortcut, x])

    return x

def conv_bn_act(x, kernel_size = (3,3), name = None):
    l1 = Conv2D(filters= 16, kernel_size=kernel_size, padding='same', name=appstr(name, 'l1'))(x)
    b1 = BatchNormalization(name = appstr(name, 'b1'))(l1)
    a1 = Activation('relu')(b1)
    return a1

def counting(x, name = None):
    c1 = Resnet_block(x, kernel_size=(3, 3), out_size=8, convtype='normal', name = appstr(name, 'res1'))
    m1 = MaxPooling2D()(c1)
    c2 = Conv2D(filters=4, kernel_size=(3, 3), name = appstr(name, 'c1'))(m1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    m2 = MaxPooling2D()(a2)
    c3 = Resnet_block(m2, kernel_size=(3, 3),out_size=2, name = appstr(name, 'res2'))
    m3 = MaxPooling2D()(c3)
    c4 = Resnet_block(m3, kernel_size=(3, 3),out_size=1, name = appstr(name, 'res3'))
    m4 = MaxPooling2D()(c4)
    r = Flatten()(m4)
    d1 = Dense(256, activation='relu', name = appstr(name, 'd1'))(r)
    d2 = Dense(128, activation='relu',name = appstr(name, 'd2'))(d1)
    d3 = Dense(32, activation='relu', name=appstr(name, 'd4'))(d2)
    d4 = Dense(8, activation='relu', name=appstr(name, 'd6'))(d3)
    d5 = Dense(1, name = appstr(name, 'd7'))(d4)
    return d5


'''train action first and then counting'''
def multitask(replica = True, name = 'multi'):
    inp = Input(shape = (224, 224, 16))
    l1 = conv_bn_act(inp, name = appstr(name,'_action1'))
    l2 = conv_bn_act(l1, kernel_size=(5,5), name = appstr(name,'_action2'))
    l3 = conv_bn_act(l2, name = appstr(name,'_action3'))
    l4 = conv_bn_act(l3,kernel_size=(5,5), name=appstr(name, '_action4'))

    resnet_model = resnet34(name=appstr(name, 'action_model1'))
    action = resnet_model(l4)
    counting_model = ResNet18(name = appstr(name,'counting_model1'))
    count = counting_model(l4)
    model = Model(inputs = inp, outputs = [count,action])

    return model

def appstr(s, a):
    """Safe appending strings."""
    try:
        return s + a
    except:
        return None

if __name__ == '__main__':
    multitask()