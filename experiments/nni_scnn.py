import os
import nengo
import nengo_dl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras import Input
from keras import layers, models
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras.callbacks import TensorBoard, Callback
import argparse
import logging
import json
import nni
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, freqz
from nni.tuner import Tuner
from nni.experiment import Experiment
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.tools.nnictl import updater, nnictl_utils


os.environ["CUDA_VISIBLE_DEVICES"] = '1'  
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'  

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)


##### SOME DEFAULT SETTING ###########################################
### These defaults will be replaced
### when calling the function file_settings

network_type = 'cnn'
device = 'watch'
subset = 2
time_window = 2

window_size = int(20*time_window) # 20 Hz sampling times the temporal length of the window

datafile = 'data_'+device+'_subset'+str(subset)+'_'+str(window_size)

searchspace_path = '../searchspaces/nni_SearchSpace_s'+network_type+'.json'


##### GET NETWORK STRUCTURE PARAMETERS from previous NNI optimization of non-spiking CNN #####
optim_nni_experiment = 'baWMlt3d' 
optim_nni_trial = 'Y7WcM' 
optim_filename = 'parameter.cfg'
optim_nni_ref = 'wisdm2_'+network_type+'_'+device+'_subset'+str(subset)+'_'+str(window_size)+'_'+optim_nni_experiment+'/trials/'+str(optim_nni_trial)
optim_nni_dir = '../output/cnn_lstm'
optim_filepath = os.path.join(optim_nni_dir,optim_nni_ref,optim_filename)

with open(optim_filepath, 'r') as f:
    data = f.read()

param_data = json.loads(data)
network_parameters = param_data['parameters']


non_spiking_params = {
                      'Conv1D_filters_1': network_parameters['nni_network/Conv1D_filters_1/randint'],
                      'Conv1D_filters_2': network_parameters['nni_network/Conv1D_filters_2/randint'],
                      'Conv1D_kernel_size_1': network_parameters['nni_network/Conv1D_kernel_size_1/randint'],
                      'Conv1D_kernel_size_2': network_parameters['nni_network/Conv1D_kernel_size_2/randint'],
                      'CNN_Dense_1': network_parameters['nni_network/CNN_Dense_1/randint'],
                      'lr': network_parameters['nni_network/lr/quniform'],
                      'batch_size': network_parameters['nni_network/batch_size/randint']
                     }
##################################################

eps = 100


def file_settings(args):

    window_size = int(20*args.time_window) # 20 Hz sampling times the temporal length of the window
    datafile = 'data_'+args.device+'_subset'+str(args.subset)+'_'+str(window_size)

    return datafile


def load_wisdm2_data(args):
    filepath = os.path.join('../data/',args.datafile+'.npz')
    a = np.load(filepath)
    return (a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3'], a['arr_4'], a['arr_5'])


def report_result(result, result_type, network_type):

    if result_type == 'test':
        report_file = out_dir + 'nni_s' + network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs'
    else:
        report_file = out_dir + 'nni_s' + network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs_' + nni.get_trial_id()
    
    with open(report_file, 'a') as f:
        f.write(str(result))
        f.write('\n')
    
    return report_file


def create_nengo_model(network_type, timesteps, input_dim, n_classes, non_spiking, params):
    
    model_nonspiking = Sequential()
    
    model_nonspiking.add(Conv1D(filters=non_spiking['Conv1D_filters_1'], kernel_size=non_spiking['Conv1D_kernel_size_1'], activation=tf.nn.relu, kernel_initializer='he_uniform', input_shape=(timesteps,input_dim), name='Conv1D_1'))
    model_nonspiking.add(Conv1D(filters=non_spiking['Conv1D_filters_2'], kernel_size=non_spiking['Conv1D_kernel_size_2'], activation=tf.nn.relu, kernel_initializer='he_uniform', name='Conv1D_2'))
    model_nonspiking.add(MaxPooling1D(pool_size=2, name='MaxPooling1D'))
    model_nonspiking.add(Flatten())
    model_nonspiking.add(Dense(non_spiking['CNN_Dense_1'], activation=tf.nn.relu, name='Dense_1'))
    model_nonspiking.add(Dense(n_classes, activation='softmax', name='Dense_2'))
    
    ### LOAD PRE-TRAINED WEIGHTS ###
    model_nonspiking.load_weights('../output/cnn_lstm/wisdm2_'+network_type+'_'+device+'_subset'+str(subset)+'_'+str(window_size)+'_'+optim_nni_experiment+'/best_test/'+'best_test_'+optim_nni_experiment)
    
    ### sequential to functional model
    input_layer = layers.Input(batch_shape=model_nonspiking.layers[0].input_shape, name='Input')
    prev_layer = input_layer
    for ii in model_nonspiking.layers:
        prev_layer = ii(prev_layer)

    model = Model([input_layer], [prev_layer])

    model.summary()

    ### REMEMBER: here the model is only converted into Nengo
    converter = nengo_dl.Converter(model,
                                   max_to_avg_pool=True,
                                  )

    return model, converter


class SendMetrics(Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        nni.report_intermediate_result(logs['val_probe_accuracy']*100)


def run_nengo(args, non_spiking_params, params):

    (x_train, x_val, x_test, y_train_oh, y_val_oh, y_test_oh) = load_wisdm2_data(args)
    y_train = np.argmax(y_train_oh, axis=-1)
    y_val = np.argmax(y_val_oh, axis=-1)
    y_test = np.argmax(y_test_oh, axis=-1)

    timesteps = len(x_train[0])
    input_dim = len(x_train[0][0])
    n_classes = len(y_train_oh[0])

    x_train = x_train.reshape((x_train.shape[0], 1, -1))
    y_train = y_train[:,None,None] 
    x_val = x_val.reshape((x_val.shape[0], 1, -1))
    y_val = y_val[:,None,None] 
    x_test = x_test.reshape((x_test.shape[0], 1, -1))
    y_test = y_test[:,None,None] 

    keras_model, converter = create_nengo_model(args.network_type, timesteps, input_dim, n_classes, non_spiking_params, params)

    keras_layers = list(keras_model.layers[ii].name for ii in range(len(keras_model.layers)))
    
    # add probes to the convolutional layers, to apply the firing rate regularization
    with converter.net:
        output_p = converter.outputs[keras_model.output]
        conv0_p = nengo.Probe(converter.layers[keras_model.layers[1].get_output_at(-1)])
        conv1_p = nengo.Probe(converter.layers[keras_model.layers[2].get_output_at(-1)])
    
    n_steps = params['nni_keras2snn_network/n_steps/randint']

    tiled_x_test = np.tile(x_test, (1, n_steps, 1))

    with nengo_dl.Simulator(converter.net, minibatch_size=params['nni_keras2snn_network/batch_size/randint']) as sim:
        
        nengo_model_summary = sim.keras_model
        nengo_params = sum(np.prod(s.shape) for s in nengo_model_summary.weights)
        nengo_trainable_params = sum(np.prod(w.shape) for w in nengo_model_summary.trainable_weights)
        LOG.debug(print('Total params:','{:,d}'.format(nengo_params)))
        LOG.debug(print('Trainable params:','{:,d}'.format(nengo_trainable_params)))
        LOG.debug(print('Non-trainable params:','{:,d}'.format(nengo_params-nengo_trainable_params)))
        
        # add regularization loss functions to the convolutional layers
        sim.compile(
                    optimizer=tf.optimizers.Adam(params['nni_keras2snn_network/lr/quniform']),
                    loss={
                          output_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                          conv0_p: tf.losses.mse,
                          conv1_p: tf.losses.mse,
                         },
                    loss_weights={
                                  output_p: 1, 
                                  conv0_p: params['nni_keras2snn_network/reg_conv0/quniform'], 
                                  conv1_p: params['nni_keras2snn_network/reg_conv1/quniform']
                                 },
                    metrics=["accuracy"],
                   )
        
        class CheckPoint(Callback):
            '''
            Keras callback to check training results epoch by epoch
            '''
            def on_epoch_end(self, epoch, logs={}):
                '''
                Run on end of each epoch
                '''
                report_file = report_result(logs['val_probe_accuracy']*100,'validation',args.network_type)

                with open(report_file, 'r') as f:
                    if logs['val_probe_accuracy']*100 >= np.max(np.asarray([(line.strip()) for line in f], dtype=np.float64)):
                        sim.save_params(out_dir+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id())

        history = sim.fit(
                          {converter.inputs[keras_model.input]: x_train},
                          {
                           output_p: y_train,
                           conv0_p: np.ones((y_train.shape[0], 1, conv0_p.size_in)) * params['nni_keras2snn_network/target_rate_0/randint'],
                           conv1_p: np.ones((y_train.shape[0], 1, conv1_p.size_in)) * params['nni_keras2snn_network/target_rate_1/randint'],
                          },
                          validation_data = (x_val, y_val),
                          epochs=args.epochs,
                          callbacks=[SendMetrics(), CheckPoint(), TensorBoard(log_dir=TENSORBOARD_DIR)],
                         )

    sim.close()

    ### Conversion to spiking 
    trained_converter = nengo_dl.Converter(keras_model,
                                           max_to_avg_pool=True,
                                           swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()},
                                           scale_firing_rates=params['nni_keras2snn_network/scale_firing_rates/randint'],
                                           synapse=params['nni_keras2snn_network/synapse/quniform'],
                                          )
    
    with trained_converter.net:
        output_p = trained_converter.outputs[keras_model.output]
        conv0_p = nengo.Probe(trained_converter.layers[keras_model.layers[1].get_output_at(-1)])
        conv1_p = nengo.Probe(trained_converter.layers[keras_model.layers[2].get_output_at(-1)])
    
    with nengo_dl.Simulator(trained_converter.net, minibatch_size=params['nni_keras2snn_network/batch_size/randint']) as sim:
        
        sim.compile(
                    optimizer=tf.optimizers.Adam(params['nni_keras2snn_network/lr/quniform']),
                    loss={
                          output_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                          conv0_p: tf.losses.mse,
                          conv1_p: tf.losses.mse,
                         },
                    loss_weights={
                                  output_p: 1, 
                                  conv0_p: params['nni_keras2snn_network/reg_conv0/quniform'], 
                                  conv1_p: params['nni_keras2snn_network/reg_conv1/quniform']
                                 },
                    metrics=["accuracy"],
                   )
        
        sim.load_params(out_dir+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id())
        
        try:
            data = sim.predict({trained_converter.inputs[keras_model.input]: tiled_x_test})
            predictions = np.argmax(data[trained_converter.outputs[keras_model.output]][:, -1], axis=-1)
            test_accuracy = (predictions[:] == y_test[:predictions.shape[0], 0, 0]).mean()
            
            report_file = report_result(test_accuracy*100, 'test', args.network_type)
            with open(report_file, 'r') as f:
                if test_accuracy*100 >= np.max(np.asarray([(line.strip()) for line in f], dtype=np.float64)):
                    sim.save_params(out_dir+'best_test_'+nni.get_experiment_id())
                    
        except Exception as e:
            LOG.exception(e)
            raise

    sim.close()

    LOG.debug(print(f"Final validation accuracy: {100 * history.history['val_probe_accuracy'][-1]:.2f}%"))
    LOG.debug(print(f"Best validation accuracy: {100 * np.max(history.history['val_probe_accuracy']):.2f}%"))
    LOG.debug(print(f"Test accuracy from training with best validation accuracy: {100 * test_accuracy:.2f}%"))
    nni.report_final_result(test_accuracy*100)

    os.remove(out_dir+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id()+'.npz')
    os.remove(out_dir+'nni_s'+args.network_type+'_'+nni.get_experiment_id()+'_validation_accs_'+nni.get_trial_id())


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--network_type", type=str, default=network_type, help="network type", required=False)
    PARSER.add_argument("--subset", type=str, default=subset, help="activities subset", required=False)
    PARSER.add_argument("--datafile", type=str, default=datafile, help="data file", required=False)
    PARSER.add_argument("--epochs", type=int, default=eps, help="Train epochs", required=False)
    PARSER.add_argument("--filename", type=str, default=searchspace_path, help="File name for search space", required=False)
    PARSER.add_argument("--id", type=str, default=nni.get_experiment_id(), help="Experiment ID", required=False)
    PARSER.add_argument("--device", type=str, default=device, help="From which device the signal is taken", required=False)
    PARSER.add_argument("--time_window", type=int, default=time_window, help="Length of the time window", required=False)
    
    ARGS, UNKNOWN = PARSER.parse_known_args()

    datafile = file_settings(ARGS)
    LOG = logging.getLogger('wisdm2_s'+ARGS.network_type+'_'+datafile[5:])
    out_dir = '../output/tmp_s' + ARGS.network_type + '_' + nni.get_experiment_id() + '_' + datafile[5:] + '/'
    os.environ['NNI_OUTPUT_DIR'] = out_dir
    TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR'] 
    
    try:

        n_tr = 200
        if (nni.get_sequence_id() > 0) & (nni.get_sequence_id()%n_tr == 0):
            updater.update_searchspace(ARGS) # it will use ARGS.filename to update the search space

        PARAMS = nni.get_next_parameter()
        LOG.debug(PARAMS)
        
        run_nengo(ARGS, non_spiking_params, PARAMS)
    
    except Exception as e:
        LOG.exception(e)
        raise