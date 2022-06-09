import tensorflow as tf
from tensorflow.keras.applications.resnet50 import *
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


from tensorflow.keras.models import load_model

################## Freeze model ###################

recent_model = '/home/systartup/data/' + '322-3.6308.hdf5'
model = load_model(recent_model)
# model = ResNet50(weights='imagenet')
# print(model.summmary())
# model.save('/home/systartup/data/infoboss_frozen_model')
# print(model)
print(model.inputs)
print(model.inputs[0].shape)
print(model.inputs[1].shape)


frozen_out_path = '/home/systartup/data/'
# name of the .pb file
frozen_graph_filename = "infoboss_frozen_model"
# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
    tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype)))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)


print("node name")
for node in frozen_func.node:
    print(node)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)

