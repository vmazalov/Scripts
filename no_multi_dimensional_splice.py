import cntk
import sys
import numpy as np
from cntk import load_model, combine

model_file = r'model.in'
new_model_file = r'model.in.fixed'
model = cntk.Function.load(model_file)

def clone_parameter_removing_leading_unit_axes(param):
	print(param)
	new_shape = param.shape
	new_value = param.value
	while ((len(new_shape) > 1) and (new_shape[0] == 1)):
		print(new_value)
		new_shape = new_shape[1:]
		new_value = new_value[0]
	
	new_param = cntk.parameter(init = new_value)
	return new_param

name_filters = [r'.b', r'.Wco', r'.Wcf', r'.Wci', r'.Wmr']
def matches_filters(name):
	for filter in name_filters:
		if filter in name:
			return True
			
	return False
	
all_model_params = model.parameters
filtered_params = [param for param in all_model_params if matches_filters(param.name)]
param_replacements = dict()
for filtered_param in filtered_params:
	param_replacements[filtered_param] = clone_parameter_removing_leading_unit_axes(filtered_param)
	
new_model = model.clone('clone', param_replacements)
new_model.save(new_model_file)
input = []
for i in range(80):
	input.append(np.float32(0.0))

node_in_graph = new_model.find_by_name("outputNode")
output_nodes  = combine([node_in_graph.owner])
output = output_nodes.eval(np.array(input))
print("Ouput................................")
for a in output:
	print(a)
