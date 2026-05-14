from collections import defaultdict, namedtuple

class SaveOutput:
    def __init__(self):
        self.outputs = defaultdict()

    def __call__(self, name):
        def hook(module, module_in, module_out):
            self.outputs[name] = module_out.detach()
            
        return hook

    def clear(self):
        for k in list(self.outputs.keys()):
            del self.outputs[k]

_transform = namedtuple('Transform', ['function', 'components_to_transform', 'keep_inputs'])
def define_transform(function, components_to_transform, keep_inputs):
    return _transform(function, components_to_transform, keep_inputs)

_extractor_config = namedtuple(
    'ExtractorConfig', 
    ['component_descriptors', 'component_transpose_dims', 'component_transforms']
    )
def define_extractor_config(component_descriptors, component_transpose_dims, component_transforms):
    return _extractor_config(component_descriptors, component_transpose_dims, component_transforms)