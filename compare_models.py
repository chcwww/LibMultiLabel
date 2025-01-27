import sys
import warnings
warnings.filterwarnings("ignore")
if len(sys.argv) != 3:
    raise RuntimeError(f'type "python {sys.argv[0]} data, linear_technique."')

import libmultilabel.linear as linear

models = [linear.load_pipeline(path)[1] for path in sys.argv[1:]]

if models[0].weights.shape != models[1].weights.shape:
    raise NotImplementedError("They aren't trained with the same dataset..")
if models[0].name=='tree' or models[1].name=='tree':
    raise NotImplementedError("We can only compare two flat model, not tree model..")

print(f'Weights Diff: {(models[0].weights-models[1].weights).sum()}')
