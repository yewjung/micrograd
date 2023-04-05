import random
from typing import List

from engine import Value

class Neuron:

    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x: List[Value]) -> Value:
        output = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        output = output.tanh()
        return Value(output)
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, xs: List[Value]) -> List[Value]:
        outs = [neuron(xs) for neuron in self.neurons]
        return outs if len(outs) > 1 else outs[0]
    
    def parameters(self):
        return [w for neuron in self.neurons for w in neuron.parameters()]

class MLP:

    def __init__(self, nin: int, nouts: List[int]) -> None:
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(nouts))]
        
    
    def __call__(self, xs: List[int]) -> List[Value]:
        for layer in self.layers:
            xs = layer(xs)
        
        return xs
    
    def parameters(self):
        return [w for layer in self.layers for w in layer.parameters()]
    
def training():

    # forward

    # flush grads

    # backward

    # update
        
    pass

if __name__ == "__main__":
    training()