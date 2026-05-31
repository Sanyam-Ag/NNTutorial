import random
from engine import Value

class Module:   # to replicate pytorch structure

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin = True):
        self.w = [Value(random.uniform(-1, 1)) for i in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        # w*x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh() if self.nonlin else act
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):

    def __init__(self, nin, nout, nonlin):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if (len(outs) == 1) else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def train(self, train_x, train_y, lr = 0.05, niters = 20):

        if(train_x is None or train_y is None): raise ValueError("dataset values missing")
        
        print("SNo \t Prediction Loss")
        for i in range(niters):
            #forward pass
            ypred = [self(x) for x in train_x]
            loss = sum((ygt - yout)**2 for ygt, yout in zip(train_y, ypred))

            #backward pass
            for p in self.parameters():    # if not used this would accumulate the grdaients from previous step
                p.grad = 0.0            # and we will have a gradient descent step size that will explode with each trainign step
            loss.backward()

            #update
            for p in self.parameters():
                p.data += -lr * p.grad

            print(i, '\t', loss.data)

        return ypred
