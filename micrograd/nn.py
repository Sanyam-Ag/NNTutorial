import random
import numpy as np
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
    def __init__(self, nin, nouts, estimatemodel = None):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        self.estimatemodel = 'probabilistic' if estimatemodel is None or estimatemodel != 'boundary' else 'boundary' # to decide loss function

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def train(self, train_x, train_y, lr = 0.05, niters = 20, batch_size = None):

        if(train_x is None or train_y is None): raise ValueError("dataset values missing")
        
        # boundary loss function
        def loss_boundary():
            if batch_size is None:
                Xb, yb = train_x, train_y
            else:
                ri = np.random.permutation(train_x.shape[0])[:batch_size]
                Xb, yb = train_x[ri], train_y[ri]
            inputs = [list(map(Value, xrow)) for xrow in Xb]
            
            # forward the model to get scores
            scores = list(map(self, inputs))
            
            # svm "max-margin" loss
            losses = [(1 + -yi*scorei).tanh() for yi, scorei in zip(yb, scores)]
            data_loss = sum(losses) * (1.0 / len(losses))
            # L2 regularization
            alpha = 1e-4
            reg_loss = alpha * sum((p*p for p in self.parameters()))
            total_loss = data_loss + reg_loss
            
            # also get accuracy
            accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
            return total_loss, sum(accuracy) / len(accuracy)

        for i in range(niters):
            #forward pass
            if self.estimatemodel == 'boundary' : total_loss, acc = loss_boundary()
            else:
                ypred = [self(x) for x in train_x]
                total_loss = sum((ygt - yout)**2 for ygt, yout in zip(train_y, ypred))

            #backward pass
            for p in self.parameters():    # if not used this would accumulate the grdaients from previous step
                p.grad = 0.0            # and we will have a gradient descent step size that will explode with each trainign step
            # or model.zero_grad()
            total_loss.backward()

            #update
            lr = lr * 0.95 if (i%10 == 0) else lr  #reduce lr as we move towards center of minima
            for p in self.parameters():
                p.data -= lr * p.grad

            if self.estimatemodel == 'boundary' : print(f"step {i} loss {total_loss.data}, accuracy {acc*100}%")
            else:   print(f"step {i} loss {total_loss.data}")

        return [self(x) for x in train_x]
