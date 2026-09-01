#imports
import torch
import torch.nn.functional as F

alpha = 20
iters = 100

data = open("../data.txt", 'r').read().splitlines()
data_modified_SE = []   #modified data with start and end characters
for w in data:
    data_modified_SE.append(['.'] + list(w) + ['.'])

#indexing chars
chridx = {chr(97+i):i+1 for i in range(26)}
chridx['.'] = 0
itos = {i:s for s, i in chridx.items()}



# creating training dataset of bigrams (x,y)
xs, ys = [], []

for w in data_modified_SE:
    for ch1, ch2 in zip(w, w[1:]):
        r = chridx[ch1]
        c = chridx[ch2]

        xs.append(r)
        ys.append(c)

# input

xs = torch.tensor(xs)
ys = torch.tensor(ys)

"""One Hot Encoding the characters to feed as nn inputs"""
x_encoded = F.one_hot(xs, num_classes=27).float()       # changed dtype for nn input dtype favorability


"""Neural Network like setting"""
W = torch.randn((27, 27), requires_grad=True)        # 27 characters for first charcter followed by 27 options as second character

for _i in range(iters):
    # forward pass
    logits = (x_encoded @ W)       # log counts ; (ninputexamples, 27) * (27, 27) -> (ninputexamples, 27)
    count = logits.exp()           # kind of like the count thing in the other bigram model
    probs = count/count.sum(1, keepdim=True)     # softmax kind of thing
    loss = - probs[torch.arange(len(xs)), ys].log().mean()
    print("Loss, iter", _i, ":", loss.item())

    # backward pass
    W.grad = None
    loss.backward()
    W.data -= alpha * W.grad

loss = - probs[torch.arange(len(xs)), ys].log().mean()
print(loss.item())

"""Generating samples"""

for i in range(5):

    out = ""
    ix = 0
    while True:
        xsamplencoded = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xsamplencoded @ W
        count = logits.exp()
        probsample = count/count.sum(1, keepdim=True)
        ix = torch.multinomial(probsample, 1, replacement=False).item()

        if ix == 0: break
        out += itos[ix]

    print(out)
