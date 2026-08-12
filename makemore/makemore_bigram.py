#imports
import torch
import matplotlib.pyplot as plt


data = open("data.txt", 'r').read().splitlines()
data_modified_SE = []   #modified data with start and end characters
for w in data:
    data_modified_SE.append(['.'] + list(w) + ['.'])


#indexing chars
chridx = {chr(97+i):i+1 for i in range(26)}
chridx['.'] = 0
itos = {i:s for s, i in chridx.items()}



#bigram model
"""
Dictionary -> Pytorch Tensors

bigram_dict = {}
for w in data_modified_SE:
    for ch1, ch2 in zip(w, w[1:]):
        bigram = (ch1, ch2)

        bigram_dict[bigram] = bigram_dict.get(bigram, 0) + 1
        #print(ch1, ch2)

"""

#rows = first character,    cols = second character,    data_values = count
bigram_tensor = torch.zeros((27, 27), dtype=torch.int32)        
for w in data_modified_SE:
    for ch1, ch2 in zip(w, w[1:]):
        r = chridx[ch1]
        c = chridx[ch2]
        bigram_tensor[r][c] += 1

# plt.imshow(bigram_tensor)
# plt.show()



# Probability Distribution for characters
""" Single item generation

p = bigram_tensor[0].float()
p = p/p.sum()
g = torch.Generator().manual_seed(2147483647)     #manual seed for deterministic result
item = torch.multinomial(p, num_samples = 1, replacement=True, generator=g).item()
"""


g = torch.Generator().manual_seed(2147483647)

P = (bigram_tensor+1).float()       # +1 to bigram_tensor for model smoothing to prevent infinity in log likelihood for a sample data sample
P /= P.sum(1, keepdim=True)    #broadcasting
""" 
for i in range(27):
    p = bigram_tensor[i].float()
    p = p/p.sum()
    P.append(p)
"""

for i in range(5):
    out = ""
    ix = 0
    while True:
        ix = torch.multinomial(P[ix], num_samples = 1, replacement=True, generator=g).item()
        """
            1. probability distribution to select the next character based on probability distribution for the previous 
            character combinations calculated using the training data. 
            2. sampling from the disribution removes determinism(though manual seed would yield same result) but reduces 
            the  chances of incorrect next token(though there is no wrong character in this use case).
            3. there would be a lot of stupid tokens generated as it as a very short context of only one previous 
            character and the newly generated character.
        """
        if ix==0:   break
        else:   out += itos[ix]

    print(out)


# Quality of model : negative log likelihood / MLE
log_likelihood = 0.0   # log likelihood
cnt = 0

for w in data_modified_SE:
    for ch1, ch2 in zip(w, w[1:]):
        r = chridx[ch1]
        c = chridx[ch2]

        logprob = torch.log(P[r, c])  # log probability learned by model

        log_likelihood += logprob
        cnt += 1

nlog_likelihood = -log_likelihood   # negative log likelihood
print(f'{nlog_likelihood/cnt}')     # the lower the value of negative log likelihood the better the model



""""
    OPTIMIZATION:

        1. Maximize likelihood of data wrt to model parameters : here in this model the probabilities of the bigrams
        2. equivalent to maximizing log likelihood as log is a monotonic function
"""