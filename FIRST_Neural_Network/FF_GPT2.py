import torch
import torch.nn as nn
from torch.nn import functional as F
from math import inf as inf

batch_size = 8 # Number of batches
context_length = 124 # Window of influence for prediction of next token. 
training_cycles = 10000 # Iterations of training
loss_iterations = 500 # Number of iterations to determine calculated average cross_entropy loss
learning_rate = 4e-3

n_embd = 144
n_heads = 6

no_layers = 6
dropout = 0.1 # Dropout rate: with probability 0.1, each unit is set to 0 during training.
               # This helps prevent overfitting by randomly "dropping" parts of the network

## (B, T, C) MEANING (No batches, length of sequence / context_length, embedding size, for each char of each timestep, of each batch)

training_path = "input.txt"
# Vocabulary will be the set of uniqe characters
training_text = "".join([r for r in open(training_path, "r")])

vocab = list(set([ord(char) for line in training_text for char in line]))
vocab_size = len(vocab)

# Define the characters we use for input/output. Define each of them as a index
mp = {vocab[i]:i for i in range(vocab_size)}
mp_inv = {i:vocab[i] for i in range(vocab_size)}
encode = lambda s: list(mp[ord(i)] for i in s)
decode = lambda l: "".join([chr(mp_inv[i]) for i in l])


device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 6969
torch.manual_seed(seed)

# turns the input file into integers in range vocab_size
encoded_training_data = torch.tensor(encode(training_text), dtype=torch.long)

# We preserve some 10% of the text for validation of outputs
split = int(0.9*len(encoded_training_data))
training_data = encoded_training_data[:split]
validation_data = encoded_training_data[split:]

# Loads data for processing, and returns the data it loaded
def sample_training_batch(training):
    data = training_data if training else validation_data
    # Pick some random indexes in the input file for every batch
    indexes = torch.randint(len(data)-context_length, (batch_size,))
    # cut out the text length context_length for each index, and the predicted tokens for each position
    contexts, targets = torch.stack([data[start_idx:start_idx+context_length] for start_idx in indexes]), torch.stack([data[start_idx+1:start_idx+context_length+1] for start_idx in indexes])
    # Load the tensors to gpu device
    contexts, targets = contexts.to(device), targets.to(device)
    return contexts, targets

@torch.no_grad()
def estimate_loss():
    estimated_losses = [0, 0]
    
    chatbot.eval()
    for mode in (1, 0): # get both training loss and validation loss
        batch_losses = 0
        for _ in range(loss_iterations):
            contexts, targets = sample_training_batch(training=mode)
            logits, loss = chatbot(contexts, targets)
            batch_losses += loss.item()
        # get the mean
        estimated_losses[mode] = batch_losses / loss_iterations
    chatbot.train()

    return estimated_losses

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # implement k, q, v for each node in the neural network
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        # we get keys for and queries for the cur attention-representation
        k, q = self.key(x), self.query(x)
        wei = k @ q.transpose(1, 2) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)

        # we use lower triangular ones, to later multiply and get the valid mean historical relations
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -inf)
        wei = F.softmax(wei, dim=2)
        wei = self.dropout(wei)
        # we multiply to get the historical mean, keys, and queries meshed together
        return wei @ self.value(x) # (B, T, T) @ (B, T, 16) -> (B, T, 16)

class MultiHeads(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create a indexable list for each head, storing its own chunk of embedding for each character
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # we concatenate the result from each head and project it as a (C, C) tensor
        return self.dropout(self.proj(torch.cat([H(x) for H in self.heads], dim = 2)))
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd * 4), nn.ReLU(), nn.Linear(n_embd * 4, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # split up the embedding size to seperate chunks we call them heads
        self.sa_head = MultiHeads(n_heads, n_embd//n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.relation_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        # Defines each processing block that sequentially introduces its own data to the output.
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(no_layers)], nn.Dropout(dropout)) # Block(n_embd, n_heads=4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, prev_contexts, targets=None):
        B, T = prev_contexts.shape
        tok_embeddings = self.relation_table(prev_contexts)  # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) with the range(0, T) for every column
        # we just use a simple way of infuencing by position. by addition of the indexes
        x = tok_embeddings + pos_embd
        # perform the self attention using multiple blocks/layers and normalizing accross all the blocks
        x = self.blocks(x)

        # get the final relationships between characters
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        # only executes when evaluating the model in @estimate_loss()
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    # Generates sequence of predictions
    def generate(self, generated, generation_length):

        for _ in range(generation_length):

            predictions, loss = self(generated[:, -context_length:])
            predictions = predictions[:, -1, :]
            # get the distribution of probabilities for the newly generated prediction
            probability_distribution = F.softmax(predictions, dim=1)

            # non-deterministically pick one, based on the frequencey of each option
            most_probable_next_char = torch.multinomial(probability_distribution, num_samples=1)

            # Add the most probable character to accumulated text
            generated = torch.cat((generated, most_probable_next_char), dim=1)
        return generated
    

model = GPT2()
chatbot = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
### Training loop for local training while the script is running
display_loss = 1000
for i in range(training_cycles):
    # check estimated loss every display_loss iterations
    if i % display_loss == 0:
        t, v = estimate_loss()
        print(f"step: {i}, train loss: {t:.4f}, val loss: {v:.4f}")
    # get som random text
    contexts, targets = sample_training_batch(training=True)
    # predict based on current knowledge
    predictions, loss = model(contexts, targets)
    # evaluate and optimize
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Initialize some text generation for each batch -> [[0] for _ in range(batches)]
context = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
## Generate some text of characters
# We just return the first batch (batch[0])... we could do some evaluation and decide to choose specific ones if we want to
print("\n",decode(chatbot.generate(context, generation_length = 3000)[0].tolist()))


        