import glob
import random
import numpy as np
import os.path as op
import cPickle as pickle
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

# Softmax function, optimized such that larger inputs are still feasible
# softmax(x + c) = softmax(x)
def softmax(x):
	orig_shape = x.shape
	if len(x.shape) > 1:
		x = x - np.max(x, axis = 1, keepdims = True)
		exp_x = np.exp(x)
		x = exp_x / np.sum(exp_x, axis = 1, keepdims = True)
	else:
		x = x - np.max(x, axis = 0)
		exp_x = np.exp(x)
		x = exp_x / np.sum(exp_x, axis = 0)
	assert x.shape == orig_shape
	return x

# Implementation for the sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_grad(sigmoid):
	return sigmoid * (1 - sigmoid)

# Gradient checker for a function f
# f is a function that takes a single argument and outputs the cost and its gradients
# x is the point to check the gradient at
def gradient_checker(f, x):

    rndstate = random.getstate()
    random.setstate(rndstate)
    cost, grad = f(x) # Evaluate function value at original point
    epsilon = 1e-4  # Tiny shift to the input to compute approximated gradient with formula

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index

        # Calculate J(theta_minus)
        x_minus = np.copy(x)
        x_minus[i] = x[i] - epsilon
        random.setstate(rndstate)
        f_minus = f(x_minus)[0]

        # Calculate J(theta_plus)
        x_plus = np.copy(x)
        x_plus[i] = x[i] + epsilon
        random.setstate(rndstate)
        f_plus = f(x_plus)[0]

        numgrad = (f_plus - f_minus) / (2 * epsilon) 

        # Compare gradients
        reldiff = abs(numgrad - grad[i]) / max(1, abs(numgrad), abs(grad[i]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(i)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[i], numgrad)
            return

        it.iternext() # Step to next dimension
    print "Gradient check passed!"

# Normalize each row of a matrix to have unit length
def normalizeRows(a):
	a = a / np.sqrt(np.sum(a ** 2, axis = 1, keepdims = True))
	return a

# Softmax cost and gradients for word2vec models
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Arguments:
    predicted -- numpy ndarray, predicted word vector
    target -- the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.
    """
    eachWordProb = softmax(np.dot(predicted, outputVectors.T))
    # Cross entropy cost for the softmax word prediction
    cost = -np.log(eachWordProb[target]) 

    # y^ - y (column vector of the softmax prediction of words - one-hot laber representation)
    eachWordProb[target] -= 1 

    # The gradient with respect to the predicted word vector
    gradPred = np.dot(eachWordProb, outputVectors)
    
    # The gradient with respect to all the other word vectors
    grad = eachWordProb[:, np.newaxis] * predicted[np.newaxis, :]

    return cost, gradPred, grad

# Sample K indexes which are not the target
def getNegativeSamples(target, dataset, K):
    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

# Negative sampling cost function for word2vec models
def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K = 10):
    # Arguments: same as softmaxCostAndGradient. K is the sample size

    # Sampling of indices
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    eachWordProb = np.dot(outputVectors, predicted)
    cost = -np.log(sigmoid(eachWordProb[target])) - np.sum(np.log(sigmoid(-eachWordProb[indices[1:]])))
    
    opposite_sign = (1 - sigmoid(-eachWordProb[indices[1:]]))
    gradPred =  (sigmoid(eachWordProb[target]) - 1) * outputVectors[target] + sum(opposite_sign[:, np.newaxis] * outputVectors[indices[1:]])
    grad = np.zeros_like(outputVectors)
    grad[target] = (sigmoid(eachWordProb[target]) - 1) * predicted

    for k in indices[1:]:
    	grad[k] += (1.0 - sigmoid(-np.dot(outputVectors[k], predicted))) * predicted

    return cost, gradPred, grad

# Implementation for the skip-gram model in word2vec
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2 * C strings, the context words
    tokens -- a dictionary that maps words to their indices in the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for a prediction vector given the target word vectors
    """

    # The cost function value for the skip-gram model
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    centerWord = tokens[currentWord]
    for contextWord in contextWords:
    	target = tokens[contextWord]
    	newCost, newGradPred, newGrad = word2vecCostAndGradient(inputVectors[centerWord], target, outputVectors, dataset)
    	cost += newCost
    	gradIn[centerWord] += newGradPred
    	gradOut += newGrad

    return cost, gradIn, gradOut

# Implementation for the CBOW model in word2vec
def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    # Arguments: same as the skip-gram model

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    target = tokens[currentWord]
    centerWord = np.sum(inputVectors[tokens[contextWord]] for contextWord in contextWords)
    
    cost, gradPred, gradOut = word2vecCostAndGradient(centerWord, target, outputVectors, dataset)

    gradIn = np.zeros_like(inputVectors)
    for contextWord in contextWords:
    	gradIn[tokens[contextWord]] += gradPred

    return cost, gradIn, gradOut

# Helper function - loads previously saved parameters and resets iteration start
def load_saved_params():
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

# Implementation for stochastic gradient descent
def sgd(f, x0, learning_rate, iterations, postprocessing = None, useSaved = False, PRINT_EVERY = 10):
    """ Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a cost and the gradient with respect to the arguments
    x0 -- the initial point to start SGD from
    learning_rate -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            learning_rate *= 0.5 ** (start_iter / ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1):
        cost = None
        cost, grad = f(x)
        x = x - learning_rate * grad
        if(postprocessing):
        	x = postprocessing(x)

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print "iter %d: %f" % (iter, expcost)

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            learning_rate *= 0.5
    return x

# ************** IMPLEMENTATION TESTS **************

def test_softmax():
    print "Running softmax tests..."
    test1 = softmax(np.array([[1,2]]))
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)
    print "Passed!\n"

def test_sigmoid():
    print "Running sigmoid tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "Passed!\n"

def test_gradient_descent_checker():
	# Test square function x^2, grad is 2 * x
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running gradient checker for quad function..."
    gradient_checker(quad, np.array(123.456))
    gradient_checker(quad, np.random.randn(3,))
    gradient_checker(quad, np.random.randn(4,5))
    print "Passed!\n"

    # Test cube function x^3, grad is 3 * x^2
    cube = lambda x: (np.sum(x ** 3), 3 * (x ** 2))

    print "Running gradient checker for cube function..."
    gradient_checker(cube, np.array(123.456))
    gradient_checker(cube, np.random.randn(3,))
    gradient_checker(cube, np.random.randn(4,5))
    print "Passed!\n"

def test_normalize_rows():
    print "Running rows normalization check..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print "Passed!\n"

def test_word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradient_checker(lambda vec: test_word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradient_checker(lambda vec: test_word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradient_checker(lambda vec: test_word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradient_checker(lambda vec: test_word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)

def sgd_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running SGD sanity checks..."
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY = 100)
    print "\nTest 1 result:", t1
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY = 100)
    print "\nTest 2 result:", t2
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY = 100)
    print "\nTest 3 result:", t3
    assert abs(t3) <= 1e-6

    print "SGD tests passed!\n"

# Run method - train word vectors with everything implemented
# Use Stanford Sentiment Treebank (SST)
# To fetch the datasets run sh get datasets.sh
def run():
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    # Train 10-dimensional vectors
    dimVectors = 10

    # Context size
    C = 5

    random.seed(31415)
    np.random.seed(9265)

    startTime = time.time()
    wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - 0.5) / dimVectors, np.zeros((nWords, dimVectors))), axis=0)
    wordVectors = sgd(lambda vec: test_word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient), wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)

    print "Sanity check: cost at convergence should be around or below 10"
    print "Training took %d seconds" % (time.time() - startTime)

    # Concatenate the input and output word vectors
    wordVectors = np.concatenate((wordVectors[:nWords,:], wordVectors[nWords:,:]), axis=0)
    # wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]

    visualizeWords = [
        "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
        "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
        "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
        "annoying"]

    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in xrange(len(visualizeWords)):
        plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.savefig('q3_word_vectors.png') # Save a visualization for the word vectors

if __name__ == "__main__":
    test_softmax()
    test_sigmoid()
    test_gradient_descent_checker()
    test_normalize_rows()
    test_word2vec()
    sgd_check()
    run()