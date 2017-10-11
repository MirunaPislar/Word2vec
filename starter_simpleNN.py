import numpy as np
import random

# Softmax function, optimized such that larger inputs are still feasible
# softmax(x + c) = softmax(x)
def softmax(x):
	orig_shape = x.shape
	x = x - np.max(x, axis = 1, keepdims = True)
	exp_x = np.exp(x)
	x = exp_x / np.sum(exp_x, axis = 1, keepdims = True)
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
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[i], numgrad)
            return

        it.iternext() # Step to next dimension
    print "Gradient check passed!"

# Compute the forward and backward propagation for the NN model
def forward_backward_prop(data, labels, params, dimensions):
	# Unpack the parameters
	Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
	offset = 0
	W1 = np.reshape(params[offset : offset + Dx * H], (Dx, H))
	offset += Dx * H
	b1 = np.reshape(params[offset : offset + 1 * H], (1, H))
	offset += 1 * H
	W2 = np.reshape(params[offset : offset + H * Dy], (H, Dy))
	offset += H * Dy
	b2 = np.reshape(params[offset : offset + 1 * Dy], (1, Dy))

	# Forward propagation
	a0 = data
	z1 = np.dot(a0, W1) + b1
	a1 = sigmoid(z1)
	z2 = np.dot(a1, W2) + b2
	a2 = softmax(z2)
	cost = - np.sum(labels * np.log(a2))

	# Backward propagation
	delta1 = a2 - labels
	dW2 = np.dot(a1.T, delta1)
	db2 = np.sum(delta1, axis = 0, keepdims = True)
	delta2 = np.multiply(np.dot(delta1, W2.T), sigmoid_grad(a1))
	dW1 = np.dot(a0.T, delta2)
	db1 = np.sum(delta2, axis = 0, keepdims = True)

	### Stack gradients
	grad = np.concatenate((dW1.flatten(), db1.flatten(),dW2.flatten(), db2.flatten()))

	return cost, grad

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

if __name__ == "__main__":
    test_softmax()
    test_sigmoid()
    test_gradient_descent_checker()