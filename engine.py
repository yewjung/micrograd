import math


class Value:
    def __init__(self, data, _children=(), _op='', label = '') -> None:
        self.data = data
        self._children = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        o = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += o.grad
            other.grad += o.grad

        self._backward = _backward
        return o
    
    def __mul__(self, other):
        o = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * o.grad
            other.grad += self.data * o.grad
            
        self._backward = _backward
        return o
    
    def backward(self):

        topo = [self]
        visited = {self}
        def dfs(node):
            for child in node._children:
                if child not in visited:
                    visited.add(child)
                    topo.append(child)
                    dfs(child)
        dfs(self)
        for node in topo:
            node._backward()

    def tanh(self):

        e = math.exp(2 *  self.data)
        output_val = (e - 1) / (e + 1)
        o =  Value(output_val, (self,), 'tanh')
        def _backward():
            self.grad += 1 - output_val ** 2

        self._backward = _backward
        return o



                

if __name__ == "__main__":
    # a = Value(1.0)
    # b = Value(2.0)
    # c = a + b
    # print(c)

    a = Value(3.0)
    b = Value(2.0)
    c = a * a + b
    d = c.tanh()
    d.grad = 1.0
    d.backward()
    print(d)
    print(c)
    print(a)
    print(b)