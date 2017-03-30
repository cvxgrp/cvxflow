
def normalize_tuple(value, n, name):
    if isinstance(value, tuple):
        if len(value) != n:
            raise ValueError("The '" + name + "' argument must be a tuple of "
                             "length " + str(n) + ". Received: " + str(value))
        return value
    return (value,)*2
