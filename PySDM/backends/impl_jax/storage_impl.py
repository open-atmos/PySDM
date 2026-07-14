import jax


@jax.jit
def multiply(output, multiplier):
    output *= multiplier
    return output


@jax.jit
def divide_out_of_place(output, dividend, divisor):
    output = output.at[:].set(dividend / divisor)
    return output
