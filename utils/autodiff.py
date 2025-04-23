import torch

def batch_jacobian(output, input):
    """
    Computes the Jacobian of a batch of outputs with respect to a batch of inputs.
    output: (batch_size, output.shape[1:])
    input: (batch_size, input.shape[1:])
    returns: (batch_size, output.shape[1:], input.shape[1:])
    """
    
    output_shape = output.shape
    output = output.view(output.shape[0], -1)
    m = output.shape[-1]

    batched_jacobian = torch.zeros(output.shape[0], m, *input.shape[1:], device=output.device)

    grad_outputs = torch.ones_like(output[:, 0], device=output.device)
    for i in range(m):
        batched_jacobian[:, i] = torch.autograd.grad(output[:, i], input, grad_outputs=grad_outputs, create_graph=True)[0]

    return batched_jacobian.view(*output_shape, *input.shape[1:])