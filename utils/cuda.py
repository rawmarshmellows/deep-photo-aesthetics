def cudarize(tensor, use_cuda):
    if use_cuda:
        return tensor.cuda()
    else:
        return tensor