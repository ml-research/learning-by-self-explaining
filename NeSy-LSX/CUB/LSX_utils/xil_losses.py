import torch

log_softmax = torch.nn.LogSoftmax(dim=1).cuda()


def rrr_loss_function(A, X, y, logits, reduce_func=torch.sum, device='cuda'):
    log_prob_ys = log_softmax(logits)

    gradXes = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys).to(device), create_graph=True)[0]

    for _ in range(len(gradXes.shape) - len(A.shape)):
        A = A.unsqueeze(dim=1)
    expand_list = [-1]*len(A.shape)
    expand_list[-3] = gradXes.shape[-3]
    A = A.expand(expand_list)
    A_gradX = torch.mul(A, gradXes) ** 2

    # right_reason_loss = l2_grads * torch.sum(A_gradX)
    right_reason_loss = torch.sum(A_gradX, dim=list(range(1, len(A_gradX.shape))))
    right_reason_loss = reduce_func(right_reason_loss)

    return right_reason_loss


def rrr_loss_function_captum(A, model_grads, reduce_func=torch.sum):

    A_gradX = torch.mul(A, model_grads) ** 2

    # right_reason_loss = l2_grads * torch.sum(A_gradX)
    right_reason_loss = torch.sum(A_gradX, dim=list(range(1, len(A_gradX.shape))))
    right_reason_loss = reduce_func(right_reason_loss)

    return right_reason_loss



def hint_loss_function(model_symb_expls, feedback_masks):

    assert len(model_symb_expls.shape) == 3
    assert feedback_masks.shape == model_symb_expls.shape

    loss = torch.nn.functional.mse_loss(model_symb_expls, feedback_masks)

    return loss
