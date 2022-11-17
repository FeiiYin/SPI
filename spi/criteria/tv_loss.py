import torch


# density_reg = 0.25
# density_reg_every = 4
density_reg_p_dist = 0.004
box_warp = 1

def cal_tv_loss(ws, G):
    initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
    perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * density_reg_p_dist

    all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
    sigma = G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
    sigma_initial = sigma[:, :sigma.shape[1]//2]
    sigma_perturbed = sigma[:, sigma.shape[1]//2:]

    TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed)
    return TVloss


def cal_monotonic_loss(ws, G):
    initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front
    perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * box_warp # Behind

    all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
    sigma = G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
    sigma_initial = sigma[:, :sigma.shape[1]//2]
    sigma_perturbed = sigma[:, sigma.shape[1]//2:]

    monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
    return monotonic_loss
