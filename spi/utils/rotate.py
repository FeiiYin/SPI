import torch
from torch.nn.functional import grid_sample


def unproject(depth_map, cam2world_matrix, intrinsics, resolution=128):
    N, M = cam2world_matrix.shape[0], resolution**2
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), 
                     torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) \
                     * (1./resolution) + (0.5/resolution)
    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

    x_cam = uv[:, :, 0].view(N, -1)
    y_cam = uv[:, :, 1].view(N, -1)
    z_cam = depth_map.view(N, M)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

    world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)  # [:, :, :3]
    return world_rel_points


def project(world_rel_points, cam2world_matrix, intrinsics, resolution=128):
    N, M = cam2world_matrix.shape[0], resolution**2
    # cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    cam_rel_points = torch.bmm(torch.inverse(cam2world_matrix), world_rel_points.permute(0, 2, 1)).permute(0, 2, 1)
    x_lift = cam_rel_points[:, :, 0]
    y_lift = cam_rel_points[:, :, 1]
    z_cam = cam_rel_points[:, :, 2]

    y_cam = (y_lift / z_cam * fy.unsqueeze(-1)) + cy.unsqueeze(-1)
    x_cam_temp = x_lift / z_cam * fx.unsqueeze(-1)
    x_cam = x_cam_temp + sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1) - cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) + cx.unsqueeze(-1)

    uv = torch.stack([x_cam, y_cam], dim=-1)
    # uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    return uv, z_cam
    


def __rotate(img1, depth1, ex1, img2, depth2, ex2, in1, in2, img2_mask=None, EPS=6e-2):
    N, H, W = depth1.shape
    resolution = H

    xyz1 = unproject(depth1, cam2world_matrix=ex1, intrinsics=in1, resolution=resolution)  # World coord torch.Size([1, 16384, 4])
    uv, z = project(xyz1, ex2, in2, resolution=resolution)
    
    grid_ = uv.view(N, H, W, 2)
    grid_ = 2 * grid_ - 1

    mask = 1 - ((grid_[:,:,:,0] < -1) | (grid_[:,:,:,0] > 1) | (grid_[:,:,:,1] < -1) | (grid_[:,:,:,1] > 1)).float()


    new_depth = z.view(N, H, W)
    # Wrong calculation
    # new_depth = grid_sample(depth2.view(N, 1, H, W), grid_, align_corners=False).view(N, H, W)

    origin_depth_2 = grid_sample(depth2.view(N, 1, H, W), grid_, align_corners=False).view(N, H, W)

    depth_mask = torch.abs(origin_depth_2 - new_depth) < EPS
    # depth_mask = torch.abs(depth2 - new_depth) < EPS
    depth_mask = depth_mask * mask
    depth_mask = depth_mask.unsqueeze(1)
    
    new_rgb = grid_sample(img2, grid_, align_corners=False)
    new_rgb = new_rgb * depth_mask

    if img2_mask is not None:
        # print(img2.shape, )
        new_mask = grid_sample(img2_mask.view(N, 1, H, W), grid_, align_corners=False) #.view(N, H, W)
        new_rgb = new_rgb * new_mask
        depth_mask = depth_mask * new_mask

    return new_rgb, depth_mask


def rotate(target_camera, target_depth, src_image, src_camera, src_depth, src_mask=None, EPS=5e-2):
    N = src_image.shape[0]
    tex = target_camera[:, :16].view(N, 4, 4)
    tin = target_camera[:, 16:].view(N, 3, 3)

    gex = src_camera[:, :16].view(N, 4, 4)
    gin = src_camera[:, 16:].view(N, 3, 3)

    resolution = src_image.shape[-1]

    tdepth = target_depth.cuda().view(N, 128, 128)
    if tdepth.shape[-1] != resolution:
        tdepth = tdepth.view(N, 1, 128, 128)
        tdepth = torch.nn.functional.interpolate(tdepth, (resolution, resolution), mode='bilinear', align_corners=False)
        tdepth = tdepth.view(N, resolution, resolution)

    gdepth = src_depth.cuda().view(N, 128, 128)
    if gdepth.shape[-1] != resolution:
        gdepth = gdepth.view(N, 1, 128, 128)
        gdepth = torch.nn.functional.interpolate(gdepth, (resolution, resolution), mode='bilinear', align_corners=False)
        gdepth = gdepth.view(N, resolution, resolution)

    new_rgb, depth_mask = __rotate(img1=None, depth1=tdepth, ex1=tex, 
                img2=src_image, depth2=gdepth, ex2=gex, in1=tin, in2=gin, img2_mask=src_mask, EPS=EPS)
    return new_rgb, depth_mask


def rotate_with_conffidence(target_camera, target_depth, src_image, src_camera, src_depth, src_mask, confidence_eps=0.1):
    warp_img, warp_mask = rotate(target_camera=target_camera, 
                target_depth=target_depth, 
                src_image=src_image, 
                src_camera=src_camera, 
                src_depth=src_depth, 
                src_mask=src_mask,
                EPS=5e-2)
    
    warp_img_rt, warp_mask_rt = rotate(target_camera=src_camera, 
                target_depth=src_depth, 
                src_image=warp_img, 
                src_camera=target_camera, 
                src_depth=target_depth, 
                src_mask=warp_mask,
                EPS=5e-2)
    # log_image(warp_img_rt, f'warp_img_rt_{_i}')
    confidence_mask = torch.abs(src_image - warp_img_rt)

    # log_image(confidence_mask, f'soft_confidence_mask_{_i}')
    confidence_mask = (torch.sum(confidence_mask, dim=1, keepdim=True) < confidence_eps).float()
    # log_image(confidence_mask, f'confidence_mask_{_i}')
    warp_confidence_mask, warp_mask = rotate(target_camera=target_camera, 
                target_depth=target_depth, 
                src_image=confidence_mask, 
                src_camera=src_camera, 
                src_depth=src_depth, 
                src_mask=src_mask,
                EPS=5e-2)
    # log_image(warp_confidence_mask, f'warp_confidence_mask_{_i}')
    # log_image(warp_confidence_mask * warp_img, f'warp_img_w_confidence_mask_{_i}')
    warp_img_w_confidence_mask = warp_confidence_mask * warp_img
    return warp_img, warp_img_rt, confidence_mask, warp_confidence_mask, warp_img_w_confidence_mask


