import torch

checkpoint_path = "/pscratch/sd/j/jy-nyu/final_expes/bi_4g_1/checkpoints/200/chkpnt0.pth"

(model_params, start_from_this_iteration) = torch.load(checkpoint_path)

(active_sh_degree, 
        xyz, 
        features_dc, 
        features_rest,
        scaling, 
        rotation, 
        opacity,
        max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        spatial_lr_scale) = model_params

# print("active_sh_degree: ", active_sh_degree)
# print("xyz: ", xyz)
# print("features_dc: ", features_dc)
# print("features_rest: ", features_rest)
# print("scaling: ", scaling)
# print("rotation: ", rotation)
# print("opacity: ", opacity)
# print("max_radii2D: ", max_radii2D)
# print("xyz_gradient_accum: ", xyz_gradient_accum)
# print("denom: ", denom)
print("opt_dict: ", opt_dict.keys()) # dict_keys(['state', 'param_groups'])
print("opt_dict['state']: ", opt_dict['state'].keys()) # dict_keys(['step', 'exp_avg', 'exp_avg_sq'])
for key in opt_dict['state'].keys():
    print("opt_dict['state'][{}]: ".format(key), opt_dict['state'][key].keys())
print("opt_dict['param_groups']: ", opt_dict['param_groups']) # list of dicts
# print("spatial_lr_scale: ", spatial_lr_scale)


