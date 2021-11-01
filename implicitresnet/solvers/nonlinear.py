import warnings
import torch
# from torch.optim import Optimizer
from functools import reduce

from .batched_lbfgs import batched_LBFGS


###############################################################################
###############################################################################
# default values

_min_iters    = 0
_max_iters    = 1000
_history_size = 100


###############################################################################
###############################################################################


# def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
#     # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
#     # Compute bounds of interpolation area
#     if bounds is not None:
#         xmin_bound, xmax_bound = bounds
#     else:
#         xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

#     # Code for most common case: cubic interpolation of 2 points
#     #   w/ function and derivative values for both
#     # Solution in this case (where x2 is the farthest point):
#     #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
#     #   d2 = sqrt(d1^2 - g1*g2);
#     #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
#     #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
#     d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
#     d2_square = d1**2 - g1 * g2
#     if d2_square >= 0:
#         d2 = d2_square.sqrt()
#         if x1 <= x2:
#             min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
#         else:
#             min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
#         return min(max(min_pos, xmin_bound), xmax_bound)
#     else:
#         return (xmin_bound + xmax_bound) / 2.


# def _strong_wolfe(obj_func,
#                   x,
#                   t,
#                   d,
#                   f,
#                   g,
#                   gtd,
#                   c1=1e-4,
#                   c2=0.9,
#                   tolerance_change=1e-9,
#                   max_ls=25):
#     # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
#     d_norm = d.abs().max()
#     g = g.clone(memory_format=torch.contiguous_format)
#     # evaluate objective and gradient using initial step
#     f_new, g_new = obj_func(x, t, d)
#     ls_func_evals = 1
#     gtd_new = g_new.dot(d)

#     # bracket an interval containing a point satisfying the Wolfe criteria
#     t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
#     done = False
#     ls_iter = 0
#     print(1,f_new)
#     while ls_iter < max_ls:
#         print(ls_iter)
#         # check conditions
#         if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
#             bracket = [t_prev, t]
#             bracket_f = [f_prev, f_new]
#             bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
#             bracket_gtd = [gtd_prev, gtd_new]
#             break

#         if abs(gtd_new) <= -c2 * gtd:
#             bracket = [t]
#             bracket_f = [f_new]
#             bracket_g = [g_new]
#             done = True
#             break

#         if gtd_new >= 0:
#             bracket = [t_prev, t]
#             bracket_f = [f_prev, f_new]
#             bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
#             bracket_gtd = [gtd_prev, gtd_new]
#             break

#         # interpolate
#         min_step = t + 0.01 * (t - t_prev)
#         max_step = t * 10
#         tmp = t
#         t = _cubic_interpolate(
#             t_prev,
#             f_prev,
#             gtd_prev,
#             t,
#             f_new,
#             gtd_new,
#             bounds=(min_step, max_step))

#         # next step
#         t_prev = tmp
#         f_prev = f_new
#         g_prev = g_new.clone(memory_format=torch.contiguous_format)
#         gtd_prev = gtd_new
#         f_new, g_new = obj_func(x, t, d)
#         ls_func_evals += 1
#         gtd_new = g_new.dot(d)
#         ls_iter += 1
#     print(1,f_new)
#     # reached max number of iterations?
#     if ls_iter == max_ls:
#         bracket = [0, t]
#         bracket_f = [f, f_new]
#         bracket_g = [g, g_new]

#     # zoom phase: we now have a point satisfying the criteria, or
#     # a bracket around it. We refine the bracket until we find the
#     # exact point satisfying the criteria
#     insuf_progress = False
#     # find high and low points in bracket
#     low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
#     while not done and ls_iter < max_ls:
#         # line-search bracket is so small
#         if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
#             break

#         # compute new trial value
#         t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
#                                bracket[1], bracket_f[1], bracket_gtd[1])

#         # test that we are making sufficient progress:
#         # in case `t` is so close to boundary, we mark that we are making
#         # insufficient progress, and if
#         #   + we have made insufficient progress in the last step, or
#         #   + `t` is at one of the boundary,
#         # we will move `t` to a position which is `0.1 * len(bracket)`
#         # away from the nearest boundary point.
#         eps = 0.1 * (max(bracket) - min(bracket))
#         if min(max(bracket) - t, t - min(bracket)) < eps:
#             # interpolation close to boundary
#             if insuf_progress or t >= max(bracket) or t <= min(bracket):
#                 # evaluate at 0.1 away from boundary
#                 if abs(t - max(bracket)) < abs(t - min(bracket)):
#                     t = max(bracket) - eps
#                 else:
#                     t = min(bracket) + eps
#                 insuf_progress = False
#             else:
#                 insuf_progress = True
#         else:
#             insuf_progress = False

#         # Evaluate new point
#         f_new, g_new = obj_func(x, t, d)
#         ls_func_evals += 1
#         gtd_new = g_new.dot(d)
#         ls_iter += 1

#         if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
#             # Armijo condition not satisfied or not lower than lowest point
#             bracket[high_pos] = t
#             bracket_f[high_pos] = f_new
#             bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
#             bracket_gtd[high_pos] = gtd_new
#             low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
#         else:
#             if abs(gtd_new) <= -c2 * gtd:
#                 # Wolfe conditions satisfied
#                 done = True
#             elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
#                 # old high becomes new low
#                 bracket[high_pos] = bracket[low_pos]
#                 bracket_f[high_pos] = bracket_f[low_pos]
#                 bracket_g[high_pos] = bracket_g[low_pos]
#                 bracket_gtd[high_pos] = bracket_gtd[low_pos]

#             # new point becomes new low
#             bracket[low_pos] = t
#             bracket_f[low_pos] = f_new
#             bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
#             bracket_gtd[low_pos] = gtd_new

#     # return stuff
#     t = bracket[low_pos]
#     f_new = bracket_f[low_pos]
#     g_new = bracket_g[low_pos]
#     return f_new, g_new, t, ls_func_evals




# def _batched_cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
#     # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
#     # Compute bounds of interpolation area
#     if bounds is not None:
#         xmin_bound, xmax_bound = bounds
#     else:
#         condition = (x1 <= x2)
#         xmin_bound, xmax_bound = torch.zeros_like(x1), torch.zeros_like(x2)
#         xmin_bound[condition],  xmax_bound[condition]  = x1[condition],  x2[condition]
#         xmin_bound[~condition], xmax_bound[~condition] = x2[~condition], x1[~condition]

#     # Code for most common case: cubic interpolation of 2 points
#     #   w/ function and derivative values for both
#     # Solution in this case (where x2 is the farthest point):
#     #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
#     #   d2 = sqrt(d1^2 - g1*g2);
#     #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
#     #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
#     d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
#     d2_square = d1**2 - g1 * g2

#     result = torch.zeros_like(xmin_bound)

#     condition = (d2_square >= 0)
#     if torch.any(condition):
#         d2 = d2_square.sqrt()
#         condition2 = (x1 <= x2)
#         condition2a = torch.logical_and( condition,  condition2 )
#         condition2b = torch.logical_and( condition, ~condition2 )
#         min_pos = torch.zeros_like(x1)
#         min_pos[condition2a] = (x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)))[condition2a]
#         min_pos[condition2b] = (x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)))[condition2b]
#         result[condition] = torch.minimum( torch.maximum(min_pos, xmin_bound), xmax_bound )[condition]
#     result[~condition] = (xmin_bound + xmax_bound)[~condition] / 2.
#     return result


# def _batched_strong_wolfe(obj_func,
#                   x,
#                   t,
#                   d,
#                   f,
#                   g,
#                   gtd,
#                   c1=1e-4,
#                   c2=0.9,
#                   tolerance_change=1e-9,
#                   max_ls=10):
#     batch_dim = d.size(0)
#     # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
#     d_norm = d.abs().max(dim=1)[0]
#     g = g.clone(memory_format=torch.contiguous_format)
#     # evaluate objective and gradient using initial step
#     f_new, g_new = obj_func(x, t, d)
#     ls_func_evals = 1
#     gtd_new = (g_new*d).sum(dim=1)

#     # bracket an interval containing a point satisfying the Wolfe criteria
#     t_prev, f_prev, g_prev, gtd_prev = torch.zeros_like(t), f, g, gtd
#     done = torch.tensor([False]*batch_dim, dtype=torch.bool, device=gtd.device)
#     nobracket = torch.tensor([True]*batch_dim, dtype=torch.bool, device=gtd.device)
#     bracket_low,     bracket_high     = torch.zeros(batch_dim, dtype=t.dtype, device=gtd.device),   torch.zeros(batch_dim, dtype=t.dtype, device=gtd.device)
#     bracket_f_low,   bracket_f_high   = torch.zeros(batch_dim, dtype=f.dtype, device=gtd.device),   torch.zeros(batch_dim, dtype=f.dtype, device=gtd.device)
#     bracket_g_low,   bracket_g_high   = torch.zeros(*g.shape,  dtype=g.dtype, device=gtd.device),   torch.zeros(*g.shape,  dtype=g.dtype, device=gtd.device)
#     bracket_gtd_low, bracket_gtd_high = torch.zeros(batch_dim, dtype=gtd.dtype, device=gtd.device), torch.zeros(batch_dim, dtype=gtd.dtype, device=gtd.device)
#     ls_iter = 0
#     while ls_iter < max_ls:
#         # check conditions
#         if ls_iter <= 1:
#             condition = torch.logical_and( (f_new > (f + c1 * t.squeeze(1) * gtd)), nobracket)
#         else:
#             condition = torch.logical_and( torch.logical_or( (f_new > (f + c1 * t.squeeze(1) * gtd)), f_new >= f_prev ), nobracket)
#         bracket_low[condition],     bracket_high[condition]     = t_prev[condition,0], t[condition,0]
#         bracket_f_low[condition],   bracket_f_high[condition]   = f_prev[condition],   f_new[condition]
#         bracket_g_low[condition,:], bracket_g_high[condition,:] = g_prev[condition,:], g_new[condition,:]
#         bracket_gtd_low[condition], bracket_gtd_high[condition] = gtd_prev[condition], gtd_new[condition]
#         nobracket[condition] = False
#         if not torch.any(nobracket):
#             break

#         condition = torch.logical_and( gtd_new.abs() <= -c2 * gtd, nobracket)
#         bracket_low[condition],     bracket_high[condition]     = t[condition,0],     t[condition,0]
#         bracket_f_low[condition],   bracket_f_high[condition]   = f_new[condition],   f_new[condition]
#         bracket_g_low[condition,:], bracket_g_high[condition,:] = g_new[condition,:], g_new[condition,:]
#         done[condition] = True
#         nobracket[condition] = False
#         if not torch.any(nobracket):
#             break

#         condition = torch.logical_and( gtd_new >= 0, nobracket)
#         bracket_low[condition],     bracket_high[condition]     = t_prev[condition,0], t[condition,0]
#         bracket_f_low[condition],   bracket_f_high[condition]   = f_prev[condition],   f_new[condition]
#         bracket_g_low[condition,:], bracket_g_high[condition,:] = g_prev[condition,:], g_new[condition,:]
#         bracket_gtd_low[condition], bracket_gtd_high[condition] = gtd_prev[condition], gtd_new[condition]
#         nobracket[condition] = False
#         if not torch.any(nobracket):
#             break

#         # interpolate
#         min_step = t + 0.01 * (t - t_prev)
#         max_step = t * 10
#         tmp = t
#         t = t.clone()
#         t[nobracket,0] = _batched_cubic_interpolate(
#             t_prev.squeeze(1),
#             f_prev,
#             gtd_prev,
#             t.squeeze(1),
#             f_new,
#             gtd_new,
#             bounds=(min_step.squeeze(1), max_step.squeeze(1)))[nobracket]

#         # next step
#         t_prev = tmp
#         f_prev = f_new
#         g_prev = g_new
#         gtd_prev = gtd_new
#         f_new, g_new = obj_func(x, t, d)
#         # print(nobracket, t[:,0], f_new)
#         ls_func_evals += 1
#         gtd_new = (g_new*d).sum(1)
#         ls_iter += 1

#     # reached max number of iterations?
#     if ls_iter == max_ls:
#         bracket_low[nobracket],     bracket_high[nobracket]     = torch.zeros_like(bracket_low[nobracket]), t[nobracket,0]
#         bracket_f_low[nobracket],   bracket_f_high[nobracket]   = f[nobracket],   f_new[nobracket]
#         bracket_g_low[nobracket,:], bracket_g_high[nobracket,:] = g[nobracket,:], g_new[nobracket,:]

#     # low_pos  = torch.zeros(batch_dim, dtype=torch.long)
#     # high_pos = torch.zeros(batch_dim, dtype=torch.long)
#     # condition = (bracket_f_low <= bracket_f_high)
#     # low_pos[condition],  low_pos[~condition]  = 0, 1
#     # high_pos[condition], high_pos[~condition] = 1, 0

#     def swap_low_high(condition):
#         bracket_tmp = bracket_low.clone();     bracket_low[condition],     bracket_high[condition]     = bracket_high[condition],     bracket_tmp[condition]
#         bracket_tmp = bracket_f_low.clone();   bracket_f_low[condition],   bracket_f_high[condition]   = bracket_f_high[condition],   bracket_tmp[condition]
#         bracket_tmp = bracket_g_low.clone();   bracket_g_low[condition,:], bracket_g_high[condition,:] = bracket_g_high[condition,:], bracket_tmp[condition,:]
#         bracket_tmp = bracket_gtd_low.clone(); bracket_gtd_low[condition], bracket_gtd_high[condition] = bracket_gtd_high[condition], bracket_tmp[condition]

#     # zoom phase: we now have a point satisfying the criteria, or
#     # a bracket around it. We refine the bracket until we find the
#     # exact point satisfying the criteria
#     insuf_progress = torch.tensor([False]*batch_dim, dtype=torch.bool, device=gtd.device)
#     # find high and low points in bracket
#     swap_low_high(bracket_f_low > bracket_f_high)
#     while not torch.all(done) and ls_iter < max_ls:
#         # line-search bracket is so small
#         proceed = (bracket_high - bracket_low).abs() * d_norm >= tolerance_change
#         if torch.any(proceed):
#             # compute new trial value
#             t = _batched_cubic_interpolate(bracket_low, bracket_f_low, bracket_gtd_low,
#                                            bracket_low, bracket_f_low, bracket_gtd_low).unsqueeze(1)

#             # test that we are making sufficient progress:
#             # in case `t` is so close to boundary, we mark that we are making
#             # insufficient progress, and if
#             #   + we have made insufficient progress in the last step, or
#             #   + `t` is at one of the boundary,
#             # we will move `t` to a position which is `0.1 * len(bracket)`
#             # away from the nearest boundary point.
#             bracket_max = torch.maximum(bracket_low, bracket_high)
#             bracket_min = torch.minimum(bracket_low, bracket_high)
#             eps = 0.1 * (bracket_max - bracket_min)
#             condition = torch.minimum( bracket_max - t.squeeze(1), t.squeeze(1) - bracket_min ) < eps
#             conditiona = torch.logical_and( proceed,  condition )
#             conditionb = torch.logical_and( proceed, ~condition )
#             if torch.any(conditiona):
#                 # interpolation close to boundary
#                 condition2  = torch.logical_or( insuf_progress, torch.logical_or( t.squeeze(1) >= bracket_max, t.squeeze(1) <= bracket_min ) )
#                 condition2a = torch.logical_and( conditiona,  condition2 )
#                 condition2b = torch.logical_and( conditiona, ~condition2 )
#                 # evaluate at 0.1 away from boundary
#                 if torch.any(condition2a):
#                     condition3  = (t.squeeze(1) - bracket_max).abs() < (t.squeeze(1) - bracket_min).abs()
#                     condition3a = torch.logical_and( condition2a,  condition3 )
#                     condition3b = torch.logical_and( condition2a, ~condition3 )
#                     t[condition3a,0] = (bracket_max - eps)[condition3a]
#                     t[condition3b,0] = (bracket_min + eps)[condition3b]
#                 insuf_progress[condition2b] = True
#             insuf_progress[conditionb] = False

#             # Evaluate new point
#             f_new, g_new = obj_func(x, t, d)
#             ls_func_evals += 1
#             gtd_new = (g_new*d).sum(1)
#             ls_iter += 1

#             condition  = torch.logical_or( f_new > (f + c1 * t.squeeze(1) * gtd), f_new >= bracket_f_low )
#             conditiona = torch.logical_and( proceed,  condition )
#             conditionb = torch.logical_and( proceed, ~condition )
#             if torch.any(conditiona):
#                 # Armijo condition not satisfied or not lower than lowest point
#                 bracket_high[conditiona] = t[conditiona,0]
#                 bracket_f_high[conditiona] = f_new[conditiona]
#                 bracket_g_high[conditiona,:] = g_new[conditiona,:].clone(memory_format=torch.contiguous_format)
#                 bracket_gtd_high[conditiona] = gtd_new[conditiona]
#                 swap_low_high(bracket_f_low > bracket_f_high)
#             condition2  = abs(gtd_new) <= -c2 * gtd
#             condition2a = torch.logical_and( conditionb,  condition2 )
#             condition2b = torch.logical_and( conditionb, ~condition2 )
#             # Wolfe conditions satisfied
#             done[condition2a] = True
#             condition3 = gtd_new * (bracket_high - bracket_low) >= 0
#             condition3 = torch.logical_and( condition2b,  condition3 )
#             # old high becomes new low
#             bracket_high[condition3] = bracket_low[condition3]
#             bracket_g_high[condition3,:] = bracket_g_low[condition3,:]
#             bracket_gtd_high[condition3] = bracket_gtd_low[condition3]
#             # new point becomes new low
#             bracket_low[conditionb] = t[conditionb,0]
#             bracket_f_low[conditionb] = f_new[conditionb]
#             bracket_g_low[conditionb,:] = g_new[conditionb,:].clone(memory_format=torch.contiguous_format)
#             bracket_gtd_low[conditionb] = gtd_new[conditionb]
#         else:
#             break

#     # return stuff
#     t = bracket_low.unsqueeze(1)
#     f_new = bracket_f_low
#     g_new = bracket_g_low
#     return f_new, g_new, t, ls_func_evals



# # def _strong_wolfe(obj_func,
# #                   x,
# #                   t,
# #                   d,
# #                   f,
# #                   g,
# #                   gtd,
# #                   c1=1e-4,
# #                   c2=0.9,
# #                   tolerance_change=1e-9,
# #                   max_ls=25):
# #     batch_dim = d.size(0)

# #     # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
# #     d_norm = d.abs().max(dim=1)[0]
# #     g = g.clone(memory_format=torch.contiguous_format)
# #     # evaluate objective and gradient using initial step
# #     f_new, g_new = obj_func(x, t, d)
# #     ls_func_evals = 1
# #     # gtd_new = g_new.dot(d)
# #     gtd_new = (g_new*d).sum(1)

# #     # bracket an interval containing a point satisfying the Wolfe criteria
# #     t_prev, f_prev, g_prev, gtd_prev = torch.zeros(batch_dim), f, g, gtd
# #     done = [False] * batch_dim
# #     bracket     = [[] for i in range(batch_dim)]
# #     bracket_f   = [[] for i in range(batch_dim)]
# #     bracket_g   = [[] for i in range(batch_dim)]
# #     bracket_gtd = [[] for i in range(batch_dim)]
# #     ls_iter = 0
# #     while ls_iter < max_ls:
# #         # check conditions
# #         for i in range(batch_dim):
# #             if f_new[i] > (f[i] + c1 * t[i] * gtd[i]) or (ls_iter > 1 and f_new[i] >= f_prev[i]):
# #                 bracket[i] = [t_prev[i], t[i]]
# #                 bracket_f[i] = [f_prev[i], f_new[i]]
# #                 bracket_g[i] = [g_prev[i], g_new[i].clone(memory_format=torch.contiguous_format)]
# #                 bracket_gtd[i] = [gtd_prev[i], gtd_new[i]]
# #                 break

# #             if abs(gtd_new[i]) <= -c2 * gtd[i]:
# #                 bracket[i] = [t[i]]
# #                 bracket_f[i] = [f_new[i]]
# #                 bracket_g[i] = [g_new[i]]
# #                 done[i] = True
# #                 break

# #             if gtd_new[i] >= 0:
# #                 bracket[i] = [t_prev[i], t[i]]
# #                 bracket_f[i] = [f_prev[i], f_new[i]]
# #                 bracket_g[i] = [g_prev[i], g_new[i].clone(memory_format=torch.contiguous_format)]
# #                 bracket_gtd[i] = [gtd_prev[i], gtd_new[i]]
# #                 break

# #         # interpolate
# #         tmp = t
# #         for i in range(batch_dim):
# #             min_step = t[i] + 0.01 * (t[i] - t_prev[i])
# #             max_step = t[i] * 10
# #             t[i] = _cubic_interpolate(
# #                 t_prev[i],
# #                 f_prev[i],
# #                 gtd_prev[i],
# #                 t[i],
# #                 f_new[i],
# #                 gtd_new[i],
# #                 bounds=(min_step, max_step))

# #         # next step
# #         t_prev = tmp
# #         f_prev = f_new
# #         g_prev = g_new.clone(memory_format=torch.contiguous_format)
# #         gtd_prev = gtd_new
# #         f_new, g_new = obj_func(x, t, d)
# #         ls_func_evals += 1
# #         # gtd_new = g_new.dot(d)
# #         gtd_new = (g_new*d).sum(1)
# #         ls_iter += 1
# #     print(t)
# #     exit()
# #     # reached max number of iterations?
# #     if ls_iter == max_ls:
# #         bracket = [[0, t[i]] for i in range(batch_dim)]
# #         bracket_f = [[f[i], f_new[i]] for i in range(batch_dim)]
# #         bracket_g = [[g[i], g_new[i]] for i in range(batch_dim)]

# #     print(t)
# #     exit()

# #     # zoom phase: we now have a point satisfying the criteria, or
# #     # a bracket around it. We refine the bracket until we find the
# #     # exact point satisfying the criteria
# #     insuf_progress = [False] * batch_dim
# #     # find high and low points in bracket
# #     low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
# #     while not done and ls_iter < max_ls:
# #         for i in range(batch_dim):
# #             # line-search bracket is so small
# #             if abs(bracket[i][1] - bracket[i][0]) * d_norm[i] < tolerance_change:
# #                break

# #             # compute new trial value
# #             t = _cubic_interpolate(bracket[i][0], bracket_f[i][0], bracket_gtd[i][0],
# #                                    bracket[i][1], bracket_f[i][1], bracket_gtd[i][1])

# #             # test that we are making sufficient progress:
# #             # in case `t` is so close to boundary, we mark that we are making
# #             # insufficient progress, and if
# #             #   + we have made insufficient progress in the last step, or
# #             #   + `t` is at one of the boundary,
# #             # we will move `t` to a position which is `0.1 * len(bracket)`
# #             # away from the nearest boundary point.
# #             eps = 0.1 * (max(bracket[i]) - min(bracket[i]))
# #             if min(max(bracket[i]) - t[i], t[i] - min(bracket[i])) < eps:
# #                 # interpolation close to boundary
# #                 if insuf_progress[i] or t[i] >= max(bracket[i]) or t[i] <= min(bracket[i]):
# #                     # evaluate at 0.1 away from boundary
# #                     if abs(t[i] - max(bracket[i])) < abs(t[i] - min(bracket[i])):
# #                         t[i] = max(bracket[i]) - eps
# #                     else:
# #                         t[i] = min(bracket[i]) + eps
# #                     insuf_progress[i] = False
# #                 else:
# #                     insuf_progress[i] = True
# #             else:
# #                 insuf_progress[i] = False

# #         # Evaluate new point
# #         f_new, g_new = obj_func(x, t, d)
# #         ls_func_evals += 1
# #         gtd_new = g_new.dot(d)
# #         ls_iter += 1

# #         if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
# #             # Armijo condition not satisfied or not lower than lowest point
# #             bracket[high_pos] = t
# #             bracket_f[high_pos] = f_new
# #             bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
# #             bracket_gtd[high_pos] = gtd_new
# #             low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
# #         else:
# #             if abs(gtd_new) <= -c2 * gtd:
# #                 # Wolfe conditions satisfied
# #                 done = True
# #             elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
# #                 # old high becomes new low
# #                 bracket[high_pos] = bracket[low_pos]
# #                 bracket_f[high_pos] = bracket_f[low_pos]
# #                 bracket_g[high_pos] = bracket_g[low_pos]
# #                 bracket_gtd[high_pos] = bracket_gtd[low_pos]

# #             # new point becomes new low
# #             bracket[low_pos] = t
# #             bracket_f[low_pos] = f_new
# #             bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
# #             bracket_gtd[low_pos] = gtd_new

# #     print(t)
# #     exit()

# #     # return stuff
# #     t = bracket[low_pos]
# #     f_new = bracket_f[low_pos]
# #     g_new = bracket_g[low_pos]
# #     return f_new, g_new, t, ls_func_evals


# class batched_LBFGS(Optimizer):
#     """Implements L-BFGS algorithm, heavily inspired by `minFunc
#     <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.

#     .. warning::
#         This optimizer doesn't support per-parameter options and parameter
#         groups (there can be only one).

#     .. warning::
#         Right now all parameters have to be on a single device. This will be
#         improved in the future.

#     .. note::
#         This is a very memory intensive optimizer (it requires additional
#         ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
#         try reducing the history size, or use a different algorithm.

#     Args:
#         lr (float): learning rate (default: 1)
#         max_iter (int): maximal number of iterations per optimization step
#             (default: 20)
#         max_eval (int): maximal number of function evaluations per optimization
#             step (default: max_iter * 1.25).
#         tolerance_grad (float): termination tolerance on first order optimality
#             (default: 1e-5).
#         tolerance_change (float): termination tolerance on function
#             value/parameter changes (default: 1e-9).
#         history_size (int): update history size (default: 100).
#         line_search_fn (str): either 'strong_wolfe' or None (default: None).
#     """

#     def __init__(self,
#                  params,
#                  lr=1,
#                  tol=1.e-8,
#                  max_iter=20,
#                  max_eval=None,
#                  tolerance_grad=1e-7,
#                  tolerance_change=1e-9,
#                  history_size=100,
#                  line_search_fn=None):
#         if max_eval is None:
#             max_eval = max_iter * 5 // 4
#         defaults = dict(
#             lr=lr,
#             tol=tol,
#             max_iter=max_iter,
#             max_eval=max_eval,
#             tolerance_grad=tolerance_grad,
#             tolerance_change=tolerance_change,
#             history_size=history_size,
#             line_search_fn=line_search_fn)
#         super(batched_LBFGS, self).__init__(params, defaults)

#         if len(self.param_groups) != 1:
#             raise ValueError("LBFGS doesn't support per-parameter options "
#                              "(parameter groups)")

#         self._params = self.param_groups[0]['params']
#         self._numel_cache = None

#     def _numel(self):
#         if self._numel_cache is None:
#             self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
#         return self._numel_cache

#     def _gather_flat_grad(self):
#         views = []
#         for p in self._params:
#             batch_dim = p.size(0)
#             if p.grad is None:
#                 # view = p.new(p.numel()).zero_()
#                 view = p.new(batch_dim,p.numel()//p.size(0)).zero_()
#             elif p.grad.is_sparse:
#             	# view = p.grad.to_dense().view(-1)
#                 view = p.grad.to_dense().view(batch_dim,-1)
#             else:
#             	# view = p.grad.view(-1)
#                 view = p.grad.view(batch_dim,-1)
#             views.append(view)
#         return torch.cat(views, 0)

#     def _add_grad(self, step_size, update):
#         offset = 0
#         # for p in self._params:
#         #     numel = p.numel()
#         #     # view as to avoid deprecated pointwise semantics
#         #     p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
#         #     offset += numel
#         # assert offset == self._numel()
#         p = self._params[0]
#         p += (step_size * update).view_as(p)

#     def _clone_param(self):
#         return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

#     def _set_param(self, params_data):
#         for p, pdata in zip(self._params, params_data):
#             p.copy_(pdata)

#     # def _directional_evaluate(self, closure, x, t, d):
#     #     self._add_grad(t, d)
#     #     # loss = float(closure())
#     #     loss = closure().float()
#     #     flat_grad = self._gather_flat_grad()
#     #     self._set_param(x)
#     #     return loss, flat_grad

#     @torch.no_grad()
#     def step(self, closure):
#         """Performs a single optimization step.

#         Args:
#             closure (callable): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         assert len(self.param_groups) == 1

#         # Make sure the closure is always called with grad enabled
#         closure = torch.enable_grad()(closure)

#         group = self.param_groups[0]
#         lr = group['lr']
#         tol = group['tol']
#         max_iter = group['max_iter']
#         max_eval = group['max_eval']
#         tolerance_grad = group['tolerance_grad']
#         tolerance_change = group['tolerance_change']
#         line_search_fn = group['line_search_fn']
#         history_size = group['history_size']

#         # NOTE: LBFGS has only global state, but we register it as state for
#         # the first param, because this helps with casting in load_state_dict
#         state = self.state[self._params[0]]
#         state.setdefault('func_evals', 0)
#         state.setdefault('n_iter', 0)

#         # evaluate initial f(x) and df/dx
#         orig_loss = closure()
#         # loss = float(orig_loss)
#         loss = orig_loss.float()
#         current_evals = 1
#         state['func_evals'] += 1

#         flat_grad = self._gather_flat_grad()
#         opt_cond = flat_grad.abs().max() <= tolerance_grad

#         batch_dim = flat_grad.size(0)

#         # optimal condition
#         if opt_cond:
#             return orig_loss

#         # tensors cached in state (for tracing)
#         d = state.get('d')
#         t = state.get('t')
#         old_dirs = state.get('old_dirs')
#         old_stps = state.get('old_stps')
#         ro = state.get('ro')
#         H_diag = state.get('H_diag')
#         prev_flat_grad = state.get('prev_flat_grad')
#         prev_loss = state.get('prev_loss')


#         n_iter = 0
#         # optimize for a max of max_iter iterations
#         while n_iter < max_iter and loss.max()>tol:
#             # keep track of nb of iterations
#             n_iter += 1
#             state['n_iter'] += 1

#             ############################################################
#             # compute gradient descent direction
#             ############################################################
#             if state['n_iter'] == 1:
#                 d = flat_grad.neg()
#                 old_dirs = []
#                 old_stps = []
#                 ro = []
#                 # H_diag = 1
#                 H_diag = torch.ones(batch_dim,1)
#             else:
#                 # do lbfgs update (update memory)
#                 y = flat_grad.sub(prev_flat_grad)
#                 s = d.mul(t)
#                 # ys = y.dot(s)  # y*s
#                 # s = d*t
#                 ys = (y*s).sum(1,keepdim=True)  # y*s
#                 # print(ys.shape, ys.max())
#                 # exit()
#                 # if ys > 1e-10:
#                 if ys.max() > 1e-10:
#                     # updating memory
#                     if len(old_dirs) == history_size:
#                         # shift history by one (limited-memory)
#                         old_dirs.pop(0)
#                         old_stps.pop(0)
#                         ro.pop(0)

#                     # store new direction/step
#                     old_dirs.append(y)
#                     old_stps.append(s)
#                     ro.append(1. / (ys+1.e-12))

#                     # update scale of initial Hessian approximation
#                     # H_diag = ys / y.dot(y)  # (y*y)
#                     H_diag = ys / ((y*y).sum(1,keepdim=True)+1.e-12)  # (y*y)

#                 # compute the approximate (L-BFGS) inverse Hessian
#                 # multiplied by the gradient
#                 num_old = len(old_dirs)

#                 if 'al' not in state:
#                     state['al'] = [None] * history_size
#                 al = state['al']

#                 # iteration in L-BFGS loop collapsed to use just one buffer
#                 q = flat_grad.neg()
#                 for i in range(num_old - 1, -1, -1):
#                     # al[i] = old_stps[i].dot(q) * ro[i]
#                     # q.add_(old_dirs[i], alpha=-al[i])
#                     al[i] = (old_stps[i]*q).sum(1,keepdim=True) * ro[i]
#                     q -= al[i]*old_dirs[i]
#                     # print(q[-2])
#                 # multiply by initial Hessian
#                 # r/d is the final direction
#                 # d = r = torch.mul(q, H_diag)
#                 d = r = q * H_diag
#                 for i in range(num_old):
#                     # be_i = old_dirs[i].dot(r) * ro[i]
#                     # r.add_(old_stps[i], alpha=al[i] - be_i)
#                     be_i = (old_dirs[i]*r).sum(1,keepdim=True) * ro[i]
#                     r += (al[i]-be_i)*old_stps[i]
#             if prev_flat_grad is None:
#                 prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
#             else:
#                 prev_flat_grad.copy_(flat_grad)
#             prev_loss = loss

#             ############################################################
#             # compute step length
#             ############################################################
#             # reset initial guess for step size
#             if state['n_iter'] == 1:
#                 # t = min(1., 1. / flat_grad.abs().sum()) * lr
#                 # t = min(1., 1. / flat_grad.abs().sum()) * lr * torch.ones_like(H_diag)
#                 t = torch.minimum(torch.tensor(1., device=flat_grad.device), 1. / flat_grad.abs().sum(1,keepdim=True)) * lr
#             else:
#             	# t = lr
#                 t = lr * torch.ones_like(H_diag)

#             # directional derivative
#             # gtd = flat_grad.dot(d)  # g * d
#             gtd = (flat_grad*d).sum(1)  # g * d
#             # print(flat_grad[0])
#             # print(d[0])
#             # print(gtd[0])
#             # exit()

#             # directional derivative is below tolerance
#             # if gtd > -tolerance_change:
#             # print(gtd.abs().mean(),d.abs().sum(1).mean())
#             if gtd.min() > -tolerance_change:
#                 break

#             # optional line search: user function
#             ls_func_evals = 0
#             if line_search_fn is not None:
#                 # perform line search, using user function
#                 if line_search_fn != "strong_wolfe":
#                     raise RuntimeError("only 'strong_wolfe' is supported")
#                 else:
#                     x_init = self._clone_param()

#                     # for i in range(batch_dim):
#                     #     def obj_func(x, t, d):
#                     #         # self._add_grad(t, d)
#                     #         p = self._params[0][i]
#                     #         p += (t * d).view_as(p)
#                     #         loss = closure().float()[i]
#                     #         flat_grad = self._gather_flat_grad()[i]
#                     #         self._params[0][i].copy_(x)
#                     #         return loss, flat_grad
#                     #         # return self._directional_evaluate(closure, x, t, d)

#                     #     lossi, flat_gradi, ti, ls_func_evals = _strong_wolfe(obj_func, x_init[0][i], t[i][0], d[i], loss[i], flat_grad[i], gtd[i])
#                     #     t[i][0] = ti
#                     #     # loss[i] = lossi
#                     #     # p = self._params[0][i]
#                     #     # # print(ti,d[i])
#                     #     # p += (ti * d[i]).view_as(p)
#                     # print(t)
#                     # exit()

#                     def obj_func(x, t, d):
#                         # self._add_grad(t, d)
#                         p = self._params[0]
#                         p += (t * d).view_as(p)
#                         loss = closure().float()
#                         flat_grad = self._gather_flat_grad()
#                         self._set_param(x)
#                         return loss, flat_grad
#                         # return self._directional_evaluate(closure, x, t, d)

#                     loss, flat_grad, t, ls_func_evals = _batched_strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd)
#                     # print(t)
#                     # exit()

#                     # def obj_func(x, t, d):
#                     #     # self._add_grad(t, d)
#                     #     p = self._params[0]
#                     #     p += (t * d).view_as(p)
#                     #     loss, flat_grad = closure().float().sum(), self._gather_flat_grad().view(-1)
#                     #     self._set_param(x)
#                     #     return loss, flat_grad
#                     #     # return self._directional_evaluate(closure, x, t, d)

#                     # loss, flat_grad, t, ls_func_evals = _strong_wolfe(obj_func, x_init, t, d.view(-1), loss.sum(), flat_grad.view(-1), gtd.sum())
#                 # print(t)
#                 # exit()
#                 self._add_grad(t, d)
#                 # with torch.enable_grad():
#                 #     loss = closure().float()
#                 # flat_grad = self._gather_flat_grad()
#                 # ls_func_evals += 1
#                 opt_cond = flat_grad.abs().max() <= tolerance_grad
#             else:
#                 # no line search, simply move with fixed-step
#                 self._add_grad(t, d)
#                 if n_iter != max_iter:
#                     # re-evaluate function only if not in last iteration
#                     # the reason we do this: in a stochastic setting,
#                     # no use to re-evaluate that function here
#                     with torch.enable_grad():
#                         # loss = float(closure())
#                         loss = closure().float()
#                     flat_grad = self._gather_flat_grad()
#                     opt_cond = flat_grad.abs().max() <= tolerance_grad
#                     ls_func_evals = 1

#             # update func eval
#             current_evals += ls_func_evals
#             state['func_evals'] += ls_func_evals

#             ############################################################
#             # check conditions
#             ############################################################
#             if n_iter == max_iter:
#                 break

#             if current_evals >= max_eval:
#                 break

#             # optimal condition
#             if opt_cond:
#                 break

#             # print(n_iter, t.view(-1), d.abs().max(), abs(loss - prev_loss).max(), loss.max(), prev_loss.max())
#             # print(n_iter, t.view(-1), loss)
#             # print(n_iter, current_evals, loss)

#             # lack of progress
#             # if d.mul(t).abs().max() <= tolerance_change:
#             if (d*t).abs().max() <= tolerance_change:
#                 break

#             # if abs(loss - prev_loss) < tolerance_change:
#             if (loss - prev_loss).abs().max() < tolerance_change:
#                 break

#             if (loss-prev_loss).max() > 0:
#                 break

#         state['d'] = d
#         state['t'] = t
#         state['old_dirs'] = old_dirs
#         state['old_stps'] = old_stps
#         state['ro'] = ro
#         state['H_diag'] = H_diag
#         state['prev_flat_grad'] = prev_flat_grad
#         state['prev_loss'] = prev_loss

#         return loss



def lbfgs( fun, x0, tol=None, max_iters=_max_iters, min_iters=_min_iters, history_size=_history_size, batched=True ):
	dtype  = x0.dtype
	device = x0.device

	if tol is None: tol = 10*torch.finfo(dtype).eps

	# check initial residual
	with torch.no_grad():
		error = fun(x0).max()
		if error<tol:
			return x0.detach(), error.detach(), 0, 0

	# initial condition: make new (that's why clone) leaf (that's why detach) node which requires gradient
	x = x0.clone().detach().requires_grad_(True)

	if not batched:
		iters = [0]
		error = [0]
		nsolver = torch.optim.LBFGS([x], lr=1, max_iter=max_iters, max_eval=10*max_iters, tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=history_size, line_search_fn='strong_wolfe')
		def closure():
			resid = fun(x)
			error[0] = resid.max().detach()
			# error[0] = resid.mean().detach()
			residual = resid.mean()
			nsolver.zero_grad()
			# if error[0]>tol: residual.backward()
			# use .grad() instead of .backward() to avoid evaluation of gradients for leaf parameters which must be frozen inside nsolver
			if error[0]>tol or iters[0]<min_iters: x.grad, = torch.autograd.grad(residual, x, only_inputs=True, allow_unused=False)
			iters[0] += 1
			return residual
		nsolver.step(closure)
		error, fevals = error[0], iters[0]
	else:
		batch_size   = x0.size(0)
		history_size = max(50,int(history_size/batch_size))
		nsolver = batched_LBFGS([x], lr=1, tol=tol, max_iter=max_iters, max_eval=10*max_iters, tolerance_grad=tol/100., tolerance_change=tol/100., history_size=history_size, line_search_fn='strong_wolfe')
		def closure():
			nsolver.zero_grad()
			residual = fun(x)
			# use .grad() instead of .backward() to avoid evaluation of gradients for leaf parameters which must be frozen inside nsolver
			x.grad, = torch.autograd.grad(residual.sum(), x, only_inputs=True, allow_unused=False)
			return residual
		error  = nsolver.step(closure).max()
		niters = nsolver.state_dict()['state'][0]['n_iter']
		fevals = nsolver.state_dict()['state'][0]['func_evals']

	flag = int(error>tol)

	# print(fevals, error)
	# exit()

	return x.detach(), error, niters, fevals, flag



def adam( fun, x0, tol=None, max_iters=_max_iters, min_iters=_min_iters, batch_error='max' ):
	error = 0
	flag  = 0

	if batch_error=='max':
		batch_err = lambda z: z.amax()
	elif batch_error=='mean':
		batch_err = lambda z: z.mean()

	dtype  = x0.dtype
	device = x0.device

	if tol is None: tol = 10*torch.finfo(dtype).eps

	# check initial residual
	with torch.no_grad():
		error = batch_err(fun(x0))
		if error<tol: return x0.detach(), error.detach(), 0, flag

	# initial condition: make new (that's why clone) leaf (that's why detach) node which requires gradient
	x = x0.clone().detach().requires_grad_(True)
	nsolver = torch.optim.Adam([x], lr=1.e0)
	for iters in range(max_iters+1):
		resid = fun(x)
		error = batch_err(resid)
		if iters>=min_iters and error<=tol:
			break
		residual = resid.mean()
		nsolver.zero_grad()
		x.grad, = torch.autograd.grad(residual, x, only_inputs=True, allow_unused=False)
		nsolver.step()

	# if error>tol:
	# 	x, error, iters2, flag = lbfgs(fun, x, tol=tol, max_iters=max_iters, min_iters=min_iters, batch_error=batch_error)
	if error>tol: flag=1

	return x.detach(), error.detach(), iters, flag



def nsolve(fun, x0, method='lbfgs', **kwargs):
	if method=='lbfgs':
		return lbfgs( fun, x0, **kwargs )
	if method=='adam':
		return adam( fun, x0, **kwargs )