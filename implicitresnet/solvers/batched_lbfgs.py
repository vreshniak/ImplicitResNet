import torch
from torch.optim import Optimizer



"""
	Adapted from https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
"""
def _batched_cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
	# ported from https://github.com/torch/optim/blob/master/polyinterp.lua
	# Compute bounds of interpolation area
	if bounds is not None:
		xmin_bound, xmax_bound = bounds
	else:
		condition = (x1 <= x2)
		xmin_bound, xmax_bound = torch.zeros_like(x1), torch.zeros_like(x2)
		xmin_bound[condition],  xmax_bound[condition]  = x1[condition],  x2[condition]
		xmin_bound[~condition], xmax_bound[~condition] = x2[~condition], x1[~condition]

	# Code for most common case: cubic interpolation of 2 points
	#   w/ function and derivative values for both
	# Solution in this case (where x2 is the farthest point):
	#   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
	#   d2 = sqrt(d1^2 - g1*g2);
	#   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
	#   t_new = min(max(min_pos,xmin_bound),xmax_bound);
	d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
	d2_square = d1**2 - g1 * g2

	result = torch.zeros_like(xmin_bound)

	condition = (d2_square >= 0)
	if torch.any(condition):
		d2 = d2_square.sqrt()
		condition2 = (x1 <= x2)
		condition2a = torch.logical_and( condition,  condition2 )
		condition2b = torch.logical_and( condition, ~condition2 )
		min_pos = torch.zeros_like(x1)
		min_pos[condition2a] = (x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)))[condition2a]
		min_pos[condition2b] = (x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)))[condition2b]
		result[condition] = torch.minimum( torch.maximum(min_pos, xmin_bound), xmax_bound )[condition]
	result[~condition] = (xmin_bound + xmax_bound)[~condition] / 2.
	return result


def _batched_strong_wolfe_old(obj_func,
				  x,
				  t,
				  d,
				  f,
				  g,
				  gtd,
				  c1=1e-4,
				  c2=0.9,
				  tolerance_change=1e-9,
				  max_ls=10):
	'''
	A Line Search satisfying the Wolfe conditions

	Ported from https://github.com/torch/optim/blob/master/lswolfe.lua and https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS

	Args:
		obj_func         : a function (the objective) that takes a single input (x), the point of evaluation, and returns f(x) and df/dx
		x                : initial point / starting location
		t                : initial step size
		d                : descent direction
		f                : initial function value
		g                : gradient at initial location
		gtd              : directional derivative at starting location
		c1               : sufficient decrease parameter
		c2               : curvature parameter
		tolerance_change : minimum allowable step length
		max_ls           : maximum nb of iterations

	Output:
		f             : function value at x+t*d
		g             : gradient value at x+t*d
		t             : the step length
		ls_func_evals : the number of function evaluations
	'''
	batch_dim = d.size(0)
	device    = d.device
	d_norm    = d.abs().max(dim=1)[0]

	x = [p.clone(memory_format=torch.contiguous_format) for p in x]
	g =  g.clone(memory_format=torch.contiguous_format)

	# evaluate objective and gradient using initial step
	f_new, g_new = obj_func(x, t, d)
	ls_func_evals = 1
	gtd_new = (g_new*d).sum(dim=1)

	# bracket an interval containing a point satisfying the Wolfe criteria
	t_prev, f_prev, g_prev, gtd_prev = torch.zeros_like(t), f, g, gtd
	done      = torch.tensor([False]*batch_dim, dtype=torch.bool, device=device)
	nobracket = torch.tensor([True]*batch_dim,  dtype=torch.bool, device=device)
	bracket_low,     bracket_high     = torch.zeros(batch_dim, dtype=t.dtype,   device=device), torch.zeros(batch_dim, dtype=t.dtype,   device=device)
	bracket_f_low,   bracket_f_high   = torch.zeros(batch_dim, dtype=f.dtype,   device=device), torch.zeros(batch_dim, dtype=f.dtype,   device=device)
	bracket_g_low,   bracket_g_high   = torch.zeros(*g.shape,  dtype=g.dtype,   device=device), torch.zeros(*g.shape,  dtype=g.dtype,   device=device)
	bracket_gtd_low, bracket_gtd_high = torch.zeros(batch_dim, dtype=gtd.dtype, device=device), torch.zeros(batch_dim, dtype=gtd.dtype, device=device)
	ls_iter = 0
	while ls_iter < max_ls:
		# check conditions
		if ls_iter <= 1:
			condition = torch.logical_and( (f_new > (f + c1 * t.squeeze(1) * gtd)), nobracket)
		else:
			condition = torch.logical_and( torch.logical_or( (f_new > (f + c1 * t.squeeze(1) * gtd)), f_new >= f_prev ), nobracket)
		bracket_low[condition],     bracket_high[condition]     = t_prev[condition,0], t[condition,0]
		bracket_f_low[condition],   bracket_f_high[condition]   = f_prev[condition],   f_new[condition]
		bracket_g_low[condition,:], bracket_g_high[condition,:] = g_prev[condition,:], g_new[condition,:]
		bracket_gtd_low[condition], bracket_gtd_high[condition] = gtd_prev[condition], gtd_new[condition]
		nobracket[condition] = False
		if not torch.any(nobracket):
			break

		condition = torch.logical_and( gtd_new.abs() <= -c2 * gtd, nobracket)
		bracket_low[condition],     bracket_high[condition]     = t[condition,0],     t[condition,0]
		bracket_f_low[condition],   bracket_f_high[condition]   = f_new[condition],   f_new[condition]
		bracket_g_low[condition,:], bracket_g_high[condition,:] = g_new[condition,:], g_new[condition,:]
		done[condition] = True
		nobracket[condition] = False
		if not torch.any(nobracket):
			break

		condition = torch.logical_and( gtd_new >= 0, nobracket)
		bracket_low[condition],     bracket_high[condition]     = t_prev[condition,0], t[condition,0]
		bracket_f_low[condition],   bracket_f_high[condition]   = f_prev[condition],   f_new[condition]
		bracket_g_low[condition,:], bracket_g_high[condition,:] = g_prev[condition,:], g_new[condition,:]
		bracket_gtd_low[condition], bracket_gtd_high[condition] = gtd_prev[condition], gtd_new[condition]
		nobracket[condition] = False
		if not torch.any(nobracket):
			break

		# interpolate
		min_step = t + 0.01 * (t - t_prev)
		max_step = t * 10
		tmp = t
		t = t.clone()
		t[nobracket,0] = _batched_cubic_interpolate(
			t_prev.squeeze(1),
			f_prev,
			gtd_prev,
			t.squeeze(1),
			f_new,
			gtd_new,
			bounds=(min_step.squeeze(1), max_step.squeeze(1)))[nobracket]

		# next step
		t_prev = tmp
		f_prev = f_new
		g_prev = g_new
		gtd_prev = gtd_new
		f_new, g_new = obj_func(x, t, d)
		# print(nobracket, t[:,0], f_new)
		ls_func_evals += 1
		gtd_new = (g_new*d).sum(1)
		ls_iter += 1

	# reached max number of iterations?
	if ls_iter == max_ls:
		bracket_low[nobracket],     bracket_high[nobracket]     = torch.zeros_like(bracket_low[nobracket]), t[nobracket,0]
		bracket_f_low[nobracket],   bracket_f_high[nobracket]   = f[nobracket],   f_new[nobracket]
		bracket_g_low[nobracket,:], bracket_g_high[nobracket,:] = g[nobracket,:], g_new[nobracket,:]

	# low_pos  = torch.zeros(batch_dim, dtype=torch.long)
	# high_pos = torch.zeros(batch_dim, dtype=torch.long)
	# condition = (bracket_f_low <= bracket_f_high)
	# low_pos[condition],  low_pos[~condition]  = 0, 1
	# high_pos[condition], high_pos[~condition] = 1, 0

	def swap_low_high(condition):
		bracket_tmp = bracket_low.clone();     bracket_low[condition],     bracket_high[condition]     = bracket_high[condition],     bracket_tmp[condition]
		bracket_tmp = bracket_f_low.clone();   bracket_f_low[condition],   bracket_f_high[condition]   = bracket_f_high[condition],   bracket_tmp[condition]
		bracket_tmp = bracket_g_low.clone();   bracket_g_low[condition,:], bracket_g_high[condition,:] = bracket_g_high[condition,:], bracket_tmp[condition,:]
		bracket_tmp = bracket_gtd_low.clone(); bracket_gtd_low[condition], bracket_gtd_high[condition] = bracket_gtd_high[condition], bracket_tmp[condition]

	# zoom phase: we now have a point satisfying the criteria, or
	# a bracket around it. We refine the bracket until we find the
	# exact point satisfying the criteria
	insuf_progress = torch.tensor([False]*batch_dim, dtype=torch.bool, device=gtd.device)
	# find high and low points in bracket
	swap_low_high(bracket_f_low > bracket_f_high)
	while not torch.all(done) and ls_iter < max_ls:
		# line-search bracket is so small
		proceed = (bracket_high - bracket_low).abs() * d_norm >= tolerance_change
		if torch.any(proceed):
			# compute new trial value
			t = _batched_cubic_interpolate(bracket_low, bracket_f_low, bracket_gtd_low,
										   bracket_high, bracket_f_high, bracket_gtd_high).unsqueeze(1)

			# test that we are making sufficient progress:
			# in case `t` is so close to boundary, we mark that we are making
			# insufficient progress, and if
			#   + we have made insufficient progress in the last step, or
			#   + `t` is at one of the boundary,
			# we will move `t` to a position which is `0.1 * len(bracket)`
			# away from the nearest boundary point.
			bracket_max = torch.maximum(bracket_low, bracket_high)
			bracket_min = torch.minimum(bracket_low, bracket_high)
			eps = 0.1 * (bracket_max - bracket_min)
			condition = torch.minimum( bracket_max - t.squeeze(1), t.squeeze(1) - bracket_min ) < eps
			conditiona = torch.logical_and( proceed,  condition )
			conditionb = torch.logical_and( proceed, ~condition )
			if torch.any(conditiona):
				# interpolation close to boundary
				condition2  = torch.logical_or( insuf_progress, torch.logical_or( t.squeeze(1) >= bracket_max, t.squeeze(1) <= bracket_min ) )
				condition2a = torch.logical_and( conditiona,  condition2 )
				condition2b = torch.logical_and( conditiona, ~condition2 )
				# evaluate at 0.1 away from boundary
				if torch.any(condition2a):
					condition3  = (t.squeeze(1) - bracket_max).abs() < (t.squeeze(1) - bracket_min).abs()
					condition3a = torch.logical_and( condition2a,  condition3 )
					condition3b = torch.logical_and( condition2a, ~condition3 )
					t[condition3a,0] = (bracket_max - eps)[condition3a]
					t[condition3b,0] = (bracket_min + eps)[condition3b]
				insuf_progress[condition2b] = True
			insuf_progress[conditionb] = False

			# Evaluate new point
			f_new, g_new = obj_func(x, t, d)
			ls_func_evals += 1
			gtd_new = (g_new*d).sum(1)
			ls_iter += 1

			condition  = torch.logical_or( f_new > (f + c1 * t.squeeze(1) * gtd), f_new >= bracket_f_low )
			conditiona = torch.logical_and( proceed,  condition )
			conditionb = torch.logical_and( proceed, ~condition )
			if torch.any(conditiona):
				# Armijo condition not satisfied or not lower than lowest point
				bracket_high[conditiona] = t[conditiona,0]
				bracket_f_high[conditiona] = f_new[conditiona]
				bracket_g_high[conditiona,:] = g_new[conditiona,:].clone(memory_format=torch.contiguous_format)
				bracket_gtd_high[conditiona] = gtd_new[conditiona]
				swap_low_high(bracket_f_low > bracket_f_high)
			condition2  = abs(gtd_new) <= -c2 * gtd
			condition2a = torch.logical_and( conditionb,  condition2 )
			condition2b = torch.logical_and( conditionb, ~condition2 )
			# Wolfe conditions satisfied
			done[condition2a] = True
			condition3 = gtd_new * (bracket_high - bracket_low) >= 0
			condition3 = torch.logical_and( condition2b,  condition3 )
			# old high becomes new low
			bracket_high[condition3] = bracket_low[condition3]
			bracket_g_high[condition3,:] = bracket_g_low[condition3,:]
			bracket_gtd_high[condition3] = bracket_gtd_low[condition3]
			# new point becomes new low
			bracket_low[conditionb] = t[conditionb,0]
			bracket_f_low[conditionb] = f_new[conditionb]
			bracket_g_low[conditionb,:] = g_new[conditionb,:].clone(memory_format=torch.contiguous_format)
			bracket_gtd_low[conditionb] = gtd_new[conditionb]
		else:
			break

	# return stuff
	t = bracket_low.unsqueeze(1)
	f_new = bracket_f_low
	g_new = bracket_g_low
	return f_new, g_new, t, ls_func_evals


# TODO: try torch split
def _batched_strong_wolfe(obj_func,
				  t,
				  d,
				  f,
				  g,
				  gtd,
				  c1=1e-4,
				  c2=0.9,
				  tolerance_change=1e-9,
				  max_ls=10):
	'''
	A Line Search satisfying the Wolfe conditions

	Ported from https://github.com/torch/optim/blob/master/lswolfe.lua and https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS

	Args:
		obj_func         : a function (the objective) that takes a single input (x), the point of evaluation, and returns f(x) and df/dx
		t                : initial step size
		d                : descent direction
		f                : initial function value
		g                : gradient at initial location
		gtd              : directional derivative at starting location
		c1               : sufficient decrease parameter
		c2               : curvature parameter
		tolerance_change : minimum allowable step length
		max_ls           : maximum nb of iterations

	Output:
		f             : function value at x+t*d
		g             : gradient value at x+t*d
		t             : the step length
		ls_func_evals : the number of function evaluations
	'''
	batch_dim = d.size(0)
	device    = d.device
	d_norm    = d.abs().max(dim=1)[0]

	g =  g.clone(memory_format=torch.contiguous_format)
	t = t.squeeze(1)

	# evaluate objective and gradient using initial step
	f_new, g_new = obj_func(t); ls_func_evals = 1
	gtd_new = (g_new*d).sum(dim=1)

	#####################################################################################
	# bracket an interval containing a point satisfying the strong Wolfe criteria
	#####################################################################################
	t_prev, f_prev, g_prev, gtd_prev = torch.zeros_like(t), f, g, gtd

	# mask indicating which batch dimensions satisfy Wolfe criteria
	done = torch.tensor([False]*batch_dim, dtype=torch.bool, device=device)
	# mask indicating which batch dimensions should be bracketed
	nobracket = torch.tensor([True]*batch_dim,  dtype=torch.bool, device=device)

	# need to create buffers for bathched brackets, can this be avoided?
	bracket     = [torch.zeros(batch_dim, dtype=t.dtype,   device=device), torch.zeros(batch_dim, dtype=t.dtype,   device=device)]
	bracket_f   = [torch.zeros(batch_dim, dtype=f.dtype,   device=device), torch.zeros(batch_dim, dtype=f.dtype,   device=device)]
	bracket_g   = [torch.zeros(*g.shape,  dtype=g.dtype,   device=device), torch.zeros(*g.shape,  dtype=g.dtype,   device=device)]
	bracket_gtd = [torch.zeros(batch_dim, dtype=gtd.dtype, device=device), torch.zeros(batch_dim, dtype=gtd.dtype, device=device)]
	ls_iter = 0
	while ls_iter < max_ls:
		# check conditions
		condition = f_new > (f + c1 * t * gtd)
		if ls_iter <= 1:
			condition = torch.logical_and( condition, nobracket)
		else:
			condition = torch.logical_and( torch.logical_or( condition, f_new >= f_prev ), nobracket)
		bracket[0][condition],     bracket[1][condition]     = t_prev[condition],   t[condition]
		bracket_f[0][condition],   bracket_f[1][condition]   = f_prev[condition],   f_new[condition]
		bracket_g[0][condition,:], bracket_g[1][condition,:] = g_prev[condition,:], g_new[condition,:]
		bracket_gtd[0][condition], bracket_gtd[1][condition] = gtd_prev[condition], gtd_new[condition]
		# batch dimensions satisfying 'condition' have been bracketed
		nobracket[condition] = False
		if not torch.any(nobracket):
			break

		# condition = torch.logical_and( gtd_new.abs() <= -c2 * gtd, nobracket)
		condition = torch.logical_and( gtd_new.abs() <= c2 * gtd.abs(), nobracket)
		bracket[0][condition],     bracket[1][condition]     = t[condition],       t[condition]
		bracket_f[0][condition],   bracket_f[1][condition]   = f_new[condition],   f_new[condition]
		bracket_g[0][condition,:], bracket_g[1][condition,:] = g_new[condition,:], g_new[condition,:]
		# batch dimensions satisfying 'condition' satisfy Wolfe criteria, no need to bracket them
		done[condition] = True
		nobracket[condition] = False
		if not torch.any(nobracket):
			break

		condition = torch.logical_and( gtd_new >= 0, nobracket)
		bracket[0][condition],     bracket[1][condition]     = t_prev[condition],   t[condition]
		bracket_f[0][condition],   bracket_f[1][condition]   = f_prev[condition],   f_new[condition]
		bracket_g[0][condition,:], bracket_g[1][condition,:] = g_prev[condition,:], g_new[condition,:]
		bracket_gtd[0][condition], bracket_gtd[1][condition] = gtd_prev[condition], gtd_new[condition]
		# batch dimensions satisfying 'condition' have been bracketed
		nobracket[condition] = False
		if not torch.any(nobracket):
			break

		# interpolate non-bracketed dimensions
		min_step = t + 0.01 * (t - t_prev)
		max_step = t * 10
		tmp = t.clone() # so that tmp and t point to different objects, and tmp is not modified
		t[nobracket] = _batched_cubic_interpolate(  t_prev, f_prev, gtd_prev,
													t,      f_new,  gtd_new,
													bounds=(min_step, max_step) )[nobracket]

		# next step
		t_prev   = tmp
		f_prev   = f_new
		g_prev   = g_new
		gtd_prev = gtd_new
		f_new, g_new = obj_func(t); ls_func_evals += 1
		gtd_new = (g_new*d).sum(1)
		ls_iter += 1

	# reached max number of iterations?
	if ls_iter == max_ls:
		bracket[0][nobracket],     bracket[1][nobracket]     = 0,              t[nobracket,0]
		bracket_f[0][nobracket],   bracket_f[1][nobracket]   = f[nobracket],   f_new[nobracket]
		bracket_g[0][nobracket,:], bracket_g[1][nobracket,:] = g[nobracket,:], g_new[nobracket,:]


	#####################################################################################
	# zoom phase: we now have a point satisfying the criteria, or a bracket around it.
	# We refine the bracket until we find the exact point satisfying the criteria
	#####################################################################################
	def swap_low_high(condition):
		bracket_tmp = bracket[0].clone();     bracket[0][condition],     bracket[1][condition]     = bracket[1][condition],     bracket_tmp[condition]
		bracket_tmp = bracket_f[0].clone();   bracket_f[0][condition],   bracket_f[1][condition]   = bracket_f[1][condition],   bracket_tmp[condition]
		bracket_tmp = bracket_g[0].clone();   bracket_g[0][condition,:], bracket_g[1][condition,:] = bracket_g[1][condition,:], bracket_tmp[condition,:]
		bracket_tmp = bracket_gtd[0].clone(); bracket_gtd[0][condition], bracket_gtd[1][condition] = bracket_gtd[1][condition], bracket_tmp[condition]

	swap_low_high(bracket_f[0] > bracket_f[1])

	insuf_progress = torch.tensor([False]*batch_dim, dtype=torch.bool, device=gtd.device)
	while not torch.all(done) and ls_iter < max_ls:
		# line-search bracket is so small
		proceed = (bracket[1] - bracket[0]).abs() * d_norm >= tolerance_change
		if torch.any(proceed):
			# compute new trial value
			t = _batched_cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
										   bracket[1], bracket_f[1], bracket_gtd[1])

			# test that we are making sufficient progress:
			# in case `t` is so close to boundary, we mark that we are making insufficient progress, and if
			#   + we have made insufficient progress in the last step, or
			#   + `t` is at one of the boundary,
			# we will move `t` to a position which is `0.1 * len(bracket)` away from the nearest boundary point.
			bracket_max = torch.maximum(bracket[0], bracket[1])
			bracket_min = torch.minimum(bracket[0], bracket[1])
			eps = 0.1 * (bracket_max - bracket_min)
			condition  = torch.minimum( bracket_max - t, t - bracket_min ) < eps
			conditiona = torch.logical_and( proceed,  condition )
			conditionb = torch.logical_and( proceed, ~condition )
			if torch.any(conditiona):
				# interpolation close to boundary
				condition2  = torch.logical_or( insuf_progress, torch.logical_or( t >= bracket_max, t <= bracket_min ) )
				condition2a = torch.logical_and( conditiona,  condition2 )
				condition2b = torch.logical_and( conditiona, ~condition2 )
				# evaluate at 0.1 away from boundary
				if torch.any(condition2a):
					condition3  = (t - bracket_max).abs() < (t - bracket_min).abs()
					condition3a = torch.logical_and( condition2a,  condition3 )
					condition3b = torch.logical_and( condition2a, ~condition3 )
					t[condition3a] = (bracket_max - eps)[condition3a]
					t[condition3b] = (bracket_min + eps)[condition3b]
				insuf_progress[condition2a] = False
				insuf_progress[condition2b] = True
			insuf_progress[conditionb] = False

			# Evaluate new point
			f_new, g_new = obj_func(t); ls_func_evals += 1
			gtd_new = (g_new*d).sum(1)
			ls_iter += 1

			condition  = torch.logical_or( f_new > (f + c1 * t * gtd), f_new >= bracket_f[0] )
			conditiona = torch.logical_and( proceed,  condition )
			conditionb = torch.logical_and( proceed, ~condition )
			if torch.any(conditiona):
				# Armijo condition not satisfied or not lower than lowest point
				bracket[1][conditiona]     = t[conditiona]
				bracket_f[1][conditiona]   = f_new[conditiona]
				bracket_g[1][conditiona,:] = g_new[conditiona,:].clone(memory_format=torch.contiguous_format)
				bracket_gtd[1][conditiona] = gtd_new[conditiona]
				swap_low_high(bracket_f[0] > bracket_f[1])
			condition2  = abs(gtd_new) <= -c2 * gtd
			condition2a = torch.logical_and( conditionb,  condition2 )
			condition2b = torch.logical_and( conditionb, ~condition2 )
			# Wolfe conditions satisfied
			done[condition2a] = True
			condition3 = gtd_new * (bracket[1] - bracket[0]) >= 0
			condition3 = torch.logical_and( condition2b,  condition3 )
			# old high becomes new low
			bracket[1][condition3]     = bracket[0][condition3]
			bracket_f[1][condition3] = bracket_f[0][condition3]
			bracket_g[1][condition3,:] = bracket_g[0][condition3,:]
			bracket_gtd[1][condition3] = bracket_gtd[0][condition3]
			# new point becomes new low
			bracket[0][conditionb]     = t[conditionb]
			bracket_f[0][conditionb]   = f_new[conditionb]
			bracket_g[0][conditionb,:] = g_new[conditionb,:].clone(memory_format=torch.contiguous_format)
			bracket_gtd[0][conditionb] = gtd_new[conditionb]
		else:
			break

	# return stuff
	t     = bracket[0].unsqueeze(1)
	f_new = bracket_f[0]
	g_new = bracket_g[0]
	return f_new, g_new, t, ls_func_evals



class batched_LBFGS(Optimizer):
	"""
	Implements batched L-BFGS algorithm

	Inspired by `minFunc <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`
	Adapted from https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS

	.. warning::
		This optimizer doesn't support per-parameter options and parameter groups (there can be only one).

	.. warning::
		Right now all parameters have to be on a single device.

	.. note::
		This is a very memory intensive optimizer (it requires additional ``param_bytes * (history_size + 1)`` bytes).
		If it doesn't fit in memory try reducing the history size

	Args:
		lr (float): learning rate (default: 1)
		tol (float): termination tolerance on objective function value
		max_iter (int): maximal number of iterations per optimization step (default: 20)
		max_eval (int): maximal number of function evaluations per optimization step (default: max_iter * 1.25).
		tolerance_grad (float): termination tolerance on first order optimality (default: 1e-7).
		tolerance_change (float): termination tolerance on function value/parameter changes (default: 1e-9).
		history_size (int): update history size (default: 100).
		line_search_fn (str): either 'strong_wolfe' or None (default: None).
	"""

	def __init__(self,
				 params,
				 lr=1,
				 tol=1.e-6,
				 max_iter=20,
				 max_eval=None,
				 tolerance_grad=1e-8,
				 tolerance_change=1e-9,
				 history_size=100,
				 line_search_fn=None):
		if max_eval is None: max_eval = max_iter * 5 // 4
		defaults = dict(
			lr=lr,
			tol=tol,
			max_iter=max_iter,
			max_eval=max_eval,
			tolerance_grad=tolerance_grad,
			tolerance_change=tolerance_change,
			history_size=history_size,
			line_search_fn=line_search_fn)
		super(batched_LBFGS, self).__init__(params, defaults)

		if len(self.param_groups) != 1:
			raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")

		self._params = self.param_groups[0]['params']
		self._numel_cache = None

	def _numel(self):
		if self._numel_cache is None:
			self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
		return self._numel_cache

	def _gather_flat_grad(self):
		views = []
		for p in self._params:
			batch_dim = p.size(0)
			if p.grad is None:
				# view = p.new(p.numel()).zero_()
				view = p.new(batch_dim,p.numel()//batch_dim).zero_()
			elif p.grad.is_sparse:
				# view = p.grad.to_dense().view(-1)
				view = p.grad.to_dense().view(batch_dim,-1)
			else:
				# view = p.grad.view(-1)
				view = p.grad.view(batch_dim,-1)
			views.append(view)
		return torch.cat(views, 0)

	def _add_grad(self, step_size, update):
		offset = 0
		# for p in self._params:
		#     numel = p.numel()
		#     # view as to avoid deprecated pointwise semantics
		#     p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
		#     offset += numel
		# assert offset == self._numel()
		p = self._params[0]
		p += (step_size * update).view_as(p)

	def _clone_param(self):
		return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

	def _set_param(self, params_data):
		for p, pdata in zip(self._params, params_data):
			p.copy_(pdata)

	def _directional_evaluate(self, closure, x, t, d):
	    self._add_grad(t, d)
	    loss = closure().float()
	    flat_grad = self._gather_flat_grad()
	    self._set_param(x)
	    return loss, flat_grad

	@torch.no_grad()
	def step(self, closure):
		"""
		Performs a single optimization step.

		Args:
			closure (callable): A closure that reevaluates the model and returns the loss.
		"""
		assert len(self.param_groups) == 1

		# Make sure the closure is always called with grad enabled
		closure = torch.enable_grad()(closure)

		# options
		group = self.param_groups[0]
		lr               = group['lr']
		tol              = group['tol']
		max_iter         = group['max_iter']
		max_eval         = group['max_eval']
		tolerance_grad   = group['tolerance_grad']
		tolerance_change = group['tolerance_change']
		line_search_fn   = group['line_search_fn']
		history_size     = group['history_size']

		# NOTE: LBFGS has only global state, but we register it as state for
		# the first param, because this helps with casting in load_state_dict
		state = self.state[self._params[0]]
		state.setdefault('func_evals', 0)
		state.setdefault('n_iter', 0)

		# evaluate initial f(x) and df/dx
		loss      = closure().float()
		flat_grad = self._gather_flat_grad()
		state['func_evals'] += 1
		#current_evals = 1

		# optimal condition
		if loss.max()<=tol or flat_grad.abs().max()<=tolerance_grad:
			return loss

		batch_dim = loss.size(0)

		# tensors cached in state (for tracing)
		d              = state.get('d')
		t              = state.get('t')
		old_dirs       = state.get('old_dirs')
		old_stps       = state.get('old_stps')
		ro             = state.get('ro')
		H_diag         = state.get('H_diag')
		prev_flat_grad = state.get('prev_flat_grad')
		prev_loss      = state.get('prev_loss')


		# n_iter = 0
		# optimize for a max of max_iter iterations
		while state['n_iter'] < max_iter and loss.max()>tol:
			# n_iter += 1
			state['n_iter'] += 1

			############################################################
			# compute gradient descent direction
			############################################################
			if state['n_iter'] == 1:
				d = flat_grad.neg()
				old_dirs = []
				old_stps = []
				ro = []
				H_diag = torch.ones(batch_dim,1)
				# H_diag = 1
			else:
				# do lbfgs update (update memory)
				y = flat_grad.sub(prev_flat_grad) # g - g_old
				s = d.mul(t)                      # d*t
				# ys = y.dot(s)  # y*s
				# s = d*t
				ys = (y*s).sum(1,keepdim=True)    # y*s, curvature condition for the approximate Hessian to be positive definite
				# print(ys.shape, ys.max())
				# exit()
				# if ys > 1e-10:
				# if ys.max() > 1e-10:
				# Checking worst-case condition
				if ys.min() > 1e-10:
					# updating memory
					if len(old_dirs) == history_size:
						# shift history by one (limited-memory)
						old_dirs.pop(0)
						old_stps.pop(0)
						ro.pop(0)

					# store new direction/step
					old_dirs.append(y)
					old_stps.append(s)
					ro.append(1./ys)
					# ro.append(1. / (ys+1.e-12))

					# update scale of initial Hessian approximation
					# H_diag = ys / y.dot(y)  # (y*y)
					# H_diag = ys / ((y*y).sum(1,keepdim=True)+1.e-12)  # (y*y)
					H_diag = ys / (y*y).sum(1,keepdim=True)

				# compute the approximate (L-BFGS) inverse Hessian multiplied by the gradient
				num_old = len(old_dirs)

				if 'al' not in state:
					state['al'] = [None] * history_size
				al = state['al']

				# iteration in L-BFGS loop collapsed to use just one buffer
				q = flat_grad.neg()
				for i in range(num_old - 1, -1, -1):
					# al[i] = old_stps[i].dot(q) * ro[i]
					# q.add_(old_dirs[i], alpha=-al[i])
					al[i] = (old_stps[i]*q).sum(1,keepdim=True) * ro[i]
					q -= al[i]*old_dirs[i]
					# print(q[-2])
				# multiply by initial Hessian
				# r = d is the final direction
				# d = r = torch.mul(q, H_diag)
				d = q * H_diag
				for i in range(num_old):
					# be_i = old_dirs[i].dot(r) * ro[i]
					# r.add_(old_stps[i], alpha=al[i] - be_i)
					be_i = (old_dirs[i]*d).sum(1,keepdim=True) * ro[i]
					d += (al[i]-be_i)*old_stps[i]
			if prev_flat_grad is None:
				prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
			else:
				prev_flat_grad.copy_(flat_grad)
			prev_loss = loss

			############################################################
			# compute step length
			############################################################
			# reset initial guess for step size
			if state['n_iter'] == 1:
				# t = min(1., 1. / flat_grad.abs().sum()) * lr
				# t = min(1., 1. / flat_grad.abs().sum()) * lr * torch.ones_like(H_diag)
				t = torch.minimum(torch.tensor(1., device=flat_grad.device), 1. / flat_grad.abs().sum(1,keepdim=True)) * lr
			else:
				# t = lr
				t = lr * torch.ones_like(H_diag)

			# directional derivative
			# gtd = flat_grad.dot(d)  # g * d
			gtd = (flat_grad*d).sum(1)  # g * d
			# print(flat_grad[0])
			# print(d[0])
			# print(gtd[0])
			# exit()

			# directional derivative is below tolerance
			# if gtd > -tolerance_change:
			# print(gtd.abs().mean(),d.abs().sum(1).mean())
			if gtd.min() > -tolerance_change:
				break

			# optional line search: user function
			ls_func_evals = 0
			if line_search_fn is not None:
				# perform line search, using user function
				if line_search_fn != "strong_wolfe":
					raise RuntimeError("only 'strong_wolfe' is supported")
				else:
					x_init = self._clone_param()
					# x_init = self._params

					# for i in range(batch_dim):
					#     def obj_func(x, t, d):
					#         # self._add_grad(t, d)
					#         p = self._params[0][i]
					#         p += (t * d).view_as(p)
					#         loss = closure().float()[i]
					#         flat_grad = self._gather_flat_grad()[i]
					#         self._params[0][i].copy_(x)
					#         return loss, flat_grad
					#         # return self._directional_evaluate(closure, x, t, d)

					#     lossi, flat_gradi, ti, ls_func_evals = _strong_wolfe(obj_func, x_init[0][i], t[i][0], d[i], loss[i], flat_grad[i], gtd[i])
					#     t[i][0] = ti
					#     # loss[i] = lossi
					#     # p = self._params[0][i]
					#     # # print(ti,d[i])
					#     # p += (ti * d[i]).view_as(p)
					# print(t)
					# exit()

					# def obj_func_old(x, t, d):
					# 	self._add_grad(t, d)
					# 	# p = self._params[0]
					# 	# p += (t * d).view_as(p)
					# 	loss = closure().float()
					# 	flat_grad = self._gather_flat_grad()
					# 	self._set_param(x)
					# 	return loss, flat_grad
					# 	# return self._directional_evaluate(closure, x, t, d)

					def obj_func(t):
						return self._directional_evaluate(closure, x_init, t.unsqueeze(1), d)

					loss, flat_grad, t, ls_func_evals = _batched_strong_wolfe(obj_func, t, d, loss, flat_grad, gtd)
					# loss, flat_grad, t, ls_func_evals = _batched_strong_wolfe_old(obj_func_old, x_init, t, d, loss, flat_grad, gtd)
					# print(t)
					# exit()

					# def obj_func(x, t, d):
					#     # self._add_grad(t, d)
					#     p = self._params[0]
					#     p += (t * d).view_as(p)
					#     loss, flat_grad = closure().float().sum(), self._gather_flat_grad().view(-1)
					#     self._set_param(x)
					#     return loss, flat_grad
					#     # return self._directional_evaluate(closure, x, t, d)

					# loss, flat_grad, t, ls_func_evals = _strong_wolfe(obj_func, x_init, t, d.view(-1), loss.sum(), flat_grad.view(-1), gtd.sum())
				# print(t)
				# exit()
				self._add_grad(t, d)
				# with torch.enable_grad():
				#     loss = closure().float()
				# flat_grad = self._gather_flat_grad()
				# ls_func_evals += 1
				# opt_cond = flat_grad.abs().max() <= tolerance_grad
			else:
				# no line search, simply move with fixed-step
				self._add_grad(t, d)
				if state['n_iter'] != max_iter:
					# re-evaluate function only if not in last iteration
					# the reason we do this: in a stochastic setting,
					# no use to re-evaluate that function here
					with torch.enable_grad():
						# loss = float(closure())
						loss = closure().float()
					flat_grad = self._gather_flat_grad()
					# opt_cond = flat_grad.abs().max() <= tolerance_grad
					ls_func_evals = 1

			# update func eval
			#current_evals += ls_func_evals
			state['func_evals'] += ls_func_evals

			############################################################
			# check conditions
			############################################################
			if state['n_iter'] == max_iter:
				break

			if state['func_evals'] >= max_eval:
				break

			if flat_grad.abs().max() <= tolerance_grad:
				break

			# print(n_iter, t.view(-1), d.abs().max(), abs(loss - prev_loss).max(), loss.max(), prev_loss.max())
			# print(n_iter, t.view(-1), loss)
			# print(n_iter, current_evals, loss)

			# lack of progress
			# if d.mul(t).abs().max() <= tolerance_change:
			if (d*t).abs().max() <= tolerance_change:
				break

			# if abs(loss - prev_loss) < tolerance_change:
			if (loss - prev_loss).abs().max() < tolerance_change:
				break

			if (loss-prev_loss).max() > 0:
				break

		state['d'] = d
		state['t'] = t
		state['old_dirs'] = old_dirs
		state['old_stps'] = old_stps
		state['ro'] = ro
		state['H_diag'] = H_diag
		state['prev_flat_grad'] = prev_flat_grad
		state['prev_loss'] = prev_loss

		return loss

