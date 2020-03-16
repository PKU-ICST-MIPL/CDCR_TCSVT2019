--local THNN = require 'nn.THNN'
local SparseMax, parent = torch.class('SparseMax', 'nn.SoftMax')

function SparseMax:updateOutput(input)
    s = input:size()
	num = s[1]
	dim = s[2]
	output = input:clone():fill(0)
	for i=1,num do
	    v = input:narrow(1,i,1)
		v_sorted = torch.sort(v, true)
		cssv = torch.cumsum(v_sorted, 2) - 1
		ind = torch.range(1, dim):cuda()
		cond = v_sorted - cssv:cdiv(ind)
		tau = dim
		for j=dim,1,-1 do
		    if cond[1][j]>0 then
			    tau = cssv[1][j] / j
				break
			end
		end
		w = torch.clamp(v-tau,0,100)
		output:narrow(1,i,1):copy(w)
	end
    return output
end

function SparseMax:updateGradInput(input, gradOutput)
    s = input:size()
	num = s[1]
	dim = s[2]
	output = input:clone():fill(0)
	out_star = self:updateOutput(input)
	for i=1,num do
	    v = out_star:narrow(1,i,1)
		mask_sum = 0
		supp_sum = 0
		for j=1,dim do
		    if v[1][j]>0 then
			    mask_sum = mask_sum + gradOutput[i][j]
				supp_sum = supp_sum + 1
			end
		end
		masked = mask_sum / supp_sum
		for j=1,dim do
		    if v[1][j]>0 then
			    output[i][j] = gradOutput[i][j] - masked
			end
		end
    end
    return output
end
