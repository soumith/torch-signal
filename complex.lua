--[[
   All functions in here expect either a 2D Nx2 Complex tensor
]]--

local complex = {}

function complex.angle(h)
   return torch.atan2(h[{{},2}],h[{{},1}])
end

function complex.exp(h)
   local out = h:clone()
   local real = h[{{},1}]
   local imag = h[{{},2}]
   out[{{},1}] = torch.exp(real):cmul(torch.cos(imag))
   out[{{},2}] = torch.exp(real):cmul(torch.sin(imag));  
   return out
end

function complex.abs(h)
   local hsquare = torch.pow(h,2)
   if h:dim() == 2 and h:size(2) == 2 then
      return torch.sqrt(hsquare[{{},1}] + hsquare[{{},2}])
   elseif h:dim() == 3 and h:size(3) == 2 then
      return torch.sqrt(hsquare[{{},{},1}] + hsquare[{{},{},2}])
   else
      error('unsupported dimensions')
   end
end

function complex.real(h)
   return h[{{},1}]
end

function complex.imag(h)
   return h[{{},2}]
end

function complex.conj(h)
   local out = h:clone()
   out[{{},2}]:mul(-1)
   return out
end

function complex.prod(h)
   local out = torch.ones(1,2):typeAs(h)
   out[1] = h[1]
   for i=2,h:size(1) do
      -- (x1 + iy1) * (x2 + iy2) = (x1x2 - y1y2) + i(x1y2 + y1x2)
      local real = (out[1][1]* h[i][1] - out[1][2] * h[i][2])
      local imag = (out[1][1]* h[i][2] + out[1][2] * h[i][1])
      out[1][1] = real
      out[1][2] = imag
   end
   return out
end

function complex.cmul(a1,b1, noargcheck)
   local a,b
   if noargcheck then
      a=a1; b=b1
   else
      if a1:dim() == 1 then -- assume that imag is 0
	 a = torch.DoubleTensor(a1:size(1), 2):zero()
	 a[{{}, 1}] = a1
      elseif a1:dim() == 2 and a1:size(2) == 2 then
	 a = a1
      else
	 error('Input has to be 1D Tensor of size N (purely real 1D tensor) or ' .. 
		  '2D Tensor of size Nx2 (Complex 1D tensor)')
      end
      if b1:dim() == 1 then -- assume that imag is 0
	 b = torch.DoubleTensor(b1:size(1), 2):zero()
	 b[{{}, 1}] = b1
      elseif b1:dim() == 2 and b1:size(2) == 2 then
	 b = b1
      else
	 error('Input has to be 1D Tensor of size N (purely real 1D tensor) or ' .. 
		  '2D Tensor of size Nx2 (Complex 1D tensor)')
      end
   end
   local c = a:clone():zero()
   a = a:contiguous()
   b = b:contiguous()
   c = c:contiguous()
   local cd = torch.data(c)
   local ad = torch.data(a)
   local bd = torch.data(b)
   for i=0,a:size(1)-1 do
      -- (x1 + iy1) * (x2 + iy2) = (x1x2 - y1y2) + i(x1y2 + y1x2)
      local re = i*2
      local im = i*2 + 1
      cd[re] = (ad[re]* bd[re] - ad[im] * bd[im])
      cd[im] = (ad[re]* bd[im] + ad[im] * bd[re])
   end
   return c
end

function complex.dot(a,b)
   if not(a:dim() == 2 and a:size(2) == 2 and b:dim() == 2 and b:size(2) == 2) then
      error('Inputs have to be 2D Tensor of size Nx2 (complex 1D tensor)')
   end
   if a:size(1) ~= b:size(1) then
      error('Both inputs need to have same number of elements')
   end
   local c = torch.sum(complex.cmul(a,b, true), 1)
   return c
end

function complex.mm(a,b)
   if not(a:dim() == 3 and a:size(3) == 2 and b:dim() == 3 and b:size(3) == 2) then
      error('Inputs have to be 3D Tensor of size NxMx2 (complex 2D tensor)')
   end
   if a:size(2) ~= b:size(1) then
      error('Matrix-Matrix product requires NxM and MxP matrices.')
   end
   local c = torch.zeros(a:size(1), b:size(2), 2):typeAs(a)
   for i=1,c:size(1) do
      for j=1,c:size(2) do
	 c[i][j] = complex.dot(a[{i,{},{}}], b[{{},j,{}}])
	 -- print(c[i][j])
      end
   end
   return c
end

function complex.diag(x)
   if x:dim() == 2 and x:size(2) == 2 then
      local y = torch.zeros(x:size(1), x:size(1), 2)
      y[{{},1}] = torch.diag(x[{{},1}])
      y[{{},2}] = torch.diag(x[{{},2}])
      return y
   elseif  x:dim() == 3 and x:size(3) == 2  then
      local yr = torch.diag(x[{{},{},1}])
      local y = torch.zeros(yr:size(1),2)
      y[{{},1}] = yr
      y[{{},2}] = torch.diag(x[{{},{},2}])
      return y      
   else
      error('Input has to be 2D Tensor of size Nx2 or ' .. 
	       '3D Tensor of size NxMx2 (Complex 2D tensor)')
   end
end

--[[
   Polynomial with specified roots
   
   Function is super unoptimized
]]--
function complex.poly(x)
   local e
   if x:dim() == 2 and x:size(1) == x:size(2) then
      e = torch.eig(x) -- square polynomial
      -- TODO: Strip out infinities in case the eigen values have any
   elseif x:dim() == 1 then
      e = x
   else
      error('Input should be a 1D Tensor or a 2D square tensor')
   end

   -- Expand recursion formula
   local n = e:size(1)
   if x:dim() == 1 then
      local c = torch.zeros(n+1) -- purely real
      c[1] = 1
      for j=1,n do
	 c[{{2,(j+1)}}] = c[{{2,(j+1)}}] - torch.mul(c[{{1,j}}],e[j])
      end
      return c
   else
      local c = torch.zeros(n+1,2) -- complex
      c[1][1] = 1
      for j=1,n do
	 -- c(2:(j+1)) = c(2:(j+1)) - e(j).*c(1:j);
	 c[{{2,(j+1)}, 1}] = c[{{2,(j+1)}, 1}] - torch.mul(c[{{1,j}, 1}],e[j][1])
	 c[{{2,(j+1)}, 2}] = c[{{2,(j+1)}, 2}] - torch.mul(c[{{1,j}, 2}],e[j][2])
      end
      -- The result should be real if the roots are complex conjugates.
      local c1 = torch.sort(e[{{torch.ge(e[{{},2}], 0)},2}])
      local c2 = torch.sort(e[{{torch.le(e[{{},2}], 0)},2}])
      if c1:size(1) == c2:size(1) and torch.eq(c1, c2):sum() == c1:size(1) then
	 c = complex.real(c);
      end
      return c
   end
end

return complex
























