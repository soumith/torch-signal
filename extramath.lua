--[[
   -- some extra math functions
]]--

local xmath = {}

function xmath.round(num)
   if num >= 0 then return math.floor(num+.5) 
   else return math.ceil(num-.5) end
end

function xmath.log2(x)
   return math.log(x)/math.log(2)
end
function xmath.nextpow2(x)
   return math.ceil(math.log2(math.abs(x)))   
end
return xmath
