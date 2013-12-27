--[[
   -- adds some extra functions to the math table
]]--

function math.round(num)
   if num >= 0 then return math.floor(num+.5) 
   else return math.ceil(num-.5) end
end

function math.log2(x)
   return math.log(x)/math.log(2)
end
function math.nextpow2(x)
   return math.ceil(math.log2(math.abs(x)))   
end