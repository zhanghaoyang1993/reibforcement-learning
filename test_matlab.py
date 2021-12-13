import matlab.engine
import numpy as np

k=np.ones((1,24))
k = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

eng = matlab.engine.start_matlab()
a = matlab.double(k)
state_profit = eng.fun_strategic_bidding_python_try(a)
state_profit = np.array(state_profit[0][0:361])
state = state_profit[0:360]
r = state_profit[360]

print(state_profit)
print(state)
print(r)
